"""工具目录索引(离线工具侧)。

每个工具经工具侧分类头得到能力域和动作类型,再由规则推出风险等级,连同向量一起入索引。
在线时按请求侧预测的能力域过滤,再按向量相似度取 Top-k。新 server 接入只需重新跑 add_tools。
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import sparse

from .rules import final_risk, tool_text


@dataclass
class ToolRecord:
    tool_id: str
    server: str
    name: str
    description: str
    domains: list
    actions: list
    risk: str
    confidence: float
    needs_review: bool = False
    examples: list = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)


def tool_id(tool: dict) -> str:
    return f"{tool.get('server', '')}/{tool['name']}"


class ToolCatalog:
    def __init__(self, tool_head, review_threshold: float = 0.6, retrieval_encoder=None):
        """retrieval_encoder:检索索引与查询用的编码器(T-019)。

        为 None 时沿用分类头的编码器,和改前逐字节一致;传入时索引与查询都用它,
        分类头(need / 能力域 / 动作)仍用它自己的编码器。工具侧分类头 `head` 不变。
        """
        self.head = tool_head
        self.encoder = retrieval_encoder if retrieval_encoder is not None else tool_head.encoder
        self.review_threshold = review_threshold
        self.records: list = []
        self._vecs = None

    def classify_tools(self, tools) -> list:
        texts = [tool_text(t) for t in tools]
        outs = self.head.predict(texts, items=tools)
        recs = []
        for t, o in zip(tools, outs):
            # 人工标注优先(例如审核过的工具)
            domains = t.get("domains") or o.domains
            actions = t.get("actions") or o.actions
            risk = final_risk(t, actions, domains)
            recs.append(ToolRecord(
                tool_id=tool_id(t), server=t.get("server", ""), name=t["name"],
                description=t.get("description", ""), domains=list(domains), actions=list(actions),
                risk=risk, confidence=o.confidence,
                needs_review=o.confidence < self.review_threshold or risk == "L3", raw=t))
        return recs

    def add_tools(self, tools, use_labels: bool = False, example_queries: dict = None):
        """example_queries: {tool_id: [示例请求]}。工具描述多为英文、请求多为中文时,
        示例请求是让检索对上号的关键;新 server 接入时用 LLM 为每个工具合成几条。"""
        if not use_labels:
            tools = [{k: v for k, v in t.items() if k not in ("domains", "actions")} for t in tools]
        recs = self.classify_tools(tools)
        example_queries = example_queries or {}
        for r in recs:
            r.examples = list(example_queries.get(r.tool_id, []))
        self.records.extend(recs)
        vecs = self.encoder.encode([self._index_text(r) for r in recs])
        vecs = vecs if sparse.issparse(vecs) else sparse.csr_matrix(vecs)
        self._vecs = vecs if self._vecs is None else sparse.vstack([self._vecs, vecs]).tocsr()
        return recs

    @staticmethod
    def _index_text(r: ToolRecord) -> str:
        return " ".join([r.name.replace("_", " "), r.description] + r.examples)

    def _allowed_domains(self, domains, domain_probs, min_prob):
        want = set(domains or [])
        if domain_probs:
            want |= {d for d, p in domain_probs.items() if p >= min_prob}
        return want

    def retrieve(self, query: str, domains=None, k: int = 5, actions=None, domain_probs=None,
                 min_prob: float = 0.1, prior_weight: float = 0.3, action_bonus: float = 0.05):
        """返回 [(ToolRecord, score)]。

        domains 与 domain_probs 都为空时不过滤(基线)。否则做软过滤:
        候选 = 预测能力域 + 概率不低于 min_prob 的能力域下的工具,避免能力域分错时把正确工具整个过滤掉;
        排序分 = 向量相似度 + prior_weight * 工具所属能力域的最大概率 + 动作匹配加分。
        """
        if not self.records:
            return []
        q = self.encoder.encode([query])
        q = q if sparse.issparse(q) else sparse.csr_matrix(q)
        scores = np.asarray((self._vecs @ q.T).todense()).ravel()
        if domain_probs:
            scores = scores + prior_weight * np.array(
                [max((domain_probs.get(d, 0.0) for d in r.domains), default=0.0) for r in self.records])
        if actions:
            want_a = set(actions)
            scores = scores + action_bonus * np.array([bool(want_a & set(r.actions)) for r in self.records])
        idx = np.arange(len(self.records))
        want = self._allowed_domains(domains, domain_probs, min_prob)
        if want:
            keep = np.array([bool(want & set(r.domains)) for r in self.records])
            if keep.any():
                idx = idx[keep]
        top = idx[np.argsort(-scores[idx])][:k]
        return [(self.records[i], float(scores[i])) for i in top]

    def candidate_count(self, domains, domain_probs=None, min_prob: float = 0.1) -> int:
        want = self._allowed_domains(domains, domain_probs, min_prob)
        if not want:
            return len(self.records)
        n = sum(1 for r in self.records if want & set(r.domains))
        return n or len(self.records)

    def export_jsonl(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                d = asdict(r)
                d.pop("raw")
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
