"""在线路由:一次编码,同时产出原有结果和 MCP 结果。

流程:请求 -> 编码 -> 原有分类头(不变) + MCP 分类头
      -> 触发兜底? 是 -> 问云端 -> 按策略合成结果(见 strategy.apply_strategy)
      -> 按能力域检索工具索引 Top-k -> 风险为 L3? 是 -> 标记需用户确认
      -> 交给 LLM Agent 调用
"""

import time
from dataclasses import dataclass, field

from .heads import HeadOutput
from .strategy import STRATEGIES, TRIGGERS, apply_strategy

# 结果被兜底改写过的来源:这时模型自己的类别概率不再可信
FALLBACK_SOURCES = frozenset({"llm_fallback", "jev_fallback", "haiku_fallback",
                              "codebuddy_fallback"})


def dedupe_by_name(hits, k):
    """按函数名去重:每个 `ToolRecord.name` 只留分数最高的变体,再取前 k。

    `hits` 是 `[(ToolRecord, score)]`,已按分数降序(见 `ToolCatalog.retrieve`),
    所以每个名字第一次出现的就是它分数最高的那个变体。去重逻辑放在这里,
    评估脚本(`tools_real._retrieve_ranked`)和在线路由共用同一份实现。
    """
    seen = set()
    out = []
    for rec, score in hits:
        if rec.name in seen:
            continue
        seen.add(rec.name)
        out.append((rec, score))
        if len(out) >= k:
            break
    return out


@dataclass
class RouteResult:
    text: str
    original: object = None          # 原有分类头的输出,原样透传
    need: str = "none"
    domains: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    confidence: float = 1.0
    source: str = "model"            # model / llm_fallback / jev_fallback / model_low_conf
    candidates: list = field(default_factory=list)   # [{tool_id, score, risk, domains}]
    requires_confirmation: bool = False
    soft_candidates: bool = False    # 软 need:need 判成 none 但低置信,候选来自软触发(T-020)
    latency_ms: float = 0.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class Router:
    def __init__(self, request_head, catalog, threshold: float = 0.55, top_k: int = 5,
                 original_head=None, llm_fallback=None,
                 fallback_strategy: str = "replace", fallback_trigger: str = "low_conf",
                 use_filter: bool = True, dedupe_by_name: bool = False,
                 soft_need_threshold: float | None = None, reranker=None):
        """
        original_head: 可选,原方案的分类函数 f(text) -> 任意结果,结果原样透传。
        llm_fallback:  可选,f(text) -> dict(need, domains, actions);触发时调用。
                       不传时行为不变:本地结果原样输出,不会去问任何人。
        threshold:     置信度阈值,可以是 float,也可以是 {domain: float} 的按类阈值。
        fallback_strategy: 兜底答案与本地结果怎么合成,见 `strategy.STRATEGIES`。
        fallback_trigger:  什么时候去问,low_conf = 置信度低于阈值,always = 每条都问。
        use_filter:    是否按能力域软过滤候选(默认 True,保持旧行为)。
                       传 False 时 retrieve 收到 domains=None,即真实目录下的推荐配置,
                       依据 `reports/eval_t011.md` 2.2(真实目录下软过滤不划算)。
        dedupe_by_name: 是否按函数名去重(默认 False,保持旧行为)。
                       真实目录里同一个函数名最多有 48 个变体(`tool_id` 靠 `#N` 区分),
                       top-k 容易被一个函数占满。打开后先多取一些候选(`top_k * 10`,最多取满
                       目录),每个函数名只留分数最高的那个变体,再取 top_k。
        soft_need_threshold: 软 need(T-020)。为 None 时行为与改前完全一样。设了值时,
                       若 `need == "none"` 且 need 头给 none 的概率**低于**该阈值,仍照常检索并
                       返回 top_k 个候选,但 `need` 字段保持 `"none"`,只把 `soft_candidates`
                       标成 True,由下游 agent 自己决定用不用。概率取请求头的 `need_probs["none"]`,
                       拿不到概率时不触发。1.01 相当于"判成 none 就一律附带候选"。
        reranker:      语义重排器(T-025,可选)。为 None 时行为与改前完全一样。设了值时,
                       检索先取 `max(k, reranker.rerank_n)` 条 tfidf 召回,再交给重排器排序,
                       最后照旧去重 / 取 top_k。重排每步耗时记在 `reranker.last_ms`。
        """
        self.head = request_head
        self.catalog = catalog
        self.threshold = threshold
        self.top_k = top_k
        self.use_filter = use_filter
        self.dedupe_by_name = dedupe_by_name
        self.soft_need_threshold = soft_need_threshold
        self.reranker = reranker
        self.original_head = original_head
        self.llm_fallback = llm_fallback
        if fallback_strategy not in STRATEGIES:
            raise ValueError(f"兜底策略只能是 {STRATEGIES} 之一,收到 {fallback_strategy!r}")
        if fallback_trigger not in TRIGGERS:
            raise ValueError(f"触发方式只能是 {TRIGGERS} 之一,收到 {fallback_trigger!r}")
        self.fallback_strategy = fallback_strategy
        self.fallback_trigger = fallback_trigger

    def _threshold_for(self, out: HeadOutput) -> float:
        if isinstance(self.threshold, dict):
            vals = [self.threshold.get(d, self.threshold.get("default", 0.55)) for d in out.domains]
            return max(vals) if vals else self.threshold.get("default", 0.55)
        return self.threshold

    def route(self, text: str) -> RouteResult:
        t0 = time.perf_counter()
        out = self.head.predict([text])[0]
        res = RouteResult(text=text, need=out.need, domains=out.domains, actions=out.actions,
                          confidence=out.confidence)
        if self.original_head is not None:
            res.original = self.original_head(text)

        # 触发方式决定要不要去问;策略决定问回来的答案怎么用(见 strategy.apply_strategy)
        triggered = self.fallback_trigger == "always" or out.confidence < self._threshold_for(out)
        if triggered and self.fallback_strategy != "local":
            if self.llm_fallback is None:
                res.source = "model_low_conf"      # 没配兜底,置信度也不够
            else:
                labels = None
                try:
                    labels = self.llm_fallback(text)
                except Exception:
                    # 兜底挂了(超时、限流)就留着本地结果,不向上抛,但来源要标成出错
                    res.source = getattr(self.llm_fallback, "error_source", "fallback_error")
                if labels is not None:
                    merged = apply_strategy(out, labels, self.fallback_strategy)
                    res.need, res.domains, res.actions = (merged["need"], merged["domains"],
                                                          merged["actions"])
                    # 兜底函数自带来源标记(Jev 是 jev_fallback,Haiku 是 haiku_fallback)
                    res.source = getattr(self.llm_fallback, "source", "llm_fallback")
                elif res.source == "model":
                    res.source = "model_low_conf"  # 兜底没抛异常但也没给答案

        # 软 need(T-020):need 判成 none,但 need 头给 none 的概率低于阈值时,照常检索并挂候选。
        # need 字段不变,只把 soft_candidates 置 True,由下游决定用不用;拿不到概率时不触发。
        soft = False
        if self.soft_need_threshold is not None and res.need == "none":
            p_none = out.need_probs.get("none") if out.need_probs else None
            if p_none is not None and p_none < self.soft_need_threshold:
                soft = True
                res.soft_candidates = True

        if res.need != "none" or soft:
            # 去重时先多取一些候选,给被同名变体挤掉的函数留出位置;
            # 开重排(T-025)时池子至少取 rerank_n 条,tfidf 先召回这么多再重排
            rr = getattr(self, "reranker", None)   # 旧 router.pkl 没有这个属性
            k = min(self.top_k * 10, len(self.catalog.records)) if self.dedupe_by_name else self.top_k
            n = k if rr is None else max(k, rr.rerank_n)
            if self.use_filter:
                # 兜底改写过类别时不再用模型概率
                probs = out.domain_probs if res.source not in FALLBACK_SOURCES else None
                hits = self.catalog.retrieve(text, res.domains, n, actions=res.actions,
                                             domain_probs=probs)
            else:
                # 软过滤关:不传能力域,也不按概率/动作加分,直接按向量相似度取 top-n
                hits = self.catalog.retrieve(text, None, n)
            if rr is not None:
                hits = rr.rerank(text, hits, self.catalog)
            if self.dedupe_by_name:
                hits = dedupe_by_name(hits, self.top_k)
            hits = hits[:self.top_k]   # 重排池可能比 top_k 大,截回 top_k(不开重排时恰好等长)
            res.candidates = [{"tool_id": r.tool_id, "score": round(s, 4), "risk": r.risk,
                               "domains": r.domains} for r, s in hits]
            # Top-1 是 L3,或请求本身就是 L3 动作时,调用前需用户确认
            res.requires_confirmation = bool(
                (res.candidates and res.candidates[0]["risk"] == "L3")
                or set(res.actions) & {"delete", "execute", "send"})
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res
