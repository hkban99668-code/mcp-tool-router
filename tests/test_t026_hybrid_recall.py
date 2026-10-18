"""T-026:混合召回(tfidf Top-50 ∪ 语义 Top-M)+ 融合重排。

- 真实 `SemanticReranker` 用假编码器桩(不依赖语义模型 / 本地 HF 缓存):
  并集把 tfidf 召回池外的工具补进来,补进来的工具 tfidf 分记 0;`recall_m=0` 时行为与 T-025 一致。
- `t026_eval` / `t026_ceiling` 用假重排器工厂跑通全流程:候选齐全、对照读 fake json、
  测试折只报一次、报告与 json 都出得来。
"""

import json
import pickle
import pathlib

import numpy as np
import pytest

from mcpcls import evaluate as ev
from mcpcls import rerank as rr_mod
from mcpcls import tools_real as tr
from mcpcls.catalog import ToolCatalog
from mcpcls.rerank import SemanticReranker


# ---------------------------------------------------------------- 假编码器 / 假重排器

class FakeST:
    """假的 SentenceTransformer:按预置的 文本 -> 向量 表编码,没见过的文本给同维零向量。"""

    def __init__(self, table):
        self.table = table
        self.dim = len(next(iter(table.values()))) if table else 4

    def encode(self, texts, normalize_embeddings=False):
        zero = np.zeros(self.dim, dtype=np.float32)
        return np.asarray([self.table.get(t, zero) for t in texts], dtype=np.float32)


def _use_fake_model(monkeypatch, table):
    """让 SemanticReranker 不碰真模型:清空进程级模型缓存,把加载函数换成假模型。"""
    monkeypatch.setattr(rr_mod, "_MODEL_CACHE", {})
    monkeypatch.setattr(rr_mod, "_load_st_model", lambda name: FakeST(table))


# ---------------------------------------------------------------- 小型系统(合成目录)

def _tool(tid, name, desc):
    return {"tool_id": tid, "name": name, "server": "Svc", "description": desc,
            "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}


def _synth_tools():
    return [dict(_tool("synth/send", "send_email", "Send an email."),
                 domains=["comms"], actions=["send"]),
            dict(_tool("synth/ls", "list_files", "List files."),
                 domains=["file"], actions=["read"]),
            dict(_tool("Svc/get_weather", "get_weather", "Get the weather."),
                 domains=["system_misc"], actions=["read"])]


def _train():
    return [{"text": "给张三发邮件", "need": "single", "domains": ["comms"], "actions": ["send"],
             "tool_ids": ["synth/send"]},
            {"text": "列出全部文件", "need": "single", "domains": ["file"], "actions": ["read"],
             "tool_ids": ["synth/ls"]},
            {"text": "今天天气怎么样", "need": "single", "domains": ["system_misc"],
             "actions": ["read"], "tool_ids": ["Svc/get_weather"]},
            {"text": "你好呀", "need": "none", "domains": [], "actions": [], "tool_ids": []}]


def _router(reranker=None, top_k=5):
    return ev.build_system("tfidf", _synth_tools(), _train(), top_k=top_k, reranker=reranker)


def _vec_table(cat, query="查天气"):
    """按目录里**实际的索引文本**(含示例查询)建表:查询最贴近"天气"工具。"""
    t = {query: np.array([0.0, 1.0], dtype=np.float32)}
    for rec in cat.records:
        key = ToolCatalog._index_text(rec)
        if "weather" in rec.name:
            t[key] = np.array([0.0, 1.0], dtype=np.float32)
        elif "send" in rec.name:
            t[key] = np.array([1.0, 0.0], dtype=np.float32)
        else:
            t[key] = np.array([0.5, 0.0], dtype=np.float32)
    return t


# ---------------------------------------------------------------- recall_m 并集召回

def test_expand_pool_adds_semantic_top_with_zero_tfidf(monkeypatch, tmp_path):
    """并集把语义 Top-M 里不在 tfidf 池里的工具补进来,补进来的 tfidf 分记 0。"""
    cat = _router().catalog
    _use_fake_model(monkeypatch, _vec_table(cat))
    r = SemanticReranker("fake/m", rerank_n=2, fusion="linear", alpha=0.5, recall_m=1,
                         cache_dir=tmp_path)
    r.warm(cat)
    # 手工造一个不含天气工具的 tfidf 池,模拟"tfidf 没召回到、语义召回到"
    hits = [(rec, 0.7) for rec in cat.records if rec.tool_id != "Svc/get_weather"]
    q = r._query_vec("查天气")
    pool = r._expand_pool(list(hits), q, cat)
    extra = pool[len(hits):]
    assert [rec.tool_id for rec, _ in extra] == ["Svc/get_weather"]   # 语义 Top-1 补进来
    assert all(s == 0.0 for _rec, s in extra)                        # 池外工具 tfidf 分记 0


def test_rerank_recall_m_brings_outside_tool(monkeypatch, tmp_path):
    """recall_m>0 时池外工具靠语义进候选;recall_m=0(默认)时进不来。"""
    cat = _router().catalog
    _use_fake_model(monkeypatch, _vec_table(cat))
    hits = [(rec, 0.7) for rec in cat.records if rec.tool_id != "Svc/get_weather"]
    r1 = SemanticReranker("fake/m", rerank_n=2, fusion="linear", alpha=0.0, recall_m=1,
                          cache_dir=tmp_path)
    ids1 = [rec.tool_id for rec, _ in r1.rerank("查天气", hits, cat)]
    assert ids1[0] == "Svc/get_weather"                     # 纯语义,池外天气排第一
    r0 = SemanticReranker("fake/m", rerank_n=2, fusion="linear", alpha=0.0, cache_dir=tmp_path)
    ids0 = [rec.tool_id for rec, _ in r0.rerank("查天气", hits, cat)]
    assert "Svc/get_weather" not in ids0                    # 不传开关 = 与 T-025 一致,池外够不着


def test_recall_m_zero_keeps_pool_and_order(monkeypatch, tmp_path):
    """recall_m=0(默认):只在 tfidf 池内重排,池子大小与成员都不变(与 T-025 一致)。"""
    cat = _router().catalog
    _use_fake_model(monkeypatch, _vec_table(cat))
    r = SemanticReranker("fake/m", rerank_n=3, fusion="linear", alpha=0.5, cache_dir=tmp_path)
    assert r.recall_m == 0
    r.warm(cat)
    hits = cat.retrieve("查天气", None, 3)
    out = r.rerank("查天气", hits, cat)
    assert len(out) == len(hits)
    assert {rec.tool_id for rec, _ in out} == {rec.tool_id for rec, _ in hits}


def test_semantic_top_descending_and_length(monkeypatch, tmp_path):
    """semantic_top 返回全目录语义前 m 个,按分数降序,长度不超过 m。"""
    cat = _router().catalog
    _use_fake_model(monkeypatch, _vec_table(cat))
    r = SemanticReranker("fake/m", rerank_n=2, fusion="linear", alpha=0.5, recall_m=0,
                         cache_dir=tmp_path)
    top = r.semantic_top("查天气", 2, cat)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]
    assert top[0][0].tool_id == "Svc/get_weather"
    assert len(r.semantic_top("查天气", 99, cat)) == len(cat.records)   # m 超过目录大小也不报错


def test_recall_m_validation_and_config_pickle(monkeypatch, tmp_path):
    """recall_m 进 config 与 pickle;负数报错。"""
    with pytest.raises(ValueError):
        SemanticReranker("fake", recall_m=-1)
    _use_fake_model(monkeypatch, {})
    r = SemanticReranker("fake/m", rerank_n=7, fusion="linear", alpha=0.3, recall_m=20)
    assert r.config() == {"model": "fake/m", "n": 7, "fusion": "linear", "alpha": 0.3,
                          "recall_m": 20}
    state = pickle.loads(pickle.dumps(r))
    assert state.recall_m == 20 and state._vecs is None


def test_cli_reranker_recall_m_from_args(monkeypatch):
    """--rerank-recall-m 传进重排器;不传按 0(与 T-025 一致)。"""
    from types import SimpleNamespace

    from mcpcls import cli

    assert cli._reranker_from_args(SimpleNamespace(rerank=None)) is None
    _use_fake_model(monkeypatch, {})
    r = cli._reranker_from_args(SimpleNamespace(rerank="BAAI/bge-m3", rerank_n=50,
                                                rerank_fusion="linear", rerank_alpha=0.3,
                                                rerank_recall_m=20))
    assert r.recall_m == 20 and r.rerank_n == 50 and r.alpha == pytest.approx(0.3)


# ---------------------------------------------------------------- load_t025_controls

def _fake_t025_json(path):
    """造一份与 reports/eval_t025.json 同结构的最小 json,主选走 e2e.selected。"""
    def cand(key, zh5, en5, label):
        return {"key": key, "cfg": {"model": "x", "n": 50, "fusion": "linear", "alpha": 0.5},
                "label": label,
                "dev": {"n": 10, "zh_main_r1_strict": 0.1, "zh_main_r5_strict": zh5,
                        "zh_main_r5_name": zh5, "en_main_r1_strict": 0.1,
                        "en_main_r5_strict": en5, "en_main_r5_name": en5}}
    d = {"candidates": [cand("baseline", 0.30, 0.50, "基线(不重排)"),
                        cand("linear|x|50|0.5", 0.44, 0.61, "linear α=0.5 x N=50")],
         "selected_key": "linear|x|50|0.5", "lightweight_key": None,
         "e2e": {"selected": {"test": {"n": 10, "zh_main_r1_strict": 0.2,
                                       "zh_main_r5_strict": 0.40, "zh_main_r5_name": 0.5,
                                       "en_main_r1_strict": 0.3, "en_main_r5_strict": 0.6,
                                       "en_main_r5_name": 0.7}}}}
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_t025_controls_reads_main_and_test(tmp_path):
    p = _fake_t025_json(tmp_path / "eval_t025.json")
    c = tr.load_t025_controls(p)
    assert c["main"]["dev"]["zh_main_r5_strict"] == pytest.approx(0.44)
    assert c["main"]["test"]["zh_main_r5_strict"] == pytest.approx(0.40)   # 取自 e2e.selected
    assert c["lightweight"] is None


# ---------------------------------------------------------------- t026_eval / t026_ceiling(假重排器)

class FakeHybridReranker:
    """t026_eval / t026_ceiling 用的假重排器:按与查询的公共字符数打分,不需要模型。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg["model"]
        self.rerank_n = cfg["n"]
        self.recall_m = cfg.get("recall_m", 0)
        self.fusion = cfg["fusion"]
        self.alpha = cfg["alpha"]
        self.last_ms = 1.0
        self.first_build = None
        self._cat = None

    def warm(self, catalog):
        self._cat = catalog
        info = {"cached": True, "build_ms": 0.0, "n_tools": len(catalog.records),
                "dim": 4, "path": "fake.npy"}
        if self.first_build is None:
            self.first_build = info
        return info

    def _sem(self, query, rec, catalog):
        text = ToolCatalog._index_text(rec) if catalog is not None else rec.description
        return float(len(set(query) & set(text)))

    def semantic_top(self, query, m, catalog=None):
        cat = catalog if catalog is not None else self._cat
        if cat is None:
            return []
        scored = sorted(((r, self._sem(query, r, cat)) for r in cat.records),
                        key=lambda p: -p[1])
        return scored[:m]

    def rerank(self, query, hits, catalog=None):
        self.last_ms = 1.0
        pool = list(hits)
        if self.recall_m > 0:
            have = {r.tool_id for r, _ in pool}
            pool = pool + [(r, 0.0) for r, _ in self.semantic_top(query, self.recall_m, catalog)
                           if r.tool_id not in have]
        out = [(rec, self.alpha * s + (1 - self.alpha) * self._sem(query, rec, catalog))
               for rec, s in pool]
        return sorted(out, key=lambda p: -p[1])


def _fake_eval_data():
    tools = _synth_tools()
    real = [_tool(f"Svc/{n}#1", n, d) for n, d in
            (("mail_svc", "发邮件 send mail"), ("ls_svc", "列目录 list directory"))]
    labeled = [{"tool_id": t["tool_id"], "domains": ["comms" if "mail" in t["name"] else "file"],
                "actions": ["send" if "mail" in t["name"] else "read"], "agreed": True}
               for t in real]
    zh = [{"tool_id": t["tool_id"], "description_zh": f"中文说明{t['name']}"} for t in real]
    requests = [{"id": f"q{i}", "text": f"帮我处理第 {i} 件事", "text_orig": f"handle item {i}",
                 "tool_ids": ["Svc/mail_svc#1" if i % 2 else "Svc/ls_svc#1"],
                 "candidate_tool_ids": ["Svc/mail_svc#1", "Svc/ls_svc#1"],
                 "need": "single" if i % 3 else "none"} for i in range(10)]
    synth_train = _train()
    real_train = [{"id": f"rt{i}", "text": f"整理第 {i} 份报表",
                   "need": "single" if i % 2 else "none", "domains": ["file"],
                   "actions": ["read"], "tool_ids": ["Svc/ls_svc#1"]} for i in range(8)]
    return tools, real, labeled, zh, requests, synth_train, real_train


def test_t026_ceiling_on_fake_data():
    tools, real, labeled, zh, requests, synth_train, real_train = _fake_eval_data()
    labeled_tools = tr.attach_labels(real, labeled)
    res = tr.t026_ceiling(labeled_tools, tools, synth_train, real_train, requests,
                          {r["tool_id"]: r["description_zh"] for r in zh},
                          reranker_factory=FakeHybridReranker)
    assert res["n"] == 10 and res["ks"] == [50]
    assert set(res["union"]) == {"bge-m3", "e5-small"}
    for short in res["union"]:
        for M in (10, 20, 50):
            for lang in ("zh", "en"):
                for suf in ("strict", "name"):
                    assert 0.0 <= res["union"][short][M][lang][suf] <= 1.0
            assert set(res["sem_only"][short][M]) == {"zh", "en"}
    worth = tr._t026_union_worth(res)
    assert "worth" in worth and worth["k"] == 50


def test_t026_eval_smoke_with_fake_rerankers(tmp_path):
    """假重排器工厂跑通全流程:候选齐全、对照读 fake json、测试折只报一次、报告与 json 可用。"""
    tools, real, labeled, zh, requests, synth_train, real_train = _fake_eval_data()
    t025 = _fake_t025_json(tmp_path / "eval_t025.json")
    res = tr.t026_eval(real, labeled, zh, requests, tools, synth_train, real_train,
                       reranker_factory=FakeHybridReranker, e2e=True, t025_json=t025)
    keys = {c["key"] for c in res["candidates"]}
    assert "baseline" in keys
    # H1:2 模型 × 3 个 M × 2 个 α = 12
    assert sum(1 for c in res["candidates"][1:] if c["cfg"]) == 12
    for c in res["candidates"][1:]:
        assert isinstance(c["qualified"], bool)
        assert c["en_drop"] == pytest.approx(
            res["candidates"][0]["dev"]["en_main_r5_strict"] - c["dev"]["en_main_r5_strict"])
        assert c["hybrid_p95"] == pytest.approx(c["dev"]["hybrid_p95"])
        assert c["cfg"]["recall_m"] in (10, 20, 50)
    assert res["selected_key"] in keys
    assert res["lightweight_key"] is None or res["lightweight_key"] in keys
    assert len(res["dev_ids"]) + len(res["test_ids"]) == 10
    assert res["t025_source"] == "eval_t025.json"
    assert isinstance(res["gains"]["benefit"], bool)
    # 测试折只报一次:基线 + 主选 + 轻量备选
    assert set(res["e2e"]) == {"baseline", "selected", "lightweight"}
    assert res["e2e"]["selected"]["test"]["n"] == len(res["test_ids"])
    report = tr.render_t026(res)
    for section in ("# 混合召回:tfidf Top-50 ∪ 语义 Top-M + 融合重排(T-026)", "## 0. 调查结论",
                    "## 1. 候选对照", "## 2. 测试折", "## 3. 部署说明", "## 4. 结论",
                    "语义独有金标", "并集值得做" if res["union_worth"]["worth"] else "并集不值得做"):
        assert section in report, section
    json.dumps(tr.t026_metrics_json(res), ensure_ascii=False)


def test_t026_eval_no_qualified_keeps_baseline(tmp_path):
    """所有候选都被约束排除时,选中项回落到基线(不改默认)。"""
    tools, real, labeled, zh, requests, synth_train, real_train = _fake_eval_data()
    t025 = _fake_t025_json(tmp_path / "eval_t025.json")

    class TerribleReranker(FakeHybridReranker):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.last_ms = 999.0                 # 约束必挂:p95 = 999 ms > 600 ms

        def rerank(self, query, hits, catalog=None):
            self.last_ms = 999.0
            return list(reversed(hits))          # 顺序倒过来,质量也只会更差

    res = tr.t026_eval(real, labeled, zh, requests, tools, synth_train, real_train,
                       reranker_factory=TerribleReranker, e2e=False, t025_json=t025)
    sel = next(c for c in res["candidates"] if c["key"] == res["selected_key"])
    assert sel["key"] == "baseline"
    assert "e2e" not in res
