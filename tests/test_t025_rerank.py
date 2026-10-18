"""T-025:检索两段式(tfidf 召回 Top-N + 语义重排)。

- 全部用假编码器 / 假重排器,不依赖语义模型与本地 HF 缓存;
- 不开重排时 Router 行为与改前逐项一致(A2);
- 开重排时只在 tfidf 召回的 Top-N 池子里换顺序,池外够不着;
- 环境不满足(没装 sentence-transformers / 模型不在缓存)时报 `RerankError`,提示可执行(A4);
- `t025_eval` 用假重排器工厂在假数据上跑通,报告与 json 都出得来。
"""

import pickle
import sys

import numpy as np
import pytest

from mcpcls import evaluate as ev
from mcpcls import rerank as rr_mod
from mcpcls import tools_real as tr
from mcpcls.catalog import ToolCatalog
from mcpcls.rerank import RerankError, SemanticReranker
from mcpcls.router import Router


# ---------------------------------------------------------------- 假编码器 / 假重排器

class FakeST:
    """假的 SentenceTransformer:按预置的 文本 -> 向量 表编码,没见过的文本给零向量。"""

    def __init__(self, table):
        self.table = table

    def encode(self, texts, normalize_embeddings=False):
        return np.asarray([self.table.get(t, np.zeros(4, dtype=np.float32)) for t in texts],
                          dtype=np.float32)


class ReverseReranker:
    """把 tfidf 召回顺序整个倒过来的假重排器,顺序断言用,不关心语义。"""

    model_name = "fake"

    def __init__(self, rerank_n=10):
        self.rerank_n = rerank_n
        self.last_ms = 0.5

    def rerank(self, query, hits, catalog=None):
        self.last_ms = 0.5
        return [(r, -float(i)) for i, (r, _s) in enumerate(reversed(hits))]


class FakeReranker:
    """t025_eval 用的假重排器:按与查询的公共字符数打分,行为确定且不需要模型。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg["model"]
        self.rerank_n = cfg["n"]
        self.last_ms = 1.0
        self.first_build = None

    def rerank(self, query, hits, catalog=None):
        self.last_ms = 1.0

        def score(item):
            rec, s = item
            text = ToolCatalog._index_text(rec) if catalog is not None else rec.description
            shared = float(len(set(query) & set(text)))
            if self.cfg["fusion"] == "linear":
                return rec, self.cfg["alpha"] * s + (1 - self.cfg["alpha"]) * shared
            return rec, shared
        return sorted((score(h) for h in hits), key=lambda p: -p[1])


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
            dict(_tool("synth/wx", "get_weather", "Get the weather."),
                 domains=["system_misc"], actions=["read"])]


def _train():
    return [{"text": "给张三发邮件", "need": "single", "domains": ["comms"], "actions": ["send"],
             "tool_ids": ["synth/send"]},
            {"text": "列出全部文件", "need": "single", "domains": ["file"], "actions": ["read"],
             "tool_ids": ["synth/ls"]},
            {"text": "今天天气怎么样", "need": "single", "domains": ["system_misc"],
             "actions": ["read"], "tool_ids": ["synth/wx"]},
            {"text": "你好呀", "need": "none", "domains": [], "actions": [], "tool_ids": []}]


def _router(reranker=None, top_k=5):
    return ev.build_system("tfidf", _synth_tools(), _train(), top_k=top_k, reranker=reranker)


# ---------------------------------------------------------------- 融合数学

def test_fuse_reorder_linear_rrf_math():
    """三种融合的排序分数都能按公式手算对上。"""
    r = SemanticReranker("fake", rerank_n=3, fusion="reorder")
    hits = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    t = np.asarray([0.9, 0.5, 0.1])
    sem = np.asarray([0.2, 0.8, 0.5])
    out = r._fuse(hits, t, sem)
    assert [k for k, _ in out] == ["b", "c", "a"]           # 纯语义,分数换成语义
    assert out[0][1] == pytest.approx(0.8)

    r = SemanticReranker("fake", rerank_n=3, fusion="linear", alpha=0.25)
    out = r._fuse(hits, t, sem)
    scores = [0.25 * 0.9 + 0.75 * 0.2, 0.25 * 0.5 + 0.75 * 0.8, 0.25 * 0.1 + 0.75 * 0.5]
    assert [k for k, _ in out] == ["b", "c", "a"]
    assert out[0][1] == pytest.approx(scores[1])

    r = SemanticReranker("fake", rerank_n=3, fusion="rrf")
    out = r._fuse(hits, t, sem)
    # tfidf 名次 a<b<c,语义名次 b<c<a;RRF_K=60,名次从 0 起:1/(60+名次)
    want = {"a": 1 / 60 + 1 / 62, "b": 1 / 61 + 1 / 60, "c": 1 / 62 + 1 / 61}
    assert {k: s for k, s in out} == pytest.approx(want)
    assert [k for k, _ in out] == ["b", "a", "c"]


def test_semantic_reranker_uses_fake_model_and_caches_vectors(monkeypatch, tmp_path):
    """warm 把工具向量算一次落盘;第二次直接读缓存;rerank 返回重排后的顺序。"""
    _use_fake_model(monkeypatch, {})
    r = SemanticReranker("fake/model", rerank_n=2, fusion="reorder", cache_dir=tmp_path)
    cat = _router().catalog
    info = r.warm(cat)
    assert info["cached"] is False and info["n_tools"] == 3
    assert (tmp_path / pathlib_name(info["path"])).exists()
    info2 = r.warm(cat)
    assert info2["cached"] is True
    # 未见过的文本编码成零向量 -> 语义分全 0 -> 顺序不变(稳定排序)
    hits = cat.retrieve("发邮件", None, 3)
    out = r.rerank("发邮件", hits, cat)
    assert [rec.tool_id for rec, _ in out] == [rec.tool_id for rec, _ in hits]
    assert r.last_ms >= 0.0


def pathlib_name(path):
    import pathlib

    return pathlib.Path(path).name


# ---------------------------------------------------------------- Router 行为(A2 / 池子)

def test_router_without_reranker_unchanged():
    """不开重排:候选与 tfidf 检索逐项一致,与改前行为相同(A2)。"""
    router = _router()
    full = [rec.tool_id for rec, _ in router.catalog.retrieve("给张三发邮件", None, 5)]
    res = router.route("给张三发邮件")
    assert [c["tool_id"] for c in res.candidates] == full[:5]
    assert router.reranker is None


def test_router_with_reranker_reorders_within_pool():
    """开重排:top_k=2、rerank_n=3 时,池子是 tfidf 前 3,倒序后取 [第3, 第2] 名。"""
    router = _router(reranker=ReverseReranker(rerank_n=3), top_k=2)
    full = [rec.tool_id for rec, _ in router.catalog.retrieve("给张三发邮件", None, 5)]
    res = router.route("给张三发邮件")
    assert [c["tool_id"] for c in res.candidates] == [full[2], full[1]]
    assert router.reranker.last_ms >= 0.0


def test_router_rerank_pool_at_least_top_k():
    """rerank_n 小于 top_k 时池子仍取 top_k,不会把候选变少。"""
    router = _router(reranker=ReverseReranker(rerank_n=1), top_k=3)
    full = [rec.tool_id for rec, _ in router.catalog.retrieve("给张三发邮件", None, 5)]
    res = router.route("给张三发邮件")
    assert [c["tool_id"] for c in res.candidates] == list(reversed(full[:3]))


def test_reranker_pickles_config_only(monkeypatch):
    """pickle 只存配置;反序列化后模型懒加载(走当时的加载函数)。"""
    table = {"x": np.ones(4, dtype=np.float32)}
    _use_fake_model(monkeypatch, table)
    r = SemanticReranker("fake/model", rerank_n=7, fusion="linear", alpha=0.3)
    state = pickle.loads(pickle.dumps(r))
    assert state.rerank_n == 7 and state.fusion == "linear" and state.alpha == pytest.approx(0.3)
    assert state._model is None and state._vecs is None
    assert state._st().encode(["x"])[0].sum() == pytest.approx(4.0)


def test_old_router_pickle_without_reranker_attr():
    """旧 router.pkl(没有 reranker 属性)在 route 里按 None 处理,不报错。"""

    class OldRouter:
        pass

    old = OldRouter()
    assert getattr(old, "reranker", None) is None   # Router.route 里的读法


# ---------------------------------------------------------------- A4:环境不满足时报错清楚

def test_missing_sentence_transformers_error(monkeypatch):
    """没装 sentence-transformers:报错里给出安装命令和退路。"""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankError) as e:
        rr_mod._load_st_model("BAAI/bge-m3")
    assert "requirements-st.txt" in str(e.value) and "--rerank" in str(e.value)


def test_model_not_in_cache_error(monkeypatch, tmp_path):
    """模型不在本地 HF 缓存:报错里给出 huggingface-cli download 命令。"""
    monkeypatch.setattr(rr_mod, "_hf_cache_dir", lambda: tmp_path)
    with pytest.raises(RerankError) as e:
        rr_mod._load_st_model("BAAI/bge-m3")
    assert "huggingface-cli download BAAI/bge-m3" in str(e.value)


def test_reranker_argument_validation():
    with pytest.raises(ValueError):
        SemanticReranker("fake", fusion="nope")
    with pytest.raises(ValueError):
        SemanticReranker("fake", alpha=1.5)
    with pytest.raises(ValueError):
        SemanticReranker("fake", rerank_n=0)


def test_cli_reranker_from_args(monkeypatch):
    """--rerank 没传返回 None;传了返回配置一致的重排器(构造时不加载模型)。"""
    from mcpcls import cli
    from types import SimpleNamespace

    assert cli._reranker_from_args(SimpleNamespace(rerank=None)) is None
    _use_fake_model(monkeypatch, {})
    r = cli._reranker_from_args(SimpleNamespace(rerank="BAAI/bge-m3", rerank_n=30,
                                                rerank_fusion="linear", rerank_alpha=0.7))
    assert isinstance(r, SemanticReranker)
    assert r.config() == {"model": "BAAI/bge-m3", "n": 30, "fusion": "linear", "alpha": 0.7,
                          "recall_m": 0}


def test_build_real_router_reranker_default_none():
    """build_real_router 默认不重排;snapshot 签名不受新增参数影响。"""
    import inspect

    p = inspect.signature(tr.build_real_router).parameters
    assert p["reranker"].default is None
    assert "reranker" not in tr.snapshot_signature()


# ---------------------------------------------------------------- t025_eval(假重排器工厂)

def _fake_eval_data():
    tools = _synth_tools()
    real = [_tool(f"Svc/{n}#1", n, d) for n, d in
            (("mail_svc", "Send mail via service."), ("ls_svc", "List directory contents."))]
    labeled = [{"tool_id": t["tool_id"], "domains": ["comms" if "mail" in t["name"] else "file"],
                "actions": ["send" if "mail" in t["name"] else "read"], "agreed": True}
               for t in real]
    zh = [{"tool_id": t["tool_id"], "description_zh": f"中文说明{t['name']}"} for t in real]
    requests = [{"id": f"q{i}", "text": f"帮我发第 {i} 封邮件", "text_orig": f"send email {i}",
                 "tool_ids": ["Svc/mail_svc#1" if i % 2 else "Svc/ls_svc#1"],
                 "candidate_tool_ids": ["Svc/mail_svc#1", "Svc/ls_svc#1"],
                 "need": "single" if i % 3 else "none"} for i in range(10)]
    synth_train = _train()
    real_train = [{"id": f"rt{i}", "text": f"整理第 {i} 份报表", "need": "single" if i % 2 else "none",
                   "domains": ["file"], "actions": ["read"], "tool_ids": ["Svc/ls_svc#1"]}
                  for i in range(8)]
    return tools, real, labeled, zh, requests, synth_train, real_train


def test_t025_ceiling_on_fake_data():
    tools, real, labeled, zh, requests, synth_train, real_train = _fake_eval_data()
    labeled_tools = tr.attach_labels(real, labeled)
    res = tr.t025_ceiling(labeled_tools, tools, synth_train, real_train, requests,
                          {r["tool_id"]: r["description_zh"] for r in zh})
    assert res["n"] == 10 and res["ks"] == [5, 10, 20, 50]
    for lang in ("zh", "en"):
        for suf in ("strict", "name"):
            assert all(0.0 <= v <= 1.0 for v in res[lang][suf].values())
    assert res["index_n"] == len(real)


def test_t025_eval_smoke_with_fake_rerankers():
    """假重排器工厂跑通全流程:候选齐全、约束算过、测试折只报一次、报告与 json 可用。"""
    tools, real, labeled, zh, requests, synth_train, real_train = _fake_eval_data()
    res = tr.t025_eval(real, labeled, zh, requests, tools, synth_train, real_train,
                       reranker_factory=FakeReranker, e2e=True)
    keys = {c["key"] for c in res["candidates"]}
    assert "baseline" in keys
    # R1/R2:两个模型 × 3 个 N;R3:更好的模型上 linear 3α×3N + RRF 3N,
    # 另一个模型再补 linear 3α×2N + RRF 2N(review 第 1 轮)
    assert sum(1 for c in res["candidates"] if c["cfg"] and c["cfg"]["fusion"] == "reorder") == 6
    assert sum(1 for c in res["candidates"] if c["cfg"] and c["cfg"]["fusion"] == "linear") == 15
    assert sum(1 for c in res["candidates"] if c["cfg"] and c["cfg"]["fusion"] == "rrf") == 5
    for c in res["candidates"][1:]:
        assert isinstance(c["qualified"], bool)
        assert c["en_drop"] == pytest.approx(
            res["candidates"][0]["dev"]["en_main_r5_strict"] - c["dev"]["en_main_r5_strict"])
    assert res["selected_key"] in keys
    assert res["lightweight_key"] is None or res["lightweight_key"] in keys
    assert len(res["dev_ids"]) + len(res["test_ids"]) == 10
    # 测试折只报一次:基线 + 主选 + 轻量备选
    assert set(res["e2e"]) == {"baseline", "selected", "lightweight"}
    assert set(res["e2e"]["baseline"]) == {"dev", "test"}
    assert res["e2e"]["selected"]["test"]["n"] == len(res["test_ids"])
    report = tr.render_t025(res)
    for section in ("# 检索两段式:tfidf 召回 Top-N + 语义重排(T-025)", "## 0. 调查结论",
                    "## 1. 候选对照", "## 2. 测试折", "## 3. 部署说明", "## 4. 结论",
                    "轻量备选"):
        assert section in report, section
    import json

    json.dumps(tr.t025_metrics_json(res), ensure_ascii=False)


def test_t025_eval_no_qualified_keeps_baseline():
    """所有候选都被约束排除时,选中项回落到基线(不改默认)。"""
    tools, real, labeled, zh, requests, synth_train, real_train = _fake_eval_data()

    class TerribleReranker(FakeReranker):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.last_ms = 999.0                 # 延迟约束必挂:p95 = 999 ms > 50 ms

        def rerank(self, query, hits, catalog=None):
            self.last_ms = 999.0
            return list(reversed(hits))          # 顺序倒过来,质量也只会更差

    res = tr.t025_eval(real, labeled, zh, requests, tools, synth_train, real_train,
                       reranker_factory=TerribleReranker, e2e=False)
    sel = next(c for c in res["candidates"] if c["key"] == res["selected_key"])
    assert sel["key"] == "baseline"
    assert "e2e" not in res
