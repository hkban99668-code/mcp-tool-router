"""T-019:检索编码器与分类头编码器分开配置 + 部署瘦身。全部离线,用假编码器。

- A2 不传 `retrieval_encoder` 时检索与改前逐条一致;`e2e_eval` 默认与改前同结果。
- A3 传了 `retrieval_encoder` 时分类头输出(need / 能力域 / 动作)逐条不变。
- A4 带 ST 编码器的 router 序列化后不含模型权重,反序列化后能正常编码。
- A5 `split_encoder_eval` 在假数据上跑通,报告与 json 都出得来,need 三行一致。
"""

import json
import pickle
import sys
import types

import numpy as np
import pytest
from scipy import sparse

from mcpcls import evaluate as ev
from mcpcls import tools_real as tr
from mcpcls.catalog import ToolCatalog
from mcpcls.encoder import Encoder, SentenceTransformerEncoder
from mcpcls.router import Router


# ---------------------------------------------------------------- 假编码器 / 假数据

class _BagEncoder(Encoder):
    """按字符映射到固定维度的稀疏向量:确定、不训练、不用下载。

    用来当"另一个检索编码器",验证换检索编码器只动检索、不动分类头。
    """

    name = "bag"

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts):
        rows, cols, data = [], [], []
        for i, t in enumerate(texts):
            for ch in t:
                rows.append(i)
                cols.append(ord(ch) % self.dim)
                data.append(1.0)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), self.dim))


class _FakeModel:
    """替身:不加载真权重。带一块 2 MB 的 blob,默认 pickle 会把它灌进去。"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.blob = np.zeros(500_000, dtype=np.float32)

    def encode(self, texts, normalize_embeddings=False, **_kw):
        return np.ones((len(texts), 4), dtype=np.float32)


def _install_fake_st(monkeypatch):
    """注入假的 sentence_transformers,返回记录构造次数的 dict。"""
    made = {"count": 0}

    def _ctor(name):
        made["count"] += 1
        made["last"] = _FakeModel(name)
        return made["last"]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _ctor
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return made


def _tool(tid, name, desc):
    return {"tool_id": tid, "name": name, "server": "Svc", "description": desc,
            "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}


def _real_tools():
    return [_tool("Svc/hi#2", "hi", "Say hi."), _tool("Svc/hello", "hello", "Say hello."),
            _tool("Svc/ls", "list_all", "List everything.")]


def _labeled_rows():
    return [{"tool_id": "Svc/hi#2", "domains": ["system_misc"], "actions": ["read"], "agreed": True},
            {"tool_id": "Svc/hello", "domains": ["system_misc"], "actions": ["read"], "agreed": True},
            {"tool_id": "Svc/ls", "domains": ["file"], "actions": ["read"], "agreed": True}]


def _zh_rows():
    return [{"tool_id": "Svc/hi#2", "description_zh": "打招呼"},
            {"tool_id": "Svc/hello", "description_zh": "问好"},
            {"tool_id": "Svc/ls", "description_zh": "列出全部"}]


def _synth_tools():
    return [dict(_tool("synth/send", "send_email", "Send an email."),
                 domains=["comms"], actions=["send"]),
            dict(_tool("synth/ls", "list_files", "List files."),
                 domains=["file"], actions=["read"])]


def _synth_train():
    return [{"text": "给张三发邮件", "need": "single", "domains": ["comms"], "actions": ["send"],
             "tool_ids": ["synth/send"]},
            {"text": "列出文件", "need": "single", "domains": ["file"], "actions": ["read"],
             "tool_ids": ["synth/ls"]},
            {"text": "你好呀", "need": "none", "domains": [], "actions": [], "tool_ids": []}]


def _requests():
    return [{"id": f"q{i}", "text": f"请求 {i}", "text_orig": f"request {i}",
             "tool_ids": ["Svc/hi#2"] if i % 2 == 0 else ["Svc/hello"],
             "candidate_tool_ids": ["Svc/hi#2", "Svc/hello", "Svc/ls"],
             "need": "single" if i % 3 else "none"} for i in range(6)]


def _head():
    enc = ev.fit_encoder("tfidf", _synth_tools(), _synth_train())
    return ev.train_tool_head(enc, _synth_tools())


# ---------------------------------------------------------------- A2 默认行为不变

def test_catalog_default_uses_head_encoder():
    """不传 retrieval_encoder 时,ToolCatalog 的检索编码器就是分类头编码器。"""
    head = _head()
    cat = ToolCatalog(head)
    assert cat.encoder is head.encoder
    assert ToolCatalog(head, retrieval_encoder=None).encoder is head.encoder


def test_catalog_retrieve_unchanged_without_retrieval_encoder():
    """默认构造的目录与显式传 None / 传 head.encoder 的目录,检索结果逐条一致。"""
    head = _head()
    cat_a = ToolCatalog(head)
    cat_a.add_tools(_real_tools(), use_labels=True)
    cat_b = ToolCatalog(head, retrieval_encoder=None)
    cat_b.add_tools(_real_tools(), use_labels=True)
    cat_c = ToolCatalog(head, retrieval_encoder=head.encoder)
    cat_c.add_tools(_real_tools(), use_labels=True)
    for q in ("打招呼", "say hi", "list everything"):
        a = [(r.tool_id, s) for r, s in cat_a.retrieve(q)]
        b = [(r.tool_id, s) for r, s in cat_b.retrieve(q)]
        c = [(r.tool_id, s) for r, s in cat_c.retrieve(q)]
        assert a == b == c


def test_build_fold_index_default_equals_explicit_none():
    """不传 retrieval_encoder 与显式传 None,build_fold_index 的检索结果一致。"""
    labeled = tr.attach_labels(_real_tools(), _labeled_rows())
    zh_by_id = {r["tool_id"]: r["description_zh"] for r in _zh_rows()}
    kwargs = dict(bilingual=True, zh_by_id=zh_by_id)
    _e1, _h1, cat1 = tr.build_fold_index(labeled, _synth_tools(), _synth_train(), **kwargs)
    _e2, _h2, cat2 = tr.build_fold_index(labeled, _synth_tools(), _synth_train(),
                                         retrieval_encoder=None, **kwargs)
    for q in ("打招呼", "say hi"):
        assert [(r.tool_id, s) for r, s in cat1.retrieve(q)] == \
               [(r.tool_id, s) for r, s in cat2.retrieve(q)]


def test_e2e_eval_default_matches_no_arg():
    """e2e_eval 不传 retrieval_encoder 与显式传 None,汇总数字完全一致。"""
    args = (_real_tools(), _labeled_rows(), _zh_rows(), _requests(), _synth_tools(),
            _synth_train(), [])
    res1 = tr.e2e_eval(*args, folds=5, seed=0, top_k=5)
    res2 = tr.e2e_eval(*args, folds=5, seed=0, top_k=5, retrieval_encoder=None)

    def _drop_latency(summary):
        return {cat: {k: v for k, v in s.items() if not k.startswith("latency")}
                for cat, s in summary.items()}

    assert _drop_latency(tr.e2e_summary(res1)) == _drop_latency(tr.e2e_summary(res2))


# ---------------------------------------------------------------- A3 分类头不变

def test_retrieval_encoder_only_changes_retrieval():
    """换检索编码器后,分类头输出(能力域 / 动作 / 置信度)逐条不变,检索命中可不同。"""
    head = _head()
    cat1 = ToolCatalog(head)
    cat1.add_tools(_real_tools(), use_labels=True)
    cat2 = ToolCatalog(head, retrieval_encoder=_BagEncoder())
    cat2.add_tools(_real_tools(), use_labels=True)

    o1 = cat1.classify_tools(_real_tools())
    o2 = cat2.classify_tools(_real_tools())
    assert [o.domains for o in o1] == [o.domains for o in o2]
    assert [o.actions for o in o1] == [o.actions for o in o2]
    assert [o.confidence for o in o1] == [o.confidence for o in o2]
    assert cat1.encoder is not cat2.encoder


def test_route_need_domains_unchanged_with_retrieval_encoder():
    """走完整路由:换检索编码器只影响候选,need 与能力域逐条一致。"""
    head = _head()
    cat1 = ToolCatalog(head)
    cat1.add_tools(_real_tools(), use_labels=True)
    cat2 = ToolCatalog(head, retrieval_encoder=_BagEncoder())
    cat2.add_tools(_real_tools(), use_labels=True)
    rhead = ev.train_request_head(head.encoder, _synth_train())
    r1 = Router(rhead, cat1, use_filter=False)
    r2 = Router(rhead, cat2, use_filter=False)
    for q in ("给张三发邮件", "列出文件", "你好呀", "say hi"):
        a, b = r1.route(q), r2.route(q)
        assert a.need == b.need
        assert a.domains == b.domains
        assert a.actions == b.actions


# ---------------------------------------------------------------- A4 pickle 不含权重

def test_st_encoder_getstate_keeps_only_model_name(monkeypatch):
    _install_fake_st(monkeypatch)
    e = SentenceTransformerEncoder("fake/repo")
    assert e.__getstate__() == {"model_name": "fake/repo"}


def test_st_encoder_pickle_has_no_weights_and_reloads(monkeypatch):
    """2 MB 的假权重不进 pickle;反序列化后重新构造模型,能继续编码。"""
    made = _install_fake_st(monkeypatch)
    e = SentenceTransformerEncoder("fake/repo")
    blob = pickle.dumps(e)
    assert len(blob) < 100_000          # 若权重被序列化会 > 2 MB
    assert b"blob" not in blob
    before = made["count"]
    e2 = pickle.loads(blob)
    assert made["count"] == before + 1   # 反序列化时从模型名重建
    assert e2.model_name == "fake/repo"
    assert e2.encode(["x"]).shape[0] == 1


def test_router_pickle_with_st_retrieval_is_slim(monkeypatch):
    """带 ST 检索编码器的 router,pickle 里只有模型名;反序列化后仍能编码检索。"""
    _install_fake_st(monkeypatch)
    head = _head()
    st = SentenceTransformerEncoder("fake/repo")
    cat = ToolCatalog(head, retrieval_encoder=st)
    cat.add_tools(_real_tools(), use_labels=True)
    rhead = ev.train_request_head(head.encoder, _synth_train())
    router = Router(rhead, cat, use_filter=False)

    blob = pickle.dumps(router)
    assert len(blob) < 500_000          # 权重 2 MB 没进去
    loaded = pickle.loads(blob)
    assert loaded.catalog.encoder.model_name == "fake/repo"
    loaded.route("打招呼")               # 反序列化后的编码器可用


# ---------------------------------------------------------------- A5 分开配置评估结构

def test_t019_label_maps_known_specs():
    assert tr.t019_label("tfidf") == "tfidf"
    assert tr.t019_label("st:BAAI/bge-m3") == "bge-m3"
    assert tr.t019_label("st:intfloat/multilingual-e5-small") == "e5-small"
    assert tr.t019_label("st:org/some-model") == "some-model"


def test_split_encoder_eval_structure_render_and_need_identical(monkeypatch):
    """分类头固定 tfidf,检索编码器逐个换:结构与渲染出得来,need 三行一致。"""
    _install_fake_st(monkeypatch)
    retrieval = [("tfidf", "tfidf"), ("st:fake/repo", "fake-st")]
    res = tr.split_encoder_eval(_real_tools(), _labeled_rows(), _zh_rows(), _requests(),
                                _synth_tools(), _synth_train(), [],
                                head_encoder="tfidf", retrieval_encoders=retrieval,
                                folds=5, seed=0)
    assert res["head_encoder"] == "tfidf"
    req_ids = {r["id"] for r in _requests()}
    for _spec, label in retrieval:
        assert set(res["per"][label]) == req_ids
        item = res["per"][label]["q0"]
        for key in ("zh_main_r1_strict", "zh_main_r5_strict", "zh_main_r1_name",
                    "zh_main_r5_name", "zh_upper_r5_strict", "en_upper_r1_name",
                    "zh_pred_none", "en_pred_none", "latency_ms", "risk_top1"):
            assert key in item
        s = tr.t019_summary(res, res["dev_ids"], label)
        assert 0.0 <= s["zh_main_r5_strict"] <= 1.0
        assert s["latency_p95"] >= s["latency_p50"] >= 0.0

    # need 判定与检索编码器无关:三种(这里两种)逐条一致
    for label in ("fake-st",):
        for req in _requests():
            assert (res["per"][label][req["id"]]["zh_pred_none"]
                    == res["per"]["tfidf"][req["id"]]["zh_pred_none"])
            assert (res["per"][label][req["id"]]["en_pred_none"]
                    == res["per"]["tfidf"][req["id"]]["en_pred_none"])
    assert (tr.t019_need_metrics(res, "tfidf", "zh")
            == tr.t019_need_metrics(res, "fake-st", "zh"))

    for _spec, label in retrieval:
        assert res["pkl"][label]["slim"] > 0
    report = tr.render_t019(res)
    for section in ("# 检索编码器与分类头编码器分开评估(T-019)", "## 0. 结论", "## 1. need 判定",
                    "## 2. 开发折", "## 3. 测试折", "## 4. 延迟", "## 5. 模型与"):
        assert section in report
    for col in ("严格 R@1", "严格 R@5", "同名 R@1", "同名 R@5"):
        assert col in report
    assert "tfidf" in report and "fake-st" in report
    assert "结论" in report
    json.dumps(tr.t019_metrics_json(res), ensure_ascii=False)


def test_split_router_pkl_size_reports_slim_and_fat(monkeypatch):
    """带 ST 检索编码器:改后 pkl 明显小于改前(权重不再进 pickle)。"""
    _install_fake_st(monkeypatch)
    labeled = tr.attach_labels(_real_tools(), _labeled_rows())
    zh_by_id = {r["tool_id"]: r["description_zh"] for r in _zh_rows()}
    out = tr.split_router_pkl_size("tfidf", "st:fake/repo", labeled, _synth_tools(),
                                   _synth_train(), [], zh_by_id)
    assert out["slim"] > 0
    assert out["fat"] > out["slim"] + 1_000_000     # 2 MB 权重被 fat 序列化进去
