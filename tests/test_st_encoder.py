"""T-016:`SentenceTransformerEncoder` 接口 + 编码器评估结构的离线测试。

不下载模型、不联网:把假的 `sentence_transformers` 模块塞进 `sys.modules`,
真模型一律不加载。验收标准 A1 要求测试不下载模型、不联网。
"""

import json
import socket
import sys
import types

import numpy as np
import pytest

from mcpcls import tools_real as tr
from mcpcls.encoder import SentenceTransformerEncoder, build_encoder


class _FakeModel:
    """替身:不加载真权重。`encode` 记下收到的文本,并按文本内容返回有区分度的向量。"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []

    def encode(self, texts, normalize_embeddings=False, **_kw):
        self.calls.append((list(texts), normalize_embeddings))
        rows = []
        for t in texts:
            rows.append([len(t), (sum(map(ord, t)) % 997) / 997.0, (abs(hash(t)) % 499) / 499.0])
        return np.asarray(rows, dtype=np.float32)


def _install_fake_st(monkeypatch):
    """注入假的 sentence_transformers,返回记录最后一次构造的 dict。"""
    made = {}
    mod = types.ModuleType("sentence_transformers")

    def _ctor(name):
        made["last"] = _FakeModel(name)
        return made["last"]

    mod.SentenceTransformer = _ctor
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return made


# ---------------------------------------------------------------- 编码器接口

def test_st_encoder_uses_given_model_and_normalizes(monkeypatch):
    made = _install_fake_st(monkeypatch)
    e = build_encoder("st:BAAI/bge-m3")
    assert isinstance(e, SentenceTransformerEncoder)
    assert e.model_name == "BAAI/bge-m3"
    assert made["last"].model_name == "BAAI/bge-m3"
    v = e.encode(["a", "b"])
    assert v.shape == (2, 3)
    assert made["last"].calls == [(["a", "b"], True)]


def test_st_encoder_empty_arg_falls_back_to_default(monkeypatch):
    _install_fake_st(monkeypatch)
    e = build_encoder("st:")
    assert e.model_name == "BAAI/bge-small-zh-v1.5"


def test_st_encoder_config_roundtrip(monkeypatch):
    _install_fake_st(monkeypatch)
    e = build_encoder("st:intfloat/multilingual-e5-small")
    assert e.config() == {"type": "st", "model_name": "intfloat/multilingual-e5-small"}


def test_unknown_encoder_spec_raises():
    with pytest.raises(ValueError):
        build_encoder("nope:x")


def test_st_encoder_never_touches_the_network(monkeypatch):
    _install_fake_st(monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("st 编码器测试不应联网")

    monkeypatch.setattr(socket, "socket", boom)
    e = build_encoder("st:BAAI/bge-m3")
    assert e.encode(["x"]).shape == (1, 3)


# ---------------------------------------------------------------- 评估结构(假数据)

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


def test_encoder_eval_structure_and_render(monkeypatch):
    """encoder_eval 在假数据上跑通:编码器 × 索引 × 中英 × 两类口径,报告与 json 都出得来。"""
    _install_fake_st(monkeypatch)
    encoders = [("tfidf", "tfidf"), ("st:fake/repo", "fake-st")]
    res = tr.encoder_eval(_real_tools(), _labeled_rows(), _zh_rows(), _requests(),
                          _synth_tools(), _synth_train(), [], encoders=encoders, folds=5, seed=0)

    req_ids = {r["id"] for r in _requests()}
    for label in ("tfidf", "fake-st"):
        for idx in ("mono", "bi"):
            assert set(res["per"][(label, idx)]) == req_ids
            item = res["per"][(label, idx)]["q0"]
            for key in ("zh_main_r1_strict", "zh_main_r5_strict", "zh_main_r1_name",
                        "zh_main_r5_name", "zh_upper_r5_strict", "en_upper_r1_name",
                        "latency_ms", "risk_top1", "pred_none"):
                assert key in item
        assert res["pkl_bytes"][label] > 0
        s = tr.t016_summary(res, res["dev_ids"], label, "bi")
        assert 0.0 <= s["zh_upper_r5_strict"] <= 1.0
        assert s["latency_p95"] >= s["latency_p50"] >= 0.0

    report = tr.render_t016(res)
    for section in ("# 多语言语义编码器评估(T-016)", "## 0. 结论", "## 1. 开发折",
                    "## 2. 测试折", "## 3. 延迟", "## 4. 模型"):
        assert section in report
    for col in ("严格 R@1", "严格 R@5", "同名 R@1", "同名 R@5", "单语索引", "双语索引"):
        assert col in report
    assert "tfidf" in report and "fake-st" in report
    json.dumps(tr.t016_metrics_json(res), ensure_ascii=False)
