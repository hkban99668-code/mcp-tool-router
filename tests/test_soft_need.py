"""T-020:低置信度的 none 也附带候选工具(软 need)。全部离线,用假头 / 假目录。

- A2 不传 / 显式传 `soft_need_threshold=None` 时,路由结果与改前逐条一致,`soft_candidates`
     恒为 False。
- A3 设了阈值时,只有"判成 none 且 need 头给的 none 概率 < 阈值"的请求才附带候选;
     `need` 字段不变,`soft_candidates` 只在软触发时为 True;拿不到概率时不触发。
-  带真实分类头 / 目录跑一条:软触发不改变 need,候选来自真实检索。
- A4 `soft_need_eval` 在假数据上跑通,阈值网格 / 误挂率 / 选择 / 测试折复核都出得来。
"""

import json
import pathlib

import pytest

from mcpcls import evaluate as ev
from mcpcls import tools_real as tr
from mcpcls.heads import HeadOutput
from mcpcls.router import Router

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- 假头 / 假目录

class _Rec:
    """最小工具记录:路由用到 tool_id / risk / domains,去重用到 name。"""

    def __init__(self, tid, risk="L0", domains=()):
        self.tool_id = tid
        self.name = tid.split("#")[0]
        self.risk = risk
        self.domains = list(domains)


class _Catalog:
    def __init__(self, recs):
        self.records = recs

    def retrieve(self, text, domains, k, actions=None, domain_probs=None, **kw):
        return [(r, 1.0 - i * 0.01) for i, r in enumerate(self.records[:k])]


class _Head:
    """text -> 预置的 HeadOutput,不训练、不编码。"""

    def __init__(self, outs):
        self._outs = outs

    def predict(self, texts):
        return [self._outs[t] for t in texts]


def _none_out(none_prob, confidence=None):
    probs = {"none": none_prob, "single": 0.05, "multi": 0.05}
    return HeadOutput(need="none", need_probs=probs, domain_probs={"file": 0.9, "code": 0.2},
                      confidence=confidence if confidence is not None else none_prob)


def _single_out():
    return HeadOutput(need="single", need_probs={"none": 0.1, "single": 0.8, "multi": 0.1},
                      domains=["file"], actions=["read"], domain_probs={"file": 0.9},
                      action_probs={"read": 0.9}, confidence=0.8)


def _router(outs, threshold=None):
    cat = _Catalog([_Rec("Svc/a#1", "L1"), _Rec("Svc/a#2", "L0"), _Rec("Svc/b", "L0")])
    return Router(_Head(outs), cat, use_filter=False, soft_need_threshold=threshold)


TEXTS = ("low", "high", "single", "noprob", "missing")


def _outs():
    return {"low": _none_out(0.4),        # none 低置信,阈值 0.5 时软触发
            "high": _none_out(0.9),       # none 高置信,阈值 0.5 时不触发
            "single": _single_out(),      # 不是 none,与阈值无关
            "noprob": HeadOutput(need="none", need_probs={}, domain_probs={}),   # 拿不到概率
            "missing": HeadOutput(need="none", need_probs={"single": 0.6, "multi": 0.4},
                                  domain_probs={})}   # 概率里没有 none 键


# ---------------------------------------------------------------- A2 默认行为不变

def test_default_and_explicit_none_are_identical():
    """不传与显式传 None 两种构造,每条请求逐字段一致,且都不软触发。"""
    base = _router(_outs())
    explicit = _router(_outs(), threshold=None)
    for text in TEXTS:
        a = base.route(text).to_dict()
        b = explicit.route(text).to_dict()
        a.pop("latency_ms")
        b.pop("latency_ms")
        assert a == b, text
        assert a["soft_candidates"] is False, text
        # 判成 none 时,默认构造不检索,候选为空
        if a["need"] == "none":
            assert a["candidates"] == []


def test_none_threshold_never_attaches_candidates():
    r = _router(_outs(), threshold=None)
    for text in TEXTS:
        res = r.route(text)
        assert res.soft_candidates is False
        assert res.candidates == [] if res.need == "none" else res.candidates != []


# ---------------------------------------------------------------- A3 软触发条件

def test_soft_trigger_only_for_low_confidence_none():
    """阈值 0.5:只有 none 且 none 概率 0.4 < 0.5 的请求附带候选;need 字段不变。"""
    r = _router(_outs(), threshold=0.5)

    low = r.route("low")
    assert low.need == "none"                 # need 字段保持 none
    assert low.soft_candidates is True
    assert [c["tool_id"] for c in low.candidates] == ["Svc/a#1", "Svc/a#2", "Svc/b"]
    assert low.domains == [] and low.actions == []

    high = r.route("high")                     # 0.9 不低于 0.5,不触发
    assert high.need == "none"
    assert high.soft_candidates is False
    assert high.candidates == []

    single = r.route("single")                 # 不是 none:照常检索,但没有软标记
    assert single.need == "single"
    assert single.soft_candidates is False
    assert single.candidates != []


def test_threshold_1_01_always_attaches_on_none():
    """1.01 相当于"判成 none 就一律附带候选"(概率最大 1.0,永远低于阈值)。"""
    r = _router(_outs(), threshold=1.01)
    for text in ("low", "high"):
        res = r.route(text)
        assert res.need == "none"
        assert res.soft_candidates is True
        assert res.candidates != []


def test_threshold_below_prob_does_not_trigger():
    """阈值 0.3:0.4 不低于 0.3,不触发;说明比较的是严格小于。"""
    r = _router(_outs(), threshold=0.3)
    res = r.route("low")
    assert res.soft_candidates is False
    assert res.candidates == []


def test_missing_need_probability_does_not_trigger():
    """need_probs 为空或没有 none 键时当作拿不到概率,不触发。"""
    r = _router(_outs(), threshold=1.01)
    for text in ("noprob", "missing"):
        res = r.route(text)
        assert res.need == "none"
        assert res.soft_candidates is False
        assert res.candidates == []


def test_soft_with_filter_on_and_dedupe():
    """软过滤开 + 按名去重时,软触发照样出候选,need 仍为 none。"""
    outs = {"low": _none_out(0.4)}
    cat = _Catalog([_Rec("Svc/a#1", "L2"), _Rec("Svc/a#2", "L1"), _Rec("Svc/b", "L0")])
    router = Router(_Head(outs), cat, top_k=2, use_filter=True, dedupe_by_name=True,
                    soft_need_threshold=0.5)
    res = router.route("low")
    assert res.need == "none"
    assert res.soft_candidates is True
    assert [c["tool_id"] for c in res.candidates] == ["Svc/a#1", "Svc/b"]   # 同名只留最高分


def test_no_retrieval_when_catalog_empty():
    """空目录时不崩:软触发但候选列表为空。"""
    router = Router(_Head({"low": _none_out(0.4)}), _Catalog([]), use_filter=False,
                    soft_need_threshold=0.5)
    res = router.route("low")
    assert res.soft_candidates is True and res.candidates == []


# ---------------------------------------------------------------- 真实分类头 / 目录

@pytest.fixture(scope="module")
def real_system():
    tools = ev.load_jsonl(DATA / "tools.jsonl")
    train = ev.load_jsonl(DATA / "requests_train.jsonl")
    return tools, train


def test_real_system_soft_need_keeps_need_and_attaches(real_system):
    """真实合成系统:阈值为 None 时结果不变;设 1.01 后 none 请求附带候选、need 不变。"""
    tools, train = real_system
    base = ev.build_system("tfidf", tools, train)
    soft = ev.build_system("tfidf", tools, train)
    soft.soft_need_threshold = 1.01
    for text in ("你好呀,今天天气不错", "解释一下精确率和召回率", "给张三发封邮件"):
        a, b = base.route(text), soft.route(text)
        assert a.need == b.need                       # 软 need 不改 need
        assert a.soft_candidates is False
        if b.need == "none":
            assert b.soft_candidates is True
            assert b.candidates != []
        else:
            assert b.soft_candidates is False


# ---------------------------------------------------------------- A4 评估结构

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


def test_soft_need_eval_structure_and_render():
    """阈值网格评估在假数据上跑通:结构齐全,None 不软触发,报告与 json 都出得来。"""
    probes = [{"id": "p1", "text": "你好呀"}, {"id": "p2", "text": "写一篇博客"}]
    res = tr.soft_need_eval(_real_tools(), _labeled_rows(), _zh_rows(), _requests(),
                            _synth_tools(), _synth_train(), [], thresholds=(None, 0.5, 1.01),
                            folds=5, seed=0, probes=probes)
    req_ids = {r["id"] for r in _requests()}
    for t in (None, 0.5, 1.01):
        assert set(res["per"][t]) == req_ids
        s = tr.t020_summary(res, res["dev_ids"], t)
        assert 0.0 <= s["zh_r5_strict"] <= 1.0
        assert s["latency_mean"] >= 0.0
    # None 基线从不软触发
    assert all(not res["per"][None][i]["zh_soft"] for i in req_ids)
    assert res["n_probes"] == 2
    report = tr.render_t020(res)
    for section in ("# 低置信度的 none", "## 0. 结论", "## 1. 阈值网格", "## 2. 误挂率",
                    "## 3. 测试折复核"):
        assert section in report
    assert "None(基线)" in report and "1.01" in report
    json.dumps(tr.t020_metrics_json(res), ensure_ascii=False)


def test_t020_pick_respects_false_attach_gate():
    """选择规则:误挂超门槛的阈值被排除;并列时取阈值小的,None 视作不启用。"""
    # 手造 res:阈值 0.5 中文 R@5 最高但误挂 40%(超门槛),0.7 次高且误挂 5%
    def summ(r5):
        return {"zh_r5_strict": r5, "zh_r5_name": r5, "en_r5_strict": r5, "en_r5_name": r5,
                "zh_pred_none": 0, "en_pred_none": 0, "zh_soft": 0, "en_soft": 0,
                "latency_ms": 1.0}
    res = {"thresholds": [None, 0.5, 0.7], "dev_ids": ["q0"], "test_ids": ["q1"],
           "requests": [{"id": "q0"}], "per": {}, "false_attach": {None: 0.0, 0.5: 0.40, 0.7: 0.05},
           "probe_hits": {None: 0, 0.5: 192, 0.7: 24}, "holdout_attach": {None: 0.0, 0.5: 0.4, 0.7: 0.05},
           "n_probes": 480, "n_probes_in_train": 393, "n_probes_holdout": 87}
    res["per"] = {t: {"q0": dict(summ(r5))} for t, r5 in ((None, 0.10), (0.5, 0.30), (0.7, 0.20))}
    sel, _ = tr._t020_pick(res)
    assert sel == 0.7        # 0.5 误挂 40% 被排除;0.7 胜出

    # 全都超门槛(只留 None) -> None(不启用)
    res["false_attach"] = {None: 0.0, 0.5: 0.40, 0.7: 0.30}
    sel, _ = tr._t020_pick(res)
    assert sel is None

