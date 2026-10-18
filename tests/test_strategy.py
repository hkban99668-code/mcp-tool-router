"""组合策略的测试:纯函数 `apply_strategy` 的每个分支,以及 Router 怎么用它。

全部用 FakeJevClient,不联网、不需要 key。
"""

import pathlib

import pytest

from mcpcls import evaluate as ev
from mcpcls.heads import HeadOutput
from mcpcls.jev import FakeJevClient, jev_fallback, request_questions
from mcpcls.router import Router
from mcpcls.strategy import STRATEGIES, TRIGGERS, apply_strategy

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

def _probs(top, classes):
    return {c: top.get(c, 0.1) for c in classes}


def _local(need="single", domains=("comms",), actions=("send",), conf=0.9):
    """造一个本地分类头输出。need=none 时 head 不填 domains / actions(与 heads.py 一致)。"""
    from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS

    return HeadOutput(
        need=need,
        domains=[] if need == "none" else list(domains),
        actions=[] if need == "none" else list(actions),
        domain_probs=_probs({"comms": 0.9, "file": 0.6}, DOMAIN_IDS),
        action_probs=_probs({"send": 0.8, "read": 0.4}, ACTION_IDS),
        confidence=conf)


def _labels(need="single", domains=("media",), actions=("create",)):
    """造一份云端答案。`answers_to_labels` 在 need=none 时把 domains / actions 置空。"""
    return {"need": need,
            "domains": [] if need == "none" else list(domains),
            "actions": [] if need == "none" else list(actions),
            "confidence": 0.6}


@pytest.fixture(scope="module")
def router():
    return ev.build_system("tfidf", ev.load_jsonl(DATA / "tools.jsonl"),
                           ev.load_jsonl(DATA / "requests_train.jsonl"))


# ---------------- A3: 四种策略 × labels 为 None / none / single ----------------

def test_local_strategy_ignores_labels():
    for labels in (None, _labels("none"), _labels("single")):
        out = apply_strategy(_local(), labels, "local")
        assert out == {"need": "single", "domains": ["comms"], "actions": ["send"], "used": False}


def test_replace_strategy_takes_everything_from_labels():
    out = apply_strategy(_local(), _labels("single", ["media"], ["create"]), "replace")
    assert out == {"need": "single", "domains": ["media"], "actions": ["create"], "used": True}
    # need=none 时连 domains / actions 一起被清掉
    assert apply_strategy(_local(), _labels("none"), "replace") == {
        "need": "none", "domains": [], "actions": [], "used": True}


def test_need_only_takes_need_from_labels_and_rest_from_local():
    out = apply_strategy(_local(domains=["comms"], actions=["send"]),
                         _labels("multi", ["media"], ["create"]), "need_only")
    assert out["need"] == "multi" and out["domains"] == ["comms"] and out["actions"] == ["send"]
    assert out["used"] is True


def test_need_only_clears_domains_when_labels_say_none():
    out = apply_strategy(_local(), _labels("none"), "need_only")
    assert out == {"need": "none", "domains": [], "actions": [], "used": True}


def test_need_only_falls_back_to_top_local_probability():
    """本地判了 none(没有 domains / actions)时,取本地概率最高的域和动作。"""
    out = apply_strategy(_local("none"), _labels("single"), "need_only")
    assert out["need"] == "single"
    assert out["domains"] == ["comms"] and out["actions"] == ["send"]     # 概率最高的那两个


def test_need_only_uses_labels_actions_without_action_probs():
    """HeadOutput 没有 action_probs 时退到云端答案的动作(有概率表时仍取本地最高的那个)。"""
    local = HeadOutput(need="none", domain_probs={"comms": 0.9, "file": 0.6}, action_probs={})
    out = apply_strategy(local, _labels("single", ["media"], ["create"]), "need_only")
    assert out["actions"] == ["create"] and out["domains"] == ["comms"]

    with_probs = HeadOutput(need="none", domain_probs={"comms": 0.9}, action_probs={"send": 0.7})
    assert apply_strategy(with_probs, _labels("single", ["media"], ["create"]),
                          "need_only")["actions"] == ["send"]


def test_both_none_only_when_both_say_none():
    cases = [
        ("none", "none", "none", [], []),                      # 两边都 none
        ("none", "single", "single", ["comms"], ["send"]),     # 只有本地 none:用云端(域取本地最高概率)
        ("single", "none", "single", ["comms"], ["send"]),     # 只有云端 none:用本地
        ("multi", "single", "multi", ["comms"], ["send"]),     # 都非 none:用本地
    ]
    for local_need, label_need, want_need, want_domains, want_actions in cases:
        out = apply_strategy(_local(local_need), _labels(label_need), "both_none")
        assert (out["need"], out["domains"], out["actions"]) == (want_need, want_domains,
                                                                 want_actions), (local_need, label_need)


def test_every_strategy_falls_back_to_local_without_labels():
    for strategy in STRATEGIES:
        out = apply_strategy(_local(), None, strategy)
        assert out == {"need": "single", "domains": ["comms"], "actions": ["send"], "used": False}


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        apply_strategy(_local(), _labels(), "whatever")


# ---------------- Router 用策略与触发方式 ----------------

def _fake(need="single", domains=("media",), actions=("create",)):
    from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS

    def answers_fn(state, questions):
        a = {"need": {"type": "choice", "choice": need,
                      "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": 0.6}}
        for d in DOMAIN_IDS:
            a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
        for x in ACTION_IDS:
            a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
        return a

    return FakeJevClient(answers_fn)


def test_router_need_only_keeps_local_domains(router):
    """同样的答案,replace 会连域一起换掉,need_only 只换 need。"""
    text = "给张三发封邮件说合同签了"
    for strategy, want_domains in (("replace", ["media"]), ("need_only", ["comms"])):
        client = _fake("single")
        router.threshold, old_t = 1.01, router.threshold
        router.llm_fallback, old_f = jev_fallback(client), router.llm_fallback
        router.fallback_strategy, router.fallback_trigger = strategy, "always"
        try:
            r = router.route(text)
            assert r.need == "single" and r.domains == want_domains, strategy
        finally:
            router.threshold, router.llm_fallback = old_t, old_f
            router.fallback_strategy, router.fallback_trigger = "replace", "low_conf"


def test_router_local_strategy_never_asks(router):
    client = _fake()
    router.threshold, old_t = 1.01, router.threshold
    router.llm_fallback, old_f = jev_fallback(client), router.llm_fallback
    router.fallback_strategy, router.fallback_trigger = "local", "always"
    try:
        r = router.route("给张三发封邮件说合同签了")
        assert r.need != "none" and "comms" in r.domains
        assert client.calls == [] and client.states == []
        assert r.source == "model"
    finally:
        router.threshold, router.llm_fallback = old_t, old_f
        router.fallback_strategy, router.fallback_trigger = "replace", "low_conf"


def test_router_trigger_always_asks_even_when_confident(router):
    client = _fake()
    router.threshold, old_t = 0.0, router.threshold     # 置信度永远够,靠 trigger 触发
    router.llm_fallback, old_f = jev_fallback(client), router.llm_fallback
    router.fallback_trigger = "always"
    try:
        r = router.route("解释一下精确率和召回率")
        assert len(client.calls) == 1
        assert r.need == "single" and r.source == "jev_fallback"     # replace 策略整体替换
    finally:
        router.threshold, router.llm_fallback = old_t, old_f
        router.fallback_trigger = "low_conf"


def test_router_rejects_unknown_strategy_or_trigger(router):
    with pytest.raises(ValueError):
        Router(router.head, router.catalog, fallback_strategy="nope")
    with pytest.raises(ValueError):
        Router(router.head, router.catalog, fallback_trigger="sometimes")


def test_router_defaults_keep_t004_behaviour(router):
    r = Router(router.head, router.catalog)
    assert (r.fallback_strategy, r.fallback_trigger) == ("replace", "low_conf")
    assert STRATEGIES == ("local", "replace", "need_only", "both_none")
    assert TRIGGERS == ("low_conf", "always")
    # 题数跟着 taxonomy 走(T-009 起 14 个域 = 21 题),不写死
    from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS

    assert len(request_questions("v1")) == 1 + len(DOMAIN_IDS) + len(ACTION_IDS)
