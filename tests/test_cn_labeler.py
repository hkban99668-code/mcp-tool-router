"""T-017 国产第二路打标:客户端小修 + pilot 选型逻辑的离线测试。

全部离线(autouse fixture 把没被单独 mock 的请求打成 AssertionError),
只覆盖不需要联网的部分:缓存名、解析失败计数、token 记录、pilot 抽样与选型规则。
"""

import json
import pathlib
import urllib.request

import pytest

from mcpcls import compare, realdata as rd, tools_real as tr
from mcpcls.jev import JevError, request_questions
from mcpcls.llm_or import OpenRouterChatClient
from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS

FAKE_KEY = "TEST-KEY-DO-NOT-USE"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("测试里不应发出任何网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path)


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(monkeypatch, seq):
    seen = []

    def fake(req, timeout=None):
        seen.append(req)
        item = seq[min(len(seen) - 1, len(seq) - 1)]
        if isinstance(item, Exception):
            raise item
        return _Resp(json.dumps(item).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


def _content(need="single", domains=("comms",), actions=("send",)):
    obj = {"need": need}
    for d in DOMAIN_IDS:
        obj[f"domain_{d}"] = d in domains
    for a in ACTION_IDS:
        obj[f"action_{a}"] = a in actions
    return json.dumps(obj, ensure_ascii=False)


def _chat(content, usage=None):
    return {"choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage or {}}


def _fake_answers(need="single", domains=("comms",), actions=("send",)):
    a = {"need": {"type": "choice", "choice": need,
                  "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": 0.6}}
    for d in DOMAIN_IDS:
        a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
    return a


# ---------------- A2:缓存名 ----------------

def test_haiku_cache_names_unchanged():
    """Haiku 的旧缓存名一个都不变,旧缓存照常命中。"""
    assert OpenRouterChatClient(api_key=FAKE_KEY).cache_name == "haiku@claude-haiku-5-5"
    assert OpenRouterChatClient(api_key=FAKE_KEY,
                                model="anthropic/claude-haiku-5.5").cache_name == "haiku"


def test_cn_model_cache_name_has_no_slash():
    """国产模型缓存名带 chat@ 前缀,且不含 /(文件名要能直接落盘)。"""
    name = OpenRouterChatClient(api_key=FAKE_KEY,
                                model="deepseek/deepseek-v4.1-flash").cache_name
    assert name == "chat@deepseek-deepseek-v4.1-flash"
    assert "/" not in name


def test_openai_body_has_big_max_tokens(monkeypatch):
    """OpenAI 格式请求体给足 max_tokens(推理模型会被截断),usage.include 照旧带上。"""
    seen = _mock_urlopen(monkeypatch, [_chat(_content())])
    OpenRouterChatClient(api_key=FAKE_KEY, api_format="openai").decide(
        "给张三发封邮件", request_questions("v1"))
    body = json.loads(seen[0].data.decode("utf-8"))
    assert body["max_tokens"] == 8192
    assert body["usage"] == {"include": True}


# ---------------- A3:解析失败计入失败数 ----------------

def test_truncated_json_counts_as_failure(monkeypatch):
    truncated = _content()[:20]          # 推理太长被截断的典型形态:JSON 只吐了一半
    _mock_urlopen(monkeypatch, [_chat(truncated)])
    client = OpenRouterChatClient(api_key=FAKE_KEY, retries=0)
    with pytest.raises(JevError):
        client.decide("给张三发封邮件", request_questions("v1"))
    assert len(client.calls) == 1 and client.calls[0].ok is False


def test_reasoning_wrapped_json_counts_as_failure(monkeypatch):
    wrapped = "先看这条请求……\n" + _content() + "\n以上是判断结果。"
    _mock_urlopen(monkeypatch, [_chat(wrapped)])
    client = OpenRouterChatClient(api_key=FAKE_KEY, retries=0)
    with pytest.raises(JevError):
        client.decide("给张三发封邮件", request_questions("v1"))
    assert len(client.calls) == 1 and client.calls[0].ok is False


def test_compare_ask_records_failure_not_empty_answer():
    """compare._ask 解析失败时不能静默当空答案:ok=False 且计入失败。"""

    class Boom:
        api_key = ""

        def __init__(self):
            self.calls = []

        def decide(self, state, questions):
            raise JevError("模型输出不是 JSON")

    rec = compare._ask(Boom(), "给张三发封邮件", "v1")
    assert rec["ok"] is False and rec["answers"] is None and "JSON" in rec["error"]


# ---------------- token 记录 ----------------

def test_token_usage_recorded_in_call_and_cache(monkeypatch):
    usage = {"prompt_tokens": 1831, "completion_tokens": 1414,
             "completion_tokens_details": {"reasoning_tokens": 1282}}
    _mock_urlopen(monkeypatch, [_chat(_content(), usage)])
    client = OpenRouterChatClient(api_key=FAKE_KEY, api_format="openai", retries=0)
    client.decide("给张三发封邮件", request_questions("v1"))
    call = client.calls[0]
    assert (call.prompt_tokens, call.completion_tokens, call.reasoning_tokens) == (1831, 1414, 1282)
    assert call.input_tokens == 1831                    # 旧字段兼容

    rec = compare._ask(client, "给张三发封邮件", "v1")
    assert (rec["prompt_tokens"], rec["completion_tokens"], rec["reasoning_tokens"]) == \
        (1831, 1414, 1282)


# ---------------- pilot 抽样 ----------------

def _synth_pool(n_per_source=5):
    pool = []
    for src in ("bfcl", "apibank", "alpaca_zh"):
        for i in range(n_per_source):
            pool.append({"id": f"{src}-{i}", "text": f"{src} 的请求 {i}", "source": src})
    return pool


def test_pilot_ids_stratified_and_deterministic():
    pool = _synth_pool(5)
    ids = tr.pilot_ids(pool, n=6, seed=0)
    assert len(ids) == 6
    assert ids == tr.pilot_ids(pool, n=6, seed=0)        # 固定 seed 可复现
    from collections import Counter
    assert Counter(i.split("-")[0] for i in ids) == {"bfcl": 2, "apibank": 2, "alpaca_zh": 2}


def test_pilot_ids_payload_shape():
    payload = tr.pilot_ids_payload(_synth_pool(5), n=6, seed=1)
    assert payload["n"] == 6 and payload["seed"] == 1
    assert payload["by_source"] == {"alpaca_zh": 2, "apibank": 2, "bfcl": 2}


# ---------------- 选型规则 ----------------

def _row(model, rate, total_tokens, is_candidate=True, n=210):
    return {"model": model, "is_candidate": is_candidate,
            "metrics": {"agree_rate": rate, "tokens": {"total": total_tokens},
                        "both_answered": n}}


def test_select_prefers_highest_main_metric():
    rows = [_row("haiku", 0.90, 10, is_candidate=False),
            _row("A", 0.80, 50), _row("B", 0.90, 500)]
    pick, tied = tr.select_cn_model(rows, total=210)
    assert pick["model"] == "B"


def test_select_tie_within_one_sample_picks_fewer_tokens():
    # 主指标差 0.00238 * 210 ≈ 0.5 条样本,落在 1 条以内 → 并列 → 取 token 少的
    rows = [_row("A", 0.80238, 300), _row("B", 0.80, 100)]
    pick, tied = tr.select_cn_model(rows, total=210)
    assert {r["model"] for r in tied} == {"A", "B"}
    assert pick["model"] == "B"


def test_select_ignores_haiku_reference_row():
    rows = [_row("haiku", 0.99, 1, is_candidate=False),
            _row("A", 0.70, 10), _row("B", 0.72, 10)]
    pick, _ = tr.select_cn_model(rows, total=210)
    assert pick["model"] == "B"


def test_select_excludes_undercovered_candidate():
    """有效条数太少的候选(如被限流只答上 20 条)主指标不可比,不参与选择。"""
    tiny = {"model": "tiny", "is_candidate": True,
            "metrics": {"agree_rate": 0.99, "both_answered": 20, "tokens": {"total": 1}}}
    rows = [_row("big", 0.70, 10), tiny]
    pick, tied = tr.select_cn_model(rows, total=210)
    assert pick["model"] == "big" and {r["model"] for r in tied} == {"big"}


# ---------------- 指标计算 ----------------

def _rec(text, answers, ok=True):
    return {"text": text, "answers": answers, "ok": ok, "error": "", "latency_ms": 100.0,
            "prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 2}


def test_side_metrics_primary_agreement_with_jev():
    items = [{"id": "x1", "text": "t1", "source": "bfcl"}]
    jev = {"t1": _rec("t1", _fake_answers("single", ["comms"]))}
    cand = [_rec("t1", _fake_answers("single", ["comms"]))]
    haiku = {"t1": _rec("t1", _fake_answers("single", ["comms"]))}
    m = tr._side_metrics(items, cand, jev, haiku, {"x1": "single"})
    assert m["both_answered"] == 1 and m["agree"] == 1 and m["agree_rate"] == 1.0
    assert m["silver_hit"] == 1 and m["silver_covered"] == 1
    assert m["domain_hit"] == 1 and m["domain_denom"] == 1
    assert m["tokens"] == {"prompt": 10.0, "completion": 5.0, "reasoning": 2.0, "total": 15.0}


def test_side_metrics_disagreement_not_counted():
    items = [{"id": "x1", "text": "t1", "source": "bfcl"}]
    jev = {"t1": _rec("t1", _fake_answers("none", []))}        # Jev 判 none
    cand = [_rec("t1", _fake_answers("single", ["comms"]))]    # 候选判 single
    m = tr._side_metrics(items, cand, jev, jev, {"x1": "none"})
    assert m["both_answered"] == 1 and m["agree"] == 0 and m["agree_rate"] == 0.0


def test_render_cn_pilot_table():
    res = {"n": 210, "need_version": "v3", "picked": "A", "tied": ["A"],
           "by_source": {"bfcl": 70, "apibank": 70, "alpaca_zh": 70},
           "rows": [{"model": "haiku", "is_candidate": False, "stats": {"asked": 0, "failed": 0},
                     "metrics": tr._side_metrics(
                         [{"id": "x", "text": "t", "source": "bfcl"}],
                         [_rec("t", _fake_answers("single", ["comms"]))],
                         {"t": _rec("t", _fake_answers("single", ["comms"]))},
                         {"t": _rec("t", _fake_answers("single", ["comms"]))}, {"x": "single"})},
                    {"model": "A", "is_candidate": True, "stats": {"asked": 210, "failed": 0},
                     "metrics": tr._side_metrics(
                         [{"id": "x", "text": "t", "source": "bfcl"}],
                         [_rec("t", _fake_answers("single", ["comms"]))],
                         {"t": _rec("t", _fake_answers("single", ["comms"]))},
                         {"t": _rec("t", _fake_answers("single", ["comms"]))}, {"x": "single"})}]}
    text = tr.render_cn_pilot(res)
    assert "小样本选型" in text and "A **(选中)**" in text


def test_realdata_labels_helper_is_defined():
    """pilot 复用了 realdata 的标签/主域口径,确认这些接口还在。"""
    assert callable(rd._labels) and callable(rd.primary_domain) and callable(rd.agreement_outcome)
