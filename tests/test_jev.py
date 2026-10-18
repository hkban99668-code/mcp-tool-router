"""Jev 兜底的测试:全部用 FakeJevClient 或 mock urlopen,不联网、不需要 key。

假 key 一律写成不存在的字符串;源码里不许出现 key,由 test_no_api_key_in_sources 盯住
(它扫的是源码目录,不扫 .collab/,任务卡里 Claude 写的验收标准文字不在检查范围内)。
"""

import json
import pathlib
import re
import socket
import time
import urllib.error
import urllib.request

import pytest

from mcpcls import cli
from mcpcls import evaluate as ev
from mcpcls.jev import (DEFAULT_JEV_MODEL, NEED_VERSIONS, FakeJevClient, JevClient, JevError,
                        JevUnavailable, answers_to_labels, jev_fallback, need_question,
                        request_questions)
from mcpcls.taxonomy import ACTIONS, ACTION_IDS, DOMAINS, DOMAIN_IDS, NEED, NEED_IDS

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
FAKE_KEY = "TEST-KEY-DO-NOT-USE"   # 故意不像真实 key,免得 key 形状的 grep 命中自己


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """整个文件都不许联网:没有被单独 mock 的测试一旦碰到 urlopen 就失败。"""
    def boom(*a, **kw):
        raise AssertionError("测试里不应发出任何网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture(scope="module")
def gold():
    return ev.load_jsonl(DATA / "requests_gold.jsonl")


@pytest.fixture(scope="module")
def router():
    return ev.build_system("tfidf", ev.load_jsonl(DATA / "tools.jsonl"),
                           ev.load_jsonl(DATA / "requests_train.jsonl"))


# ---------------- 假响应与 mock ----------------

def _answers(need="single", domains=(), actions=(), confidence=0.6):
    """按 Jev 的响应形状造一份 answers。"""
    a = {"need": {"type": "choice", "choice": need,
                  "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": confidence}}
    for d in DOMAIN_IDS:
        a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
    return a


def _fake(answers=None, **kw):
    return FakeJevClient(lambda state, questions: _answers() if answers is None else answers, **kw)


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
    """seq 每项是要返回的响应体(dict)或要抛的异常;用完之后一直重复最后一项。"""
    seen = []

    def fake(req, timeout=None):
        seen.append(req)
        item = seq[min(len(seen) - 1, len(seq) - 1)]
        if isinstance(item, Exception):
            raise item
        return _Resp(json.dumps(item).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


# ---------------- A2: key 只在参数或环境变量里 ----------------

def test_client_requires_key_at_construction(monkeypatch):
    """key 缺失要在构造时就报,不能等到第一次调用才炸。"""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(JevUnavailable):
        JevClient()
    assert JevClient(api_key=FAKE_KEY).model == DEFAULT_JEV_MODEL
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    assert JevClient().api_key == FAKE_KEY


def test_no_api_key_in_sources():
    """源码与文档里不能有 key。匹配式不写字面量,免得这个测试自己把 key 引进仓库。"""
    root = pathlib.Path(__file__).resolve().parent.parent
    files = [p for d in ("mcpcls", "tests", "scripts") for p in (root / d).rglob("*")
             if p.is_file() and p.suffix in (".py", ".md")]
    files += list(root.glob("*.md"))
    key_like = re.compile(r"sk-[a-z]{2}-[A-Za-z0-9]{8,}")
    bad = [str(p) for p in files if key_like.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert bad == []


def test_error_and_call_record_never_contain_key(monkeypatch):
    """异常信息和 JevCall.error 里都不能有 key,即使底层错误把请求头带了出来。"""
    leaky = urllib.error.URLError(f"请求失败,Authorization: Bearer {FAKE_KEY}")
    _mock_urlopen(monkeypatch, [leaky])
    client = JevClient(api_key=FAKE_KEY, retries=1)
    with pytest.raises(JevError) as e:
        client.decide("给张三发邮件", request_questions())
    assert FAKE_KEY not in str(e.value)
    assert len(client.calls) == 1 and client.calls[0].ok is False
    assert FAKE_KEY not in client.calls[0].error


# ---------------- A3: 超时与重试 ----------------

def test_retry_then_success_records_one_ok_call(monkeypatch):
    payload = {"answers": _answers(need="single", domains=["comms"]),
               "usage": {"cost": 1.8e-5, "input_tokens": 438, "output_tokens": 56}}
    _mock_urlopen(monkeypatch, [socket.timeout("timed out"), payload])
    client = JevClient(api_key=FAKE_KEY, retries=1)
    answers = client.decide("给张三发封邮件", need_question("v1"))
    assert answers["need"]["choice"] == "single"
    assert len(client.calls) == 1 and client.calls[0].ok is True
    assert client.calls[0].cost == pytest.approx(1.8e-5)
    assert client.calls[0].input_tokens == 438


def test_both_attempts_fail_records_one_failed_call(monkeypatch):
    err = urllib.error.HTTPError(DEFAULT_JEV_MODEL, 429, "Too Many Requests", {}, None)
    _mock_urlopen(monkeypatch, [err])
    client = JevClient(api_key=FAKE_KEY, retries=1)
    with pytest.raises(JevError):
        client.decide("给张三发封邮件", need_question("v1"))
    assert len(client.calls) == 1 and client.calls[0].ok is False
    assert "429" in client.calls[0].error


def test_response_without_answers_is_an_error(monkeypatch):
    _mock_urlopen(monkeypatch, [{"model": "typesafe/jev-1.13"}])
    with pytest.raises(JevError):
        JevClient(api_key=FAKE_KEY, retries=0).decide("给张三发封邮件", need_question("v1"))


def test_latency_counts_the_failed_attempt_too(monkeypatch):
    """T-005 第 1 个小问题:重试成功时 latency 要含失败那次的耗时,否则端到端延迟被低估。"""
    payload = {"answers": _answers(need="single"), "usage": {"cost": 1e-5, "input_tokens": 100}}
    clock = {"t": 0.0}
    attempts = []

    def fake_perf_counter():
        return clock["t"]

    def fake(req, timeout=None):
        attempts.append(1)
        clock["t"] += 12.0 if len(attempts) == 1 else 3.0     # 第一次超时耗掉 12 秒
        if len(attempts) == 1:
            raise socket.timeout("timed out")
        return _Resp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = JevClient(api_key=FAKE_KEY, retries=1)
    client.decide("给张三发封邮件", need_question("v1"))
    assert len(attempts) == 2
    assert client.calls[0].ok is True
    assert client.calls[0].latency_ms == pytest.approx(15000.0)     # 12 + 3 秒


def test_default_timeout_is_30_seconds(monkeypatch):
    """第 3 个小问题:全量 Jev 的 p95 约 16 秒,默认超时从 15 提到 30。"""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert JevClient(api_key=FAKE_KEY).timeout == 30.0
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    assert JevClient().timeout == 30.0


# ---------------- A4: 问题构造与答案解析 ----------------

def test_request_questions_come_from_taxonomy():
    q = request_questions()
    # 题数跟着 taxonomy 走(T-009 起 14 个域 = 21 题),不把数字写死在测试里
    assert len(q) == 1 + len(DOMAIN_IDS) + len(ACTION_IDS)
    assert q["need"]["type"] == "choice" and set(q["need"]["criteria"]) == set(NEED_IDS)
    for c in DOMAINS:
        assert q[f"domain_{c.id}"]["type"] == "noul"
        assert c.definition in q[f"domain_{c.id}"]["instructions"]
    for c in ACTIONS:
        assert q[f"action_{c.id}"]["type"] == "noul"
        assert c.definition in q[f"action_{c.id}"]["instructions"]


# ---------------- A8: need 说明的 v1 / v2 两个版本 ----------------

# 下面三段文字逐字照抄任务卡 T-005,改一个标点都会被这条测试抓住
_NEED_V2_INSTRUCTIONS = (
    "判断这条用户请求能否只靠语言模型自身的知识完成。只要需要读取、查询或改动用户的外部数据或系统"
    "（邮件、消息、日程、任务、文件、数据库、网页、浏览器、代码仓库、云服务、订单或客户记录、音视频等），"
    "就需要工具。只按请求本身判断，不要替用户设想没说出来的需求。")
_NEED_V2_CRITERIA = {
    "none": "不需要任何外部工具：解释概念、知识问答、写作、翻译、润色、出主意、闲聊或表达观点，"
            "模型凭自身知识就能直接回答，不涉及读取或改动用户的任何外部数据。",
    "single": "需要一个外部工具完成一件事：请求里有明确的操作对象，例如发一封邮件或消息、查或改日程、"
              "建一个任务、读写某个文件、查数据库、搜网页、操作网页、查日志或部署、改订单或客户记录。"
              "即使请求很短、语气很随意，只要要对外部数据或系统做一件事，就是 single。",
    "multi": "需要先后或组合使用两个及以上不同的工具或系统，例如先查日程再发邮件、先转写录音再存成文件、"
             "先查 issue 再建待办。",
}


def test_need_v2_text_matches_the_task_card():
    q = need_question("v2")
    assert q["type"] == "choice"
    assert q["instructions"] == _NEED_V2_INSTRUCTIONS
    assert q["criteria"] == _NEED_V2_CRITERIA
    assert request_questions("v2")["need"] == q


def test_need_v1_is_unchanged_and_stays_the_default():
    q = need_question("v1")
    assert q["instructions"] == ("判断这条用户请求是否需要调用外部工具(MCP 工具)才能完成。"
                                "只按请求本身判断,不要替用户设想没说出来的需求。")
    assert q["criteria"] == {c.id: f"{c.name}。{c.definition}" for c in NEED}
    assert request_questions() == request_questions("v1")      # 不加参数仍是 v1
    assert len(request_questions("v2")) == 1 + len(DOMAIN_IDS) + len(ACTION_IDS)
    assert NEED_VERSIONS == ("v1", "v2", "v3")


def test_unknown_need_version_is_rejected():
    with pytest.raises(ValueError):
        need_question("v4")          # v3 起算合法版本,T-009 之后未知版本用 v4 试


def test_answers_to_labels_none_clears_domains_and_actions():
    labels = answers_to_labels(_answers(need="none", domains=["comms"], actions=["send"]))
    assert labels == {"need": "none", "domains": [], "actions": [], "confidence": 0.6}


def test_answers_to_labels_picks_every_domain_over_threshold():
    labels = answers_to_labels(_answers(need="multi", domains=["comms", "file"], actions=["send"]))
    assert labels["need"] == "multi"
    assert set(labels["domains"]) == {"comms", "file"}
    assert labels["actions"] == ["send"]


def test_answers_to_labels_falls_back_to_top_domain():
    """没有域过阈值时取概率最高的一个,否则路由会拿到空域。"""
    a = _answers(need="single")            # 所有域都是 0.1
    a["domain_calendar_task"] = {"type": "noul", "noul": 0.4}
    a["action_read"] = {"type": "noul", "noul": 0.45}
    labels = answers_to_labels(a)
    assert labels["domains"] == ["calendar_task"]
    assert labels["actions"] == ["read"]
    # 阈值调低后 0.45 也算命中
    assert answers_to_labels(a, noul_threshold=0.3)["actions"] == ["read"]


# ---------------- A5 + A1: Router 挂 Jev 兜底 ----------------

def test_router_uses_jev_fallback_below_threshold(router):
    client = _fake(_answers(need="single", domains=["media"], actions=["create"]))
    router.threshold, old = 1.01, router.threshold     # 强制走兜底
    router.llm_fallback = jev_fallback(client)
    try:
        r = router.route("给张三发封邮件说合同签了")
        assert r.source == "jev_fallback"
        assert (r.need, r.domains, r.actions) == ("single", ["media"], ["create"])
        assert len(client.calls) == 1 and client.calls[0].ok is True
    finally:
        router.threshold, router.llm_fallback = old, None


def test_router_keeps_local_result_when_jev_fails(router):
    """T-005:兜底抛错时保留本地结果不向上抛,但来源要标成 jev_error(不再混在 model_low_conf 里)。"""
    client = FakeJevClient(lambda s, q: _answers(), fail=True)
    router.threshold, old = 1.01, router.threshold
    router.llm_fallback = jev_fallback(client)
    try:
        r = router.route("给张三发封邮件说合同签了")
        assert r.source == "jev_error"
        assert r.need != "none" and "comms" in r.domains      # 本地结果还在
        assert len(client.calls) == 1 and client.calls[0].ok is False
    finally:
        router.threshold, router.llm_fallback = old, None


def test_router_without_fallback_stays_model_low_conf(router):
    """没配兜底、置信度又不够时仍是 model_low_conf,不能被 fallback_error 顶掉。"""
    router.threshold, old = 1.01, router.threshold
    try:
        assert router.route("随便说点什么").source == "model_low_conf"
    finally:
        router.threshold = old


def test_plain_fallback_error_is_fallback_error(router):
    """兜底函数没标来源时(比如 LLMLabeler),出错记为 fallback_error。"""
    def boom(text):
        raise RuntimeError("兜底挂了")

    router.threshold, old = 1.01, router.threshold
    router.llm_fallback = boom
    try:
        assert router.route("随便说点什么").source == "fallback_error"
    finally:
        router.threshold, router.llm_fallback = old, None


def test_router_does_not_call_jev_when_confident(router):
    client = _fake()
    router.threshold, old = 0.0, router.threshold     # 置信度永远够
    router.llm_fallback = jev_fallback(client)
    try:
        r = router.route("给张三发封邮件说合同签了")
        assert r.source == "model"
        assert client.calls == [] and client.states == []
    finally:
        router.threshold, router.llm_fallback = old, None


def test_plain_fallback_stays_llm_fallback(router):
    """兜底函数没有 source 属性时仍按 llm_fallback 标记(LLMLabeler.as_fallback 的行为不变)。"""
    router.threshold, old = 1.01, router.threshold
    router.llm_fallback = lambda text: {"need": "single", "domains": ["media"], "actions": ["read"]}
    try:
        r = router.route("随便说点什么")
        assert r.source == "llm_fallback" and r.domains == ["media"]
    finally:
        router.threshold, router.llm_fallback = old, None


def test_jev_fallback_is_tagged_with_its_source():
    assert jev_fallback(_fake()).source == "jev_fallback"


# ---------------- A7: eval_jev 两种模式 ----------------

def test_eval_jev_two_modes_with_fake_client(router, gold):
    def run(mode):
        client = _fake(_answers(need="single", domains=["comms"], actions=["send"]), cost=2e-5)
        return ev.eval_jev(router, gold, client, mode=mode), client

    fb, fb_client = run("fallback")
    full, full_client = run("full")
    for res in (fb, full):
        assert res["n"] == len(gold)
        assert 0.0 <= res["need"]["accuracy_3class"] <= 1.0
        assert 0.0 <= res["need"]["macro_f1_3class"] <= 1.0
        assert set(res["need"]["confusion"]) == set(NEED_IDS)
        assert {"micro_f1", "macro_f1"} == set(res["domain"]) == set(res["action"])
        assert res["latency_ms"]["p50"] <= res["latency_ms"]["p95"]
        assert set(res["jev"]) >= {"fallback_rate", "calls", "failures", "cost_usd",
                                   "jev_latency_ms", "input_tokens"}

    assert full["jev"]["fallback_rate"] == 1.0
    assert full["jev"]["calls"] == len(gold)
    assert full["jev"]["cost_usd"] == pytest.approx(len(gold) * 2e-5)
    assert full["jev"]["failures"] == 0
    assert fb["jev"]["calls"] == len(fb_client.calls) < full["jev"]["calls"]
    assert fb["jev"]["cost_usd"] == pytest.approx(fb["jev"]["calls"] * 2e-5)

    md = ev.render_jev_report(ev.eval_requests(router, gold), fb, full)
    assert "本地基线(不接 Jev)" in md and "全量交给 Jev" in md and "两个兜底比例不是一回事" in md


def test_eval_jev_counts_failures(router, gold):
    res = ev.eval_jev(router, gold, FakeJevClient(lambda s, q: _answers(), fail=True),
                      mode="full")
    assert res["jev"]["calls"] == len(gold) and res["jev"]["failures"] == len(gold)
    assert res["jev"]["fallback_rate"] == 0.0      # 全部回退到本地


def test_eval_jev_rejects_unknown_mode(router, gold):
    with pytest.raises(ValueError):
        ev.eval_jev(router, gold, _fake(), mode="whatever")


# ---------------- A8: 没有 key 时退出码 2 ----------------

def test_cli_live_jev_without_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert cli.main(["evaluate", "--live-jev"]) == 2
    out = capsys.readouterr()
    text = out.out + out.err
    assert "Traceback" not in text
    assert len([l for l in text.splitlines() if l.strip()]) == 1     # 只有一行提示
    assert "OPENROUTER_API_KEY" in text


# ---------------- 可配置的服务商(JEV_BASE_URL / JEV_MODEL / JEV_API_KEY) ----------------

def _clear_provider_env(monkeypatch):
    for k in ("JEV_BASE_URL", "JEV_MODEL", "JEV_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_default_provider_is_commandcode(monkeypatch):
    """什么都不设时走 commandcode 网关的 typesafe/jev(2026-10-10 起的默认)。"""
    _clear_provider_env(monkeypatch)
    seen = _mock_urlopen(monkeypatch, [{"answers": _answers(), "usage": {}}])
    client = JevClient(api_key=FAKE_KEY)
    client.decide("给张三发邮件", request_questions())
    assert seen[0].full_url == "https://api.commandcode.ai/provider/v1/systemone"
    assert json.loads(seen[0].data)["model"] == "typesafe/jev"


def test_cache_name_separates_models(monkeypatch):
    """旧的 jev-1.13 缓存仍叫 jev;新模型换缓存名,新旧答案不会混在一个文件里。"""
    _clear_provider_env(monkeypatch)
    assert JevClient(api_key=FAKE_KEY, model="typesafe/jev-1.13").cache_name == "jev"
    assert JevClient(api_key=FAKE_KEY).cache_name == "jev@typesafe-jev"


def test_base_url_model_and_key_from_env(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("JEV_BASE_URL", "https://relay.example/provider/v1/")
    monkeypatch.setenv("JEV_MODEL", "typesafe/jev")
    monkeypatch.setenv("JEV_API_KEY", "user_fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-fake")
    seen = _mock_urlopen(monkeypatch, [{"answers": _answers(), "usage": {}}])
    client = JevClient()
    client.decide("给张三发邮件", request_questions())
    req = seen[0]
    assert req.full_url == "https://relay.example/provider/v1/systemone"
    assert json.loads(req.data)["model"] == "typesafe/jev"
    assert req.get_header("Authorization") == "Bearer user_fake"   # JEV_API_KEY 优先


def test_requests_carry_a_non_default_user_agent(monkeypatch):
    """有的网关会拦 Python-urllib 默认 UA(Cloudflare 1010),所以要带自己的 UA。"""
    _clear_provider_env(monkeypatch)
    seen = _mock_urlopen(monkeypatch, [{"answers": _answers(), "usage": {}}])
    JevClient(api_key=FAKE_KEY).decide("给张三发邮件", request_questions())
    ua = seen[0].get_header("User-agent")
    assert ua and "Python-urllib" not in ua



def test_more_than_20_questions_are_split_into_calls(monkeypatch):
    """Jev 一次最多 20 题;14 域的 v3 有 21 题,要拆成两次调用再合并答案。"""
    from mcpcls.jev import MAX_QUESTIONS_PER_CALL
    qs = request_questions("v3")
    assert len(qs) > MAX_QUESTIONS_PER_CALL
    full = _answers()
    calls = []

    def fake(req, timeout=None):
        asked = json.loads(req.data)["questions"]
        calls.append(len(asked))
        assert len(asked) <= MAX_QUESTIONS_PER_CALL
        ans = {k: full.get(k, {"type": "noul", "noul": 0.1}) for k in asked}
        return _Resp(json.dumps({"answers": ans, "usage": {"input_tokens": 100, "cost": 1e-5}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = JevClient(api_key=FAKE_KEY)
    answers = client.decide("北京明天天气怎么样", qs)
    assert set(answers) == set(qs) and len(calls) == 2 and sum(calls) == len(qs)
    assert len(client.calls) == 1 and client.calls[0].input_tokens == 200
