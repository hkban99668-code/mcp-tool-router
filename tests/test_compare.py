"""离线对比框架 + Haiku 客户端的测试。

Haiku 的 mock 打在 `urllib.request.urlopen` 上,返回构造好的 chat completions JSON;
整个文件不许联网(autouse fixture 把没被单独 mock 的请求打成 AssertionError)。
"""

import json
import pathlib
import urllib.error
import urllib.request

import pytest
from sklearn.model_selection import StratifiedKFold

from mcpcls import cli, compare, evaluate as ev, realdata as rd
from mcpcls.codebuddy_client import CodeBuddyClient
from mcpcls.jev import (DEFAULT_JEV_MODEL, FakeJevClient, JevError, JevUnavailable, JevClient,
                        PaymentRequired, RateLimited, answers_to_labels, jev_fallback, make_client,
                        request_questions)
from mcpcls.llm_or import DEFAULT_CHAT_MODEL, OpenRouterChatClient, render_prompt
from mcpcls.router import Router
from mcpcls.strategy import STRATEGIES, TRIGGERS
from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
FAKE_KEY = "TEST-KEY-DO-NOT-USE"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("测试里不应发出任何网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch, tmp_path):
    """缓存一律落到本测试的 tmp 目录,免得测试污染仓库里的 .cache/answers。"""
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path)


@pytest.fixture(scope="module")
def data():
    return (ev.load_jsonl(DATA / "tools.jsonl"), ev.load_jsonl(DATA / "requests_train.jsonl"),
            ev.load_jsonl(DATA / "requests_gold.jsonl"))


# ---------------- mock 工具 ----------------

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


def _haiku_content(need="single", domains=("comms",), actions=("send",)):
    obj = {"need": need}
    for d in DOMAIN_IDS:
        obj[f"domain_{d}"] = d in domains
    for a in ACTION_IDS:
        obj[f"action_{a}"] = a in actions
    return json.dumps(obj, ensure_ascii=False)


def _chat(content, cost=3e-6, input_tokens=900):
    return {"choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"cost": cost, "prompt_tokens": input_tokens}}


def _fake_answers(need="single", domains=("comms",), actions=("send",)):
    a = {"need": {"type": "choice", "choice": need,
                  "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": 0.6}}
    for d in DOMAIN_IDS:
        a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
    return a


# ---------------- A5: Haiku 客户端 ----------------

def test_haiku_decide_returns_jev_shaped_answers(monkeypatch):
    seen = _mock_urlopen(monkeypatch, [_chat(_haiku_content("single", ["comms"], ["send"]))])
    client = OpenRouterChatClient(api_key=FAKE_KEY, api_format="openai")   # T-005 的 OpenAI 兼容口径
    answers = client.decide("给张三发封邮件", request_questions("v1"))

    assert answers["need"]["type"] == "choice" and answers["need"]["choice"] == "single"
    assert answers["need"]["probabilities"]["single"] == 1.0
    assert answers["need"]["probabilities"]["multi"] == 0.0
    assert answers["domain_comms"]["noul"] == 1.0 and answers["domain_file"]["noul"] == 0.0
    assert answers["action_send"]["noul"] == 1.0
    # 解析出来的答案能直接喂给现有的 answers_to_labels
    assert answers_to_labels(answers) == {"need": "single", "domains": ["comms"],
                                          "actions": ["send"], "confidence": 1.0}

    body = json.loads(seen[0].data.decode("utf-8"))
    assert body["model"] == DEFAULT_CHAT_MODEL
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert body["usage"] == {"include": True}
    assert FAKE_KEY not in json.dumps(body, ensure_ascii=False)
    prompt = body["messages"][0]["content"]
    assert prompt == render_prompt("给张三发封邮件", request_questions("v1"))
    for c in request_questions("v1").values():
        assert c["instructions"] in prompt
    assert len(client.calls) == 1 and client.calls[0].ok is True
    assert client.calls[0].cost == pytest.approx(3e-6) and client.calls[0].input_tokens == 900


def test_haiku_defaults_to_commandcode_messages(monkeypatch):
    """默认走 commandcode 的 Anthropic messages 接口,回复取 content[].text,缓存名和 T-005 的分开。"""
    for k in ("HAIKU_BASE_URL", "HAIKU_MODEL", "HAIKU_API_FORMAT"):
        monkeypatch.delenv(k, raising=False)
    reply = {"type": "message", "content": [{"type": "text",
             "text": _haiku_content("none", [], [])}], "usage": {"input_tokens": 812}}
    seen = _mock_urlopen(monkeypatch, [reply])
    client = OpenRouterChatClient(api_key=FAKE_KEY)
    answers = client.decide("解释一下什么是精确率", request_questions("v2"))
    assert answers_to_labels(answers)["need"] == "none"
    req = seen[0]
    assert req.full_url == "https://api.commandcode.ai/provider/v1/messages"
    assert req.get_header("Anthropic-version") == "2023-06-01"
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "claude-haiku-5-5" and "max_tokens" in body
    assert "response_format" not in body and "temperature" not in body
    assert client.calls[0].input_tokens == 812
    assert client.cache_name == "haiku@claude-haiku-5-5"
    assert OpenRouterChatClient(api_key=FAKE_KEY, model="anthropic/claude-haiku-5.5").cache_name == "haiku"


def test_haiku_bad_json_retries_then_raises(monkeypatch):
    seen = _mock_urlopen(monkeypatch, [_chat("这不是 JSON"), _chat("还不是 JSON")])
    client = OpenRouterChatClient(api_key=FAKE_KEY, retries=1)
    with pytest.raises(JevError):
        client.decide("给张三发封邮件", request_questions("v1"))
    assert len(seen) == 2                      # 重试了一次
    assert len(client.calls) == 1 and client.calls[0].ok is False
    assert "JSON" in client.calls[0].error


def test_haiku_missing_question_is_an_error(monkeypatch):
    body = _haiku_content()
    obj = json.loads(body)
    del obj["need"]
    _mock_urlopen(monkeypatch, [_chat(json.dumps(obj))])
    with pytest.raises(JevError):
        OpenRouterChatClient(api_key=FAKE_KEY, retries=0).decide("给张三发封邮件",
                                                                request_questions("v1"))


def test_haiku_never_leaks_the_key(monkeypatch):
    leaky = urllib.error.URLError(f"请求失败,Authorization: Bearer {FAKE_KEY}")
    _mock_urlopen(monkeypatch, [leaky])
    client = OpenRouterChatClient(api_key=FAKE_KEY, retries=0)
    with pytest.raises(JevError) as e:
        client.decide("给张三发封邮件", request_questions("v1"))
    assert FAKE_KEY not in str(e.value)
    assert FAKE_KEY not in client.calls[0].error


def test_haiku_requires_key_like_jev(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(JevUnavailable):
        OpenRouterChatClient()
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    assert OpenRouterChatClient().api_key == FAKE_KEY


def test_make_client_picks_the_right_source(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    jev_client = make_client("jev", api_key=FAKE_KEY)
    haiku_client = make_client("haiku", api_key=FAKE_KEY)
    assert isinstance(jev_client, JevClient) and isinstance(haiku_client, OpenRouterChatClient)
    assert (jev_client.source, jev_client.error_source) == ("jev_fallback", "jev_error")
    assert (haiku_client.source, haiku_client.error_source) == ("haiku_fallback", "haiku_error")
    # 兜底函数把这两个标记带上,Router 才能区分是谁改的结果 / 谁出的错
    assert jev_fallback(haiku_client).source == "haiku_fallback"
    assert jev_fallback(haiku_client).error_source == "haiku_error"
    with pytest.raises(ValueError):
        make_client("sonnet")


# ---------------- A6: 答案缓存 ----------------

def test_fetch_answers_caches_and_does_not_ask_twice(monkeypatch, tmp_path):
    items = [{"text": f"请求{i}"} for i in range(3)]
    client = FakeJevClient(lambda s, q: _fake_answers())
    first = compare.fetch_answers(client, "jev", items, "v1", "gold", live=True)
    assert len(client.calls) == 3 and all(r["ok"] for r in first)

    second = compare.fetch_answers(client, "jev", items, "v1", "gold", live=True)
    assert len(client.calls) == 3                                  # 第二次不再问
    assert [r["answers"] for r in second] == [r["answers"] for r in first]
    assert len((tmp_path / "jev-v1-gold.jsonl").read_text().strip().splitlines()) == 3


def test_fetch_answers_dedupes_repeated_texts(monkeypatch, tmp_path):
    items = [{"text": "同一条请求"}, {"text": "同一条请求"}, {"text": "另一条"}]
    client = FakeJevClient(lambda s, q: _fake_answers())
    out = compare.fetch_answers(client, "jev", items, "v1", "gold", live=True)
    assert len(client.calls) == 2
    assert out[0]["answers"] == out[1]["answers"]


def test_fetch_answers_offline_never_calls_the_client(monkeypatch, tmp_path):
    items = [{"text": "请求A"}, {"text": "请求B"}]
    client = FakeJevClient(lambda s, q: _fake_answers())
    out = compare.fetch_answers(None, "jev", items, "v1", "gold", live=False)
    assert [r["answers"] for r in out] == [None, None]
    assert all(r["ok"] is False for r in out)
    assert client.calls == []


def test_fetch_answers_offline_still_reads_the_cache(monkeypatch, tmp_path):
    items = [{"text": "请求A"}, {"text": "请求B"}]
    compare.fetch_answers(FakeJevClient(lambda s, q: _fake_answers()), "jev", items[:1], "v1",
                          "gold", live=True)
    out = compare.fetch_answers(None, "jev", items, "v1", "gold", live=False)
    assert out[0]["ok"] is True and out[0]["answers"]["need"]["choice"] == "single"
    assert out[1]["ok"] is False and out[1]["answers"] is None


# ---------------- 训练集折外预测 ----------------

def test_cv_splits_are_what_cross_validation_uses(data):
    train = data[1]
    labels = [r["need"] for r in train]
    want = [(list(a), list(b)) for a, b in
            StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(labels, labels)]
    assert [(list(a), list(b)) for a, b in ev.cv_splits(train)] == want


def test_local_outputs_on_train_matches_out_of_fold_predictions(data):
    """train 数据集用 5 折折外预测:这里按同一份分折独立重算一遍,逐条比对。"""
    tools, train, gold = data
    outs = compare.local_outputs("train", tools, train, gold)
    want = [None] * len(train)
    for tr_idx, te_idx in ev.cv_splits(train):
        fold = [train[i] for i in tr_idx]
        head = ev.train_request_head(ev.fit_encoder("tfidf", tools, fold), fold)
        for i, o in zip(te_idx, head.predict([train[i]["text"] for i in te_idx])):
            want[i] = o
    got = [(o.need, o.domains, o.actions) for o in outs]
    assert got == [(o.need, o.domains, o.actions) for o in want]


# ---------------- A7: 离线对比跑通 ----------------

def test_compare_offline_falls_back_to_local_everywhere(monkeypatch, tmp_path, data):
    res = compare.compare("gold", ["jev", "haiku"], ["v1", "v2"], live=False, tools=data[0],
                          train=data[1], gold=data[2])
    assert res["n"] == len(data[2]) and res["live"] is False
    assert len(res["rows"]) == 1 + len(res["clients"]) * len(res["need_versions"]) * len(STRATEGIES) * len(TRIGGERS)
    base = res["baseline"]
    assert 0.0 <= base["need_accuracy"] <= 1.0 and base["none_total"] > 0
    for row in res["rows"][1:]:
        assert row["called_rate"] == 0.0 and row["cost_usd"] == 0.0
        assert row["need_accuracy"] == base["need_accuracy"]      # 没有答案 → 全部退回本地
        assert (row["domain"], row["action"]) == (base["domain"], base["action"])
    md = compare.render_compare_report(res)
    assert "兜底组合策略对比" in md and "haiku" in md and "both_none" in md


def test_compare_live_uses_the_fake_client(monkeypatch, tmp_path, data):
    calls = []

    def make(name):
        calls.append(name)
        return FakeJevClient(lambda s, q: _fake_answers("none"), cost=2e-5)

    res = compare.compare("gold", ["jev"], ["v1"], live=True, tools=data[0], train=data[1],
                          gold=data[2], make=make)
    assert calls == ["jev"]
    rows = [r for r in res["rows"] if r["strategy"] != "local"]
    assert rows and all(r["called_rate"] > 0 for r in rows if r["trigger"] == "always")
    assert any(r["used_rate"] == 1.0 for r in rows)        # replace 策略每条都被改写
    # 全部说 none 时,both_none 的 need 准确率应当掉下来(本地的 single 被保住才对)
    always = {(r["strategy"], r["trigger"]): r for r in res["rows"] if r["client"] == "jev"}
    assert always[("both_none", "always")]["recall_needs_mcp"] >= always[("replace", "always")]["recall_needs_mcp"]


# ---------------- A4: Router 与离线对比逐条一致 ----------------

def test_router_and_compare_agree_item_by_item(monkeypatch, tmp_path, data):
    tools, train, gold = data
    # 用和 compare 完全相同的分类头(同一个 need 特征口径),否则比的是两个模型
    router = ev.build_system("tfidf", tools, train, need_feature_fn=compare.MAIN_NEED_FN)
    # 一半请求让云端说 none,另一半说 single:四种策略在这份答案上会走出不同分支
    none_texts = {g["text"] for g in gold[::2]}

    def answers_fn(state, questions):
        return _fake_answers("none" if state in none_texts else "single", ["comms"], ["send"])

    for strategy in ("replace", "need_only", "both_none"):
        client = FakeJevClient(answers_fn)
        r = Router(router.head, router.catalog, llm_fallback=jev_fallback(client),
                   fallback_strategy=strategy, fallback_trigger="always")
        routed = [(x.need, x.domains, x.actions) for x in (r.route(g["text"]) for g in gold)]

        res = compare.compare("gold", ["jev"], ["v1"], live=True, tools=tools, train=train,
                              gold=gold, make=lambda name: FakeJevClient(answers_fn),
                              with_preds=True)
        row = next(x for x in res["rows"]
                   if x["strategy"] == strategy and x["trigger"] == "always")
        got = [(p["need"], p["domains"], p["actions"]) for p in row["preds"]]
        assert got == routed, strategy


# ---------------- A7: CLI ----------------

def test_cli_compare_offline_writes_both_reports(monkeypatch, tmp_path, data):
    monkeypatch.setattr(cli, "REPORTS", tmp_path)
    code = cli.main(["compare", "--dataset", "gold", "--clients", "jev,haiku",
                     "--need-versions", "v1,v2", "--tools", str(DATA / "tools.jsonl"),
                     "--requests", str(DATA / "requests_train.jsonl"),
                     "--gold", str(DATA / "requests_gold.jsonl")])
    assert code == 0
    md = (tmp_path / "compare_gold.md").read_text(encoding="utf-8")
    assert "兜底组合策略对比" in md and "need_only" in md
    payload = json.loads((tmp_path / "compare_gold.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "gold" and len(payload["rows"]) == 33


def test_cli_compare_live_without_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert cli.main(["compare", "--dataset", "gold", "--live"]) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "Traceback" not in text
    assert len([l for l in text.splitlines() if l.strip()]) == 1
    assert "OPENROUTER_API_KEY" in text


def test_cli_compare_rejects_unknown_client_or_version(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert cli.main(["compare", "--clients", "jev,sonnet"]) == 2
    assert "不认识的参数值" in capsys.readouterr().err
    assert cli.main(["compare", "--need-versions", "v1,v9"]) == 2


# ---------------- A3/A4: 限流与欠费 ----------------

def test_jev_429_raises_rate_limited_no_retry(monkeypatch):
    """mock urlopen 返回 429 → RateLimited,不重试,立刻停。"""
    err = urllib.error.HTTPError(DEFAULT_JEV_MODEL, 429, "Too Many Requests", {}, None)
    seen = _mock_urlopen(monkeypatch, [err])
    client = JevClient(api_key=FAKE_KEY, retries=1)
    with pytest.raises(RateLimited):
        client.decide("x", request_questions())
    assert len(seen) == 1                       # 不重试


def test_haiku_402_raises_payment_required_no_retry(monkeypatch):
    """mock urlopen 返回 402 → PaymentRequired,不重试,立刻停。"""
    err = urllib.error.HTTPError(DEFAULT_JEV_MODEL, 402, "Payment Required", {}, None)
    seen = _mock_urlopen(monkeypatch, [err])
    client = OpenRouterChatClient(api_key=FAKE_KEY, retries=1)
    with pytest.raises(PaymentRequired):
        client.decide("x", request_questions("v1"))
    assert len(seen) == 1


def test_ask_many_stops_on_fatal_http_error(monkeypatch, tmp_path):
    """_ask_many 遇到 RateLimited 立刻停,stats.stopped 带上原因。"""
    err = urllib.error.HTTPError(DEFAULT_JEV_MODEL, 429, "Too Many Requests", {}, None)
    _mock_urlopen(monkeypatch, [err])
    client = JevClient(api_key=FAKE_KEY, retries=0)
    items = [{"text": f"请求{i}"} for i in range(5)]
    stats = compare.FetchStats()
    out = compare.fetch_answers(client, "jev", items, "v1", "gold", live=True, workers=1,
                               stats=stats)
    assert stats.stopped.startswith("限流")
    assert all(r["ok"] is False for r in out)


def test_dual_label_stops_with_exit_4_on_rate_limited(tmp_path):
    """realdata.dual_label 遇到限流抛 Stopped,exit_code=4。"""
    runner = lambda argv, timeout, env=None: ("", "429 您的使用量已超出频率限制,将在 10 秒后重置")
    client = CodeBuddyClient(runner=runner)
    pool = [{"id": "p-0", "text": "请求", "text_orig": "x", "source": "bfcl"}]
    with pytest.raises(rd.Stopped) as e:
        rd.dual_label(pool, [("codebuddy", client)], cost=rd.CostBook(tmp_path / "c.json"),
                      live=True)
    assert e.value.exit_code == 4
    assert "限流" in e.value.reason


def test_answer_keys_with_type_suffix_are_accepted():
    """Haiku 偶尔把 key 写成 "translation(text)"(照抄 prompt 小标题),要能对上题目。"""
    from mcpcls.llm_or import parse_answers_obj
    qs = {"translation": {"type": "text", "instructions": "翻译", "criteria": {}},
          "need": {"type": "choice", "instructions": "", "criteria": {"none": "", "single": ""}}}
    out = parse_answers_obj({"translation(text)": "你好", "need（choice）": "none"}, qs)
    assert out["translation"]["text"] == "你好" and out["need"]["choice"] == "none"
