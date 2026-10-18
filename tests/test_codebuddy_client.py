"""CodeBuddy 命令行客户端 + 批量问询路径的测试。

全程不起真的 `codebuddy` 进程:一律用注入的 `runner(argv, timeout, env=...)` 返回
构造好的 stdout。任务卡只允许新增这一个测试文件,所以 T-006 开放问题 2、4 的测试
(候选表预填、build_pool 的 math_like)也放在这里。
"""

import json
import pathlib
import sys
import urllib.request

import pytest

from mcpcls import compare, realdata as rd
from mcpcls.codebuddy_client import CodeBuddyClient, render_batch_prompt
from mcpcls.jev import (CLIENT_NAMES, FakeJevClient, JevError, JevUnavailable, RateLimited,
                        make_client, request_questions)
from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import build_pool                       # noqa: E402

FAKE_KEY = "TEST-KEY-DO-NOT-USE"
Q = request_questions("v2")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("测试里不应发出任何网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "answers")


# ---------------- 小工具 ----------------

def _answers(need="single", domains=("comms",), actions=("send",)):
    """一条 Jev 形状的 19 题答案。"""
    a = {"need": {"type": "choice", "choice": need,
                  "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": 0.6}}
    for d in DOMAIN_IDS:
        a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
    return a


def _items(pairs) -> str:
    """pairs: [(i, {题名: 答案}), ...] -> 模型该输出的那段 JSON 文本。"""
    return json.dumps({"items": [dict({"i": i}, **body) for i, body in pairs]},
                      ensure_ascii=False)


def _stdout(payload: str, is_error: str = "False") -> str:
    """`codebuddy --output-format json` 的输出:一个数组,最后一个是 result。"""
    return json.dumps([{"type": "message", "role": "assistant", "content": "…"},
                       {"type": "result", "is_error": is_error, "result": payload,
                        "total_cost_usd": "0", "duration_ms": "39000"}], ensure_ascii=False)


def _client(payloads, **kw):
    """payloads: 每次调用返回的 result 文本(或要抛的异常);runner 按顺序取。"""
    seen = []

    def runner(argv, timeout, env=None):
        seen.append({"argv": list(argv), "timeout": timeout, "env": env})
        item = payloads[len(seen) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    client = CodeBuddyClient(runner=runner, **kw)
    client.seen = seen
    return client


# ---------------- 命令行与子进程环境(A5) ----------------

def test_model_flag_is_only_added_when_given():
    c = _client([_stdout(_items([(1, _full_body())]))])
    c.decide("给张三发封邮件", Q)
    assert "--model" not in c.seen[0]["argv"]          # 没给 model 就不加

    c2 = _client([_stdout("{}")], model="hy4-preview")
    with pytest.raises(JevError):
        c2.decide("x", Q)
    assert c2.seen[0]["argv"][-2:] == ["--model", "hy4-preview"]


def test_argv_matches_the_card_exactly():
    """任务卡指定的参数顺序:`[binary, -p, prompt, --tools, "", --no-session-persistence,
    --output-format, json]`。"""
    c = _client([_stdout("{}")])
    with pytest.raises(JevError):
        c.decide("x", Q)
    argv = c.seen[0]["argv"]
    assert argv[0] == "codebuddy"
    assert argv[1] == "-p"
    assert argv[3:] == ["--tools", "", "--no-session-persistence", "--output-format", "json"]


def test_subprocess_env_has_no_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    c = _client([_stdout(_items([(1, _full_body())]))])
    c.decide("给张三发封邮件", Q)
    env = c.seen[0]["env"]
    assert env is not None and "OPENROUTER_API_KEY" not in env
    assert env["PATH"]                                 # 其他变量照常带过去


def test_runner_without_env_argument_still_works():
    """任务卡里的签名是 `runner(argv, timeout)`,两种都要能用。"""
    seen = []

    def runner(argv, timeout):
        seen.append(argv)
        return _stdout(_items([(1, _full_body())]))

    c = CodeBuddyClient(runner=runner)
    assert c.decide("x", Q)["need"]["choice"] == "single"
    assert len(seen) == 1


def test_missing_binary_raises_unavailable(monkeypatch):
    monkeypatch.setattr("mcpcls.codebuddy_client.shutil.which", lambda name: None)
    with pytest.raises(JevUnavailable):
        CodeBuddyClient()


# ---------------- 解析(A2) ----------------

def _full_body(need="single", domains=("comms",), actions=("send",)) -> dict:
    """19 题的扁平答案:choice 题给 id,noul 题给 true/false(模型实际就是这么输出的)。"""
    body = {"need": need}
    for d in DOMAIN_IDS:
        body[f"domain_{d}"] = d in domains
    for x in ACTION_IDS:
        body[f"action_{x}"] = x in actions
    return body


def test_parses_fenced_json_with_preamble():
    """前置说明文字 + ```json 代码围栏:照常解析。"""
    payload = ("I don't have access to a StructuredOutput tool, but here you go:\n"
               "```json\n" + _items([(1, _full_body("single", ["file"], ["create"]))]) + "\n```")
    c = _client([_stdout(payload)])
    ans = c.decide("帮我建个文件", Q)
    assert ans["need"]["choice"] == "single"
    assert ans["domain_file"]["noul"] == 1.0 and ans["domain_comms"]["noul"] == 0.0
    assert ans["action_create"]["noul"] == 1.0
    assert c.calls[0].ok is True


def test_items_out_of_order_are_aligned_by_i():
    """模型把顺序写反了:按 i 对回原顺序。"""
    pairs = [(3, _full_body("multi", ["file"], ["read"])),
             (1, _full_body("single", ["comms"], ["send"])),
             (2, _full_body("none"))]
    c = _client([_stdout(_items(pairs))])
    got = c.decide_batch(["a", "b", "c"], Q)
    assert [g["need"]["choice"] for g in got] == ["single", "none", "multi"]
    assert got[2]["domain_file"]["noul"] == 1.0


def test_missing_item_becomes_none():
    """少一条:那个位置是 None,其余照常。"""
    c = _client([_stdout(_items([(1, _full_body()), (3, _full_body("none"))]))])
    got = c.decide_batch(["a", "b", "c"], Q)
    assert got[0] is not None and got[1] is None and got[2] is not None
    assert [call.ok for call in c.calls] == [True, False, True]
    assert "没解析出答案" in c.calls[1].error


def test_invalid_choice_becomes_none():
    """某条的选项 id 非法:那个位置是 None。"""
    bad = _full_body()
    bad["need"] = "banana"                              # 不在 none/single/multi 里
    c = _client([_stdout(_items([(1, _full_body()), (2, bad)]))])
    got = c.decide_batch(["a", "b"], Q)
    assert got[0] is not None and got[1] is None
    assert [call.ok for call in c.calls] == [True, False]


def test_no_json_at_all_is_a_batch_failure():
    c = _client([_stdout("抱歉,我做不到")], retries=0)
    with pytest.raises(JevError):
        c.decide_batch(["a", "b"], Q)
    assert len(c.calls) == 2 and all(call.ok is False for call in c.calls)


# ---------------- 整批失败与重试(A3) ----------------

def test_batch_retries_and_succeeds_on_the_second_try():
    """第一次 is_error,第二次正常:整批重试后成功,calls 里只有这一批(按条)。"""
    c = _client([_stdout("出错了", is_error="True"),
                 _stdout(_items([(1, _full_body()), (2, _full_body("none"))]))])
    got = c.decide_batch(["a", "b"], Q)
    assert len(c.seen) == 2                             # 真的重试了一次
    assert len(c.calls) == 2 and all(call.ok for call in c.calls)
    assert [g["need"]["choice"] for g in got] == ["single", "none"]


def test_batch_fails_twice_raises_and_marks_every_item():
    c = _client([_stdout("出错了", is_error="True"), _stdout("又错了", is_error="True")])
    with pytest.raises(JevError):
        c.decide_batch(["a", "b", "c"], Q)
    assert len(c.calls) == 3                            # 本批每条一条记录
    assert all(call.ok is False for call in c.calls)
    assert all("is_error" in call.error for call in c.calls)
    assert all(call.latency_ms > 0 for call in c.calls)


def test_decide_is_a_batch_of_one():
    payload = _stdout(_items([(1, _full_body("multi", ["file"], ["read"]))]))
    c = _client([payload])
    assert c.decide("x", Q)["need"]["choice"] == "multi"
    assert len(c.calls) == 1
    assert c.decide_batch([], Q) == []


def test_prompt_lists_numbered_requests_once():
    prompt = render_batch_prompt(["第一条", "第二条"], Q)
    assert "1. 第一条" in prompt and "2. 第二条" in prompt
    assert prompt.count("## need(") == 1                # 题目说明只渲染一次
    assert "items" in prompt                            # 要求输出 items
    assert prompt.rstrip().endswith("说明。")            # "只输出 JSON"放在最后


# ---------------- 批量问询路径(A4) ----------------

class FakeBatchClient:
    """有 decide_batch 的假客户端:记录每次收到的一批文本,none_at 里的位置返回 None。"""

    def __init__(self, none_at=(), fail=False, batch_size=20):
        self.none_at, self.fail, self.batch_size = set(none_at), fail, batch_size
        self.batches: list = []

    def decide_batch(self, states, questions):
        self.batches.append(list(states))
        if self.fail:
            raise JevError("fake: 整批失败")
        return [None if i in self.none_at else _answers()
                for i in range(1, len(states) + 1)]


def test_ask_many_slices_45_texts_into_three_batches(tmp_path):
    items = [{"text": f"请求{i}"} for i in range(45)]
    client = FakeBatchClient()
    out = compare.fetch_answers(client, "codebuddy", items, "v2", "pool", live=True, workers=4)
    # 恰好 3 次 decide_batch;并发下完成顺序不定,所以按块大小排序后比
    assert sorted(len(b) for b in client.batches) == [5, 20, 20]
    assert len(client.batches) == 3
    assert sum(len(b) for b in client.batches) == 45
    assert all(r["ok"] for r in out)
    lines = [l for l in compare.cache_path("codebuddy", "v2", "pool")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [json.loads(l)["text"] for l in lines]
    assert len(texts) == 45 and len(set(texts)) == 45           # 45 行无重复


def test_ask_many_counts_none_items_as_failures(tmp_path):
    """块内 None:不进缓存、计入失败统计。"""
    items = [{"text": f"请求{i}"} for i in range(6)]
    client = FakeBatchClient(none_at={2, 3}, batch_size=6)
    stats = compare.FetchStats()
    out = compare.fetch_answers(client, "codebuddy", items, "v2", "pool", live=True, workers=1,
                               stats=stats)
    assert [r["ok"] for r in out] == [True, False, False, True, True, True]
    assert stats.asked == 6 and stats.failed == 2
    path = compare.cache_path("codebuddy", "v2", "pool")
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4                                      # 失败的两条没写缓存


def test_ask_many_without_decide_batch_is_unchanged():
    """没有 decide_batch 的客户端(jev / haiku):一条一次调用,行为不变。"""
    items = [{"text": f"请求{i}"} for i in range(7)]
    client = FakeJevClient(lambda s, q: _answers())
    out = compare.fetch_answers(client, "jev", items, "v2", "pool", live=True, workers=2)
    assert len(client.calls) == 7
    assert all(r["ok"] for r in out)


def test_whole_batch_failure_keeps_semantics(tmp_path):
    """整批抛错:这一批每条都算失败(和连续失败计数用同一套口径)。"""
    items = [{"text": f"请求{i}"} for i in range(4)]
    client = FakeBatchClient(fail=True, batch_size=4)
    stats = compare.FetchStats()
    out = compare.fetch_answers(client, "codebuddy", items, "v2", "pool", live=True, workers=1,
                               stats=stats)
    assert all(r["ok"] is False for r in out)
    assert stats.failed == 4 and "整批失败" in stats.first_error


class FakeBatchClientWithPattern:
    """按批返回成功/失败,用于验证连续失败计数按批计。"""

    def __init__(self, batch_size=2, pattern=()):
        self.batch_size = batch_size
        self.pattern = list(pattern)          # True 表示该批成功,False 表示整批失败
        self.batches: list = []

    def decide_batch(self, states, questions):
        self.batches.append(list(states))
        ok = self.pattern[min(len(self.batches) - 1, len(self.pattern) - 1)]
        if not ok:
            raise JevError("fake: 整批失败")
        return [_answers() for _ in states]


def test_batch_consecutive_failures_count_by_batch(tmp_path):
    """A2:第 1 批失败,第 2 批成功 → 不停;连续 3 批失败 → 停,且批内部分失败不累加。"""
    # 10 条,batch_size=2 -> 5 批;模式:败、成、败、败、败(第 3~5 批连续 3 批停)
    items = [{"text": f"请求{i}"} for i in range(10)]
    client = FakeBatchClientWithPattern(batch_size=2,
                                        pattern=[False, True, False, False, False])
    stats = compare.FetchStats()
    out = compare.fetch_answers(client, "codebuddy", items, "v2", "pool", live=True, workers=1,
                               stats=stats)
    assert len(client.batches) == 5           # 5 批全部发出后才停
    assert stats.stopped.startswith("连续 3 批")
    assert sum(r["ok"] for r in out) == 2   # 只有第 2 批的 2 条成功


def test_rate_limited_stops_immediately_no_retry(tmp_path):
    """A3:runner 返回空 stdout + 429 stderr → RateLimited,不重试,_ask_many 立刻停。"""
    def runner(argv, timeout, env=None):
        return ("", "429 您的使用量已超出频率限制,将在 10 秒后重置")

    client = CodeBuddyClient(runner=runner, retries=1)
    with pytest.raises(RateLimited):
        client.decide_batch(["a"], Q)
    assert len(client.calls) == 1 and client.calls[0].ok is False
    assert "429" in client.calls[0].error

    # _ask_many 看到 RateLimited 立刻停,不等连续 3 批
    items = [{"text": f"请求{i}"} for i in range(6)]
    stats = compare.FetchStats()
    out = compare.fetch_answers(client, "codebuddy", items, "v2", "pool", live=True, workers=1,
                               stats=stats)
    assert stats.stopped.startswith("限流")
    assert all(r["ok"] is False for r in out)


def test_default_and_named_model_cache_names(tmp_path):
    """A5:默认模型缓存名为 codebuddy;指定模型为 codebuddy@<model>,文件名 @ 变 _at_。"""
    assert compare.cache_path("codebuddy", "v2", "pool").name == "codebuddy-v2-pool.jsonl"
    assert compare.cache_path("codebuddy@glm-5.3", "v2", "pool").name \
        == "codebuddy_at_glm-5.3-v2-pool.jsonl"

    c1 = CodeBuddyClient(runner=lambda a, t, e=None: _stdout(_items([(1, _full_body())])))
    assert c1.cache_name == "codebuddy"
    c2 = CodeBuddyClient(model="glm-5.3",
                         runner=lambda a, t, e=None: _stdout(_items([(1, _full_body())])))
    assert c2.cache_name == "codebuddy@glm-5.3"


# ---------------- 客户端工厂与来源标记 ----------------

def test_make_client_builds_codebuddy(monkeypatch):
    assert "codebuddy" in CLIENT_NAMES
    runner = lambda argv, timeout, env=None: _stdout("{}")      # noqa: E731
    client = make_client("codebuddy", runner=runner)
    assert isinstance(client, CodeBuddyClient)
    assert (client.source, client.error_source) == ("codebuddy_fallback", "codebuddy_error")
    with pytest.raises(ValueError):
        make_client("sonnet")


def test_codebuddy_fallback_source_is_known_to_the_router():
    """兜底改写过结果时 Router 不再用模型自己的概率(见 router.FALLBACK_SOURCES)。"""
    from mcpcls.router import FALLBACK_SOURCES

    assert "codebuddy_fallback" in FALLBACK_SOURCES


def test_realdata_raises_the_timeout_for_batch_clients(monkeypatch):
    """一批要几十秒,realdata 那条 30 秒的默认超时会把每批都打成超时。"""
    monkeypatch.setattr("mcpcls.codebuddy_client.shutil.which", lambda name: "/usr/bin/codebuddy")
    opened = []

    def fake_make(name, **kw):
        opened.append((name, kw))
        return CodeBuddyClient(runner=lambda argv, timeout, env=None: _stdout("{}"), **kw)

    monkeypatch.setattr(rd, "make_client", fake_make)
    rd.open_client("codebuddy", timeout=30.0)
    assert opened[0][1]["timeout"] == CodeBuddyClient.min_timeout >= 120.0


# ---------------- T-006 开放问题 4:math_like ----------------

def test_math_like_flag_on_function_names_and_descriptions():
    assert build_pool.math_like([{"name": "calculate_triangle_area",
                                  "description": "Calculate the area of a triangle"}])
    assert build_pool.math_like([{"name": "solve_equation", "description": "Solve it"}])
    assert build_pool.math_like([{"name": "gcd", "description": "greatest common divisor"}])
    assert not build_pool.math_like([{"name": "get_weather", "description": "Get the weather"}])
    assert not build_pool.math_like([])


def _tiny_raw(tmp_path) -> pathlib.Path:
    """一份很小的 raw:两条 BFCL(一条算学、一条不是)+ 一条 alpaca。"""
    raw = tmp_path / "raw"
    (raw / "bfcl").mkdir(parents=True)
    (raw / "apibank").mkdir(parents=True)
    (raw / "alpaca_zh").mkdir(parents=True)
    rows = [
        {"question": [[{"role": "user", "content": "Find the area of a triangle"}]],
         "function": [{"name": "calculate_triangle_area", "description": "Calculate the area"}]},
        {"question": [[{"role": "user", "content": "Send an email to Bob"}]],
         "function": [{"name": "send_email", "description": "Send an email"}]},
    ]
    (raw / "bfcl" / "BFCL_v3_simple.json").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (raw / "apibank" / "lv1-train.json").write_text(json.dumps([]), encoding="utf-8")
    (raw / "alpaca_zh" / "alpaca_gpt4_data_zh.json").write_text(
        json.dumps([{"instruction": "写一段介绍", "input": "", "output": "x"}],
                   ensure_ascii=False), encoding="utf-8")
    return raw


def test_math_like_hint_does_not_move_or_drop_rows(tmp_path, monkeypatch):
    """A6:加了 math_like 以后,池子的文本与顺序和改动前逐条一致。"""
    raw = _tiny_raw(tmp_path)
    pool, _ = build_pool.build(raw)
    assert len(pool) == 3
    assert [r["hint"]["math_like"] for r in pool if r["source"] == "bfcl"] == [True, False]
    assert all("math_like" not in r["hint"] for r in pool if r["source"] != "bfcl")

    # 把 math_like 恒定为 False 再跑一次:除了 hint 里的这个字段,其余必须逐条相同
    monkeypatch.setattr(build_pool, "math_like", lambda funcs: False)
    baseline, _ = build_pool.build(raw)
    assert [r["id"] for r in baseline] == [r["id"] for r in pool]
    assert [r["text_orig"] for r in baseline] == [r["text_orig"] for r in pool]
    assert [r["source"] for r in baseline] == [r["source"] for r in pool]
    assert [r["lang"] for r in baseline] == [r["lang"] for r in pool]
    for a, b in zip(baseline, pool):
        assert {k: v for k, v in a["hint"].items() if k != "math_like"} == \
               {k: v for k, v in b["hint"].items() if k != "math_like"}


def test_math_like_splits_the_labeling_stats(tmp_path):
    """打标统计按 math_like 分开报:一致率与 need 分布。"""
    pool = [{"id": "m-0", "text": "算一条", "text_orig": "x", "source": "bfcl",
             "hint": {"math_like": True}},
            {"id": "o-0", "text": "普通一条", "text_orig": "y", "source": "bfcl",
             "hint": {"math_like": False}}]
    a = FakeJevClient(lambda s, q: _answers("single", ["comms"], ["send"]))
    b = FakeJevClient(lambda s, q: _answers("single", ["comms"], ["send"]))
    out = rd.dual_label(pool, [("jev", a), ("haiku", b)], cost=rd.CostBook(tmp_path / "c.json"),
                        live=True)
    split = out["by_math_like"]
    assert split["math_like"] == {"n": 1, "agreed": 1, "review": 0, "agree_rate": 1.0,
                                  "need": {"single": 1}}
    assert split["other"]["n"] == 1 and split["other"]["agree_rate"] == 1.0
    assert rd.is_math_like(pool[0]) is True and rd.is_math_like({"text": "x"}) is False


def test_holdout_row_reports_math_like_split(tmp_path, monkeypatch):
    """evaluate-real 的留出集那行带上按 math_like 拆分的准确率。"""
    holdout = [{"id": f"h-{i}", "text": f"留出请求 {i}", "need": "single", "domains": ["comms"],
                "actions": ["send"], "source": "bfcl",
                "hint": {"math_like": i < 2}} for i in range(4)]

    class Head:
        def predict(self, texts):
            from mcpcls.heads import HeadOutput

            return [HeadOutput(need="single", domains=["comms"], actions=["send"],
                               confidence=0.9, domain_probs={}, action_probs={})
                    for _ in texts]

    row = rd._holdout_row(type("R", (), {"head": Head()}), holdout)
    assert set(row["by_math_like"]) == {"math_like", "other"}
    assert row["by_math_like"]["math_like"] == {"n": 2, "accuracy": 1.0, "need": {"single": 2}}
    assert row["by_math_like"]["other"]["n"] == 2


# ---------------- T-006 开放问题 2:候选表预填 ----------------

def test_review_row_prefills_the_primary_actions():
    """两路不一致的行:actions 预填第一路(基准路)的。"""
    review = [{"id": "rev-0", "text": "待人工请求", "text_orig": "x", "source": "bfcl",
               "labelers": ["jev", "haiku"],
               "jev": {"need": "single", "domains": ["comms"], "actions": ["send", "read"],
                       "ok": True},
               "haiku": {"need": "multi", "domains": ["file"], "actions": ["read"], "ok": True}}]
    rows = rd.candidates(review, [], lambda texts: [("single", ["comms"], ["read"])] * len(texts),
                         n=1)
    assert rows[0]["actions"] == "send;read"            # 第一路的,不是第二路的
    assert rows[0]["domains"] == "comms"                # domains 也还是第一路的
    assert rows[0]["need"] == "single"                  # need 预填本地模型
    assert rows[0]["note"] == "两路不一致"
