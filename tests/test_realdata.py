"""真实数据流水线(T-006)的测试。全程不联网:客户端一律用 Fake,缓存落到 tmp 目录。

autouse 的 no_network 把没被单独 mock 的网络请求打成 AssertionError,
所以这个文件里任何一次真实的 urlopen 都会让测试失败。
"""

import csv
import json
import pathlib
import random
import sys
import urllib.request

import pytest

from mcpcls import cli, compare, evaluate as ev, realdata as rd
from mcpcls.jev import FakeJevClient
from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))          # 让测试能 import scripts/ 下的取样脚本
from scripts.build_pool import build, is_english      # noqa: E402

FAKE_KEY = "TEST-KEY-DO-NOT-USE"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("测试里不应发出任何网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "answers")


@pytest.fixture(scope="module")
def data():
    return (ev.load_jsonl(DATA / "tools.jsonl"), ev.load_jsonl(DATA / "requests_train.jsonl"),
            ev.load_jsonl(DATA / "requests_gold.jsonl"))


# ---------------- 小工具 ----------------

def _pool_row(i, text, lang="en", source="bfcl"):
    return {"id": f"{source}-{i}", "text_orig": text, "lang": lang, "source": source,
            "source_file": f"{source}.json", "hint": {}}


def _answers(need="single", domains=("comms",), actions=("send",)):
    a = {"need": {"type": "choice", "choice": need,
                  "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": 0.6}}
    for d in DOMAIN_IDS:
        a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
    return a


def _translation(text):
    return {"translation": {"type": "text", "text": text}}


def _translator(prefix="帮我") -> FakeJevClient:
    """翻译用的假客户端:把英文原文前面加个中文前缀,便于断言"确实翻了"。"""
    return FakeJevClient(lambda state, q: _translation(f"{prefix}{state}"))


def _fake_real(n, seed=0, offset=0):
    """假的真实数据:三种 source、三种 need 都有,够了分层切分的最低要求。"""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        need = ("none", "single", "multi")[i % 3]
        tail = ("解释一下这个概念", "查一下订单状态", "先查日程再发邮件")[i % 3]
        out.append({"id": f"fake-{offset + i}", "text": f"真实请求 {offset + i} 号,{tail}",
                    "text_orig": f"real request {offset + i}", "source":
                    ("bfcl", "apibank", "alpaca_zh")[i % 3], "need": need,
                    "domains": [] if need == "none" else [("file", "database", "comms")[i % 3]],
                    "actions": [] if need == "none" else [("read", "create", "update")[i % 3]],
                    "label_source": "jev+haiku"})
    rnd.shuffle(out)
    return out


# ---------------- A2: build_pool 可复现 ----------------

def test_build_pool_is_byte_identical_on_the_same_raw(tmp_path):
    raw = tmp_path / "raw"
    (raw / "bfcl").mkdir(parents=True)
    (raw / "apibank").mkdir(parents=True)
    (raw / "alpaca_zh").mkdir(parents=True)
    (raw / "bfcl" / "BFCL_v3_simple.json").write_text(
        "\n".join(json.dumps({"question": [[{"role": "user", "content": f"Find item {i}"}]],
                              "function": [{"name": "f"}]}) for i in range(60)),
        encoding="utf-8")
    (raw / "apibank" / "lv1-train.json").write_text(
        json.dumps([{"input": '{"apiCode": "x"}\nUser: 帮我查一下订单 %d\nGenerate API Request: ' % i}
                    for i in range(40)], ensure_ascii=False), encoding="utf-8")
    (raw / "alpaca_zh" / "alpaca_gpt4_data_zh.json").write_text(
        json.dumps([{"instruction": f"写一段关于 {i} 的介绍", "input": "", "output": "x"}
                    for i in range(50)], ensure_ascii=False), encoding="utf-8")

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    build(raw)                       # 两次都在同一份 raw 上跑
    pool_a, _ = build(raw)
    with open(a, "w", encoding="utf-8") as f:
        for r in pool_a:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool_b, _ = build(raw)
    with open(b, "w", encoding="utf-8") as f:
        for r in pool_b:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert a.read_bytes() == b.read_bytes()
    assert len(pool_a) == len(pool_b) > 0
    ids = [r["id"] for r in pool_a]
    assert len(set(ids)) == len(ids)          # id 不重复
    for r in pool_a:
        assert set(r) == {"id", "text_orig", "lang", "source", "source_file", "hint"}


def test_is_english_rule():
    assert is_english("Can you retrieve the details for the user with the ID 7890?")
    assert not is_english("帮我查一下上海明天的天气")
    # 中文句子里夹英文地名,整句仍算中文
    assert not is_english("从 同仁医院 到 维多利亚酒店 怎么走,顺便查一下 weather")


# ---------------- A3: 翻译与打标都走缓存 ----------------

def test_translate_only_asks_english_rows_and_caches(tmp_path):
    pool = [_pool_row(0, "Check the status of my claim", source="bfcl"),
            _pool_row(1, "帮我查一下订单", lang="zh", source="alpaca_zh")]
    client = _translator()
    cost = rd.CostBook(tmp_path / "cost.json")
    out = rd.translate(pool, client, "haiku", cost=cost, live=True)
    assert len(client.calls) == 1                       # 中文那条没问
    assert out["rows"][0]["text"] == "帮我Check the status of my claim"
    assert out["rows"][1]["text"] == "帮我查一下订单"      # 中文原样保留

    second = rd.translate(pool, _translator(), "haiku", cost=cost, live=True)
    assert len(client.calls) == 1                       # 第二次读缓存,不再问
    assert second["rows"][0]["text"] == out["rows"][0]["text"]


def test_dual_label_caches_both_labelers(tmp_path):
    pool = [{"id": f"p-{i}", "text": f"请求{i}", "text_orig": f"request {i}",
             "source": "bfcl"} for i in range(4)]
    a = FakeJevClient(lambda s, q: _answers("single", ["comms"], ["send"]))
    b = FakeJevClient(lambda s, q: _answers("single", ["comms"], ["send", "read"]))
    cost = rd.CostBook(tmp_path / "cost.json")
    first = rd.dual_label(pool, [("jev", a), ("haiku", b)], cost=cost, live=True)
    assert len(a.calls) == 4 and len(b.calls) == 4
    assert len(first["agreed"]) == 4 and first["review"] == []

    a2 = FakeJevClient(lambda s, q: _answers())
    b2 = FakeJevClient(lambda s, q: _answers())
    second = rd.dual_label(pool, [("jev", a2), ("haiku", b2)], cost=cost, live=True)
    assert a2.calls == [] and b2.calls == []            # A3:第二次调用次数为 0
    assert second["agreed"] == first["agreed"]


def test_translate_keeps_original_when_it_fails(tmp_path, capsys):
    """翻译失败不丢数据:保留原文继续走打标,条数报出来。"""
    pool = [_pool_row(0, "Check the status of my claim")]
    client = FakeJevClient(lambda s, q: _answers(), fail=True)
    out = rd.translate(pool, client, "haiku", cost=rd.CostBook(tmp_path / "c.json"), live=True)
    assert out["rows"][0]["text"] == "Check the status of my claim"
    assert out["untranslated"] == 1
    assert "失败 1 条" in capsys.readouterr().out       # T-005 第 7 节:失败要打印


# ---------------- A4: 预算 ----------------

def test_budget_stops_before_the_next_call(tmp_path):
    pool = [_pool_row(i, f"Check item {i}") for i in range(2)]
    cost = rd.CostBook(tmp_path / "cost.json")
    out = rd.translate(pool, FakeJevClient(lambda s, q: _translation("译"), cost=0.6),
                       "haiku", budget=1.0, cost=cost, live=True)
    assert cost.total() == pytest.approx(1.2)           # 第一次跑完,结果已落盘
    rd.write_jsonl(tmp_path / "pool_zh.jsonl", out["rows"])
    assert len(rd.load_jsonl(tmp_path / "pool_zh.jsonl")) == 2

    with pytest.raises(rd.Stopped) as e:                # 第二次调用前停下
        rd.translate(pool, FakeJevClient(lambda s, q: _translation("译"), cost=0.6),
                     "haiku", budget=1.0, cost=cost, live=True)
    assert e.value.exit_code == 3
    assert "预算" in e.value.reason


def test_client_cannot_be_built_stops_with_exit_4(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(rd.Stopped) as e:
        rd.open_client("haiku")
    assert e.value.exit_code == 4 and "haiku" in e.value.reason


def test_consecutive_failures_stop_early(tmp_path, capsys):
    """连续 5 条都失败(比如 HTTP 402)立刻停,退出码 4,不剩下的继续烧钱。"""
    pool = [_pool_row(i, f"Check item {i}") for i in range(20)]
    client = FakeJevClient(lambda s, q: _answers(), fail=True)
    with pytest.raises(rd.Stopped) as e:
        rd.dual_label([dict(r, text=r["text_orig"]) for r in pool],
                      [("jev", client), ("haiku", FakeJevClient(lambda s, q: _answers()))],
                      workers=1, cost=rd.CostBook(tmp_path / "c.json"), live=True)
    assert e.value.exit_code == 4
    assert len(client.calls) < 20                       # 没把 20 条全问完
    assert "失败" in capsys.readouterr().out


# ---------------- A5: 并发 ----------------

def test_concurrent_cache_has_no_lost_or_duplicated_lines(tmp_path):
    items = [{"text": f"请求{i}"} for i in range(50)]
    client = FakeJevClient(lambda s, q: _answers())
    compare.fetch_answers(client, "jev", items, "v2", "pool", live=True, workers=4)
    assert len(client.calls) == 50
    path = compare.cache_path("jev", "v2", "pool")
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 50
    texts = [json.loads(l)["text"] for l in lines]
    assert len(set(texts)) == 50                        # 并发写缓存不丢行、不重复


# ---------------- A6: 一致判定与标签合成 ----------------

def _lab(need, domains=(), actions=()):
    return {"need": need, "domains": list(domains), "actions": list(actions), "ok": True}


def test_agreement_rules():
    assert rd.agree(_lab("none"), _lab("none")) is True                  # none 一致
    assert rd.agree(_lab("single", ["comms"]), _lab("single", ["comms"])) is True
    # need 不同 -> 不一致
    assert rd.agree(_lab("single", ["comms"]), _lab("multi", ["comms"])) is False
    # need 相同但域不同 -> 不一致
    assert rd.agree(_lab("single", ["comms"]), _lab("single", ["file"])) is False
    # need 非 none 时域相同、顺序不同也算一致
    assert rd.agree(_lab("single", ["comms", "file"]), _lab("single", ["file", "comms"])) is True


def test_merge_actions_takes_intersection_then_jev():
    a = _lab("single", ["comms"], ["send", "read"])
    b = _lab("single", ["comms"], ["read", "update"])
    assert rd.merge_actions(a, b) == ["read"]           # 交集
    c = _lab("single", ["comms"], ["send"])
    d = _lab("single", ["comms"], ["delete"])
    assert rd.merge_actions(c, d) == ["send"]           # 交集为空取第一路(Jev)的


def test_dual_label_splits_agree_and_review(tmp_path):
    pool = [{"id": "a", "text": "给张三发封邮件", "text_orig": "x", "source": "bfcl"},
            {"id": "b", "text": "帮我查订单", "text_orig": "y", "source": "apibank"}]

    def ans(state, q):
        return _answers("single", ["comms"], ["send"]) if state == "给张三发封邮件" \
            else _answers("none")

    jev = FakeJevClient(ans)
    haiku = FakeJevClient(lambda s, q: _answers("single", ["comms"], ["send"]))
    out = rd.dual_label(pool, [("jev", jev), ("haiku", haiku)],
                        cost=rd.CostBook(tmp_path / "c.json"), live=True)
    assert [r["id"] for r in out["agreed"]] == ["a"]
    assert out["agreed"][0]["label_source"] == "jev+haiku"
    assert [r["id"] for r in out["review"]] == ["b"]
    # review 行带着两路各自的结果,便于人工看
    assert out["review"][0]["jev"]["need"] == "none"
    assert out["review"][0]["haiku"]["need"] == "single"


# ---------------- 切分 ----------------

def test_split_is_stratified_and_disjoint():
    agreed = _fake_real(60)
    train, hold = rd.split(agreed)
    assert len(train) + len(hold) == 60
    assert len(hold) == pytest.approx(12, abs=1)        # 20%
    assert {r["id"] for r in train} & {r["id"] for r in hold} == set()
    for need in ("none", "single", "multi"):
        assert sum(1 for r in hold if r["need"] == need) > 0   # 每一类都留了样本


# ---------------- A7: 金标候选表 ----------------

def _review_rows(n, offset=0):
    out = []
    for i in range(n):
        need = ("none", "single", "multi")[i % 3]
        out.append({"id": f"rev-{offset + i}", "text": f"待人工请求 {offset + i}",
                    "text_orig": f"review {offset + i}", "source": "bfcl",
                    "labelers": ["jev", "haiku"],
                    "jev": {"need": need, "domains": ["comms"], "actions": ["send"], "ok": True},
                    "haiku": {"need": "single", "domains": ["file"], "actions": ["read"],
                              "ok": True}})
    return out


def test_gold_candidates_csv_is_excel_readable(tmp_path):
    review = _review_rows(120)
    holdout = _fake_real(120)
    rows = rd.candidates(review, holdout, lambda texts: [("single", ["comms"], ["read"])] * len(texts))
    path = tmp_path / "gold_candidates.csv"
    rd.write_gold_candidates(path, rows)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")              # 带 BOM,Excel 直接开
    with open(path, encoding="utf-8-sig", newline="") as f:
        parsed = list(csv.reader(f))
    assert parsed[0] == list(rd.CAND_COLUMNS)           # 列顺序与任务卡一致
    assert len(parsed) - 1 == 200


def test_candidates_take_half_from_each_side():
    review, holdout = _review_rows(120), _fake_real(120)
    rows = rd.candidates(review, holdout, lambda t: [("single", ["comms"], ["read"])] * len(t))
    assert len(rows) == 200
    rev_ids = {r["id"] for r in review}
    assert sum(1 for r in rows if r["id"] in rev_ids) == 100
    # review 行:need 预填本地模型的,domains 填 Jev 的,note 写"两路不一致"
    rev_row = next(r for r in rows if r["id"] in rev_ids)
    assert rev_row["need"] == "single" and rev_row["domains"] == "comms"
    assert rev_row["note"] == "两路不一致"
    hold_row = next(r for r in rows if r["id"] not in rev_ids)
    assert hold_row["note"] == "" and hold_row["need"] == hold_row["labeler1_need"]


def test_load_gold_candidates_reports_the_line_number(tmp_path):
    review, holdout = _review_rows(4), _fake_real(4)
    rows = rd.candidates(review, holdout, lambda t: [("single", ["comms"], ["read"])] * len(t),
                         n=8)
    path = tmp_path / "c.csv"
    rd.write_gold_candidates(path, rows)
    good = rd.load_gold_candidates(path)
    assert len(good) == 8 and all(set(r) >= {"text", "need", "domains", "actions"} for r in good)

    # 把第 3 行(第一条数据在第 2 行)的 domains 改成一个不存在的 id
    lines = path.read_text(encoding="utf-8").splitlines()
    domains_idx = lines[0].split(",").index("domains")
    fields = lines[2].split(",")
    fields[domains_idx] = "not_a_domain"
    lines[2] = ",".join(fields)
    (tmp_path / "bad.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    with pytest.raises(ValueError) as e:
        rd.load_gold_candidates(tmp_path / "bad.csv")
    assert "第 3 行" in str(e.value) and "not_a_domain" in str(e.value)


def test_candidates_readme_lists_every_column():
    text = rd.render_candidates_readme()
    for col in rd.CAND_COLUMNS:
        assert col in text
    assert "能力域 domains" in text      # 类别定义卡来自 taxonomy


# ---------------- A8: evaluate-real ----------------

def test_evaluate_real_covers_three_trainsets_and_two_evalsets(data):
    tools, synth, gold = data
    res = rd.evaluate_real(tools, synth, gold, _fake_real(30), _fake_real(12, offset=100))
    variants = {r["variant"] for r in res["rows"]}
    assert variants == {"A 只用合成训练集", "B 只用 real_train", "C 合成 + real_train"}
    eval_sets = {r["eval_set"] for r in res["rows"]}
    assert eval_sets == {"合成 gold", "银标留出集"}
    for r in res["rows"]:
        assert 0.0 <= r["need_accuracy"] <= 1.0
        assert r["domain"]["micro_f1"] is not None and r["action"]["macro_f1"] is not None
    hold_rows = [r for r in res["rows"] if r["eval_set"] == "银标留出集"]
    assert all(set(r["by_source"]) == {"bfcl", "apibank", "alpaca_zh"} for r in hold_rows)
    gold_rows = [r for r in res["rows"] if r["eval_set"] == "合成 gold"]
    assert all(set(r["recall"]) == {1, 3, 5} for r in gold_rows)     # Recall@k 只在 gold 上
    assert all(r["recall"] is None for r in hold_rows)
    assert res["cv_on_real"]["folds"] >= 2                           # C 的交叉验证
    md = rd.render_real_eval(res)
    assert "银标留出集" in md and "不是人工金标" in md and "按来源拆分" in md


def test_cli_evaluate_real_writes_both_reports(monkeypatch, tmp_path, data):
    tools, synth, gold = data
    train, hold = _fake_real(30), _fake_real(12, offset=100)
    rd.write_jsonl(tmp_path / "real_train.jsonl", train)
    rd.write_jsonl(tmp_path / "holdout.jsonl", hold)
    monkeypatch.setattr(cli, "REPORTS", tmp_path)
    code = cli.main(["evaluate-real", "--tools", str(DATA / "tools.jsonl"),
                     "--requests", str(DATA / "requests_train.jsonl"),
                     "--gold", str(DATA / "requests_gold.jsonl"),
                     "--real-train", str(tmp_path / "real_train.jsonl"),
                     "--holdout", str(tmp_path / "holdout.jsonl")])
    assert code == 0
    md = (tmp_path / "eval_t006.md").read_text(encoding="utf-8")
    assert "A 只用合成训练集" in md and "C 合成 + real_train" in md
    payload = json.loads((tmp_path / "eval_t006.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 6


# ---------------- CLI 与退出码 ----------------

def test_cli_realdata_translate_without_key_exits_4(monkeypatch, tmp_path):
    """没 key 时 open_client 直接停,退出码 4,不打堆栈。"""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    rd.write_jsonl(tmp_path / "pool.jsonl", [_pool_row(0, "Check my claim")])
    code = cli.main(["realdata", "translate", "--live", "--pool", str(tmp_path / "pool.jsonl"),
                     "--pool-zh", str(tmp_path / "pool_zh.jsonl"),
                     "--cost-file", str(tmp_path / "cost.json")])
    assert code == 4
    assert not (tmp_path / "pool_zh.jsonl").exists()


def test_cli_realdata_split_works_offline(tmp_path):
    rd.write_jsonl(tmp_path / "agree.jsonl", _fake_real(30))
    code = cli.main(["realdata", "split", "--agree", str(tmp_path / "agree.jsonl"),
                     "--real-train", str(tmp_path / "train.jsonl"),
                     "--holdout", str(tmp_path / "hold.jsonl")])
    assert code == 0
    assert len(rd.load_jsonl(tmp_path / "train.jsonl")) + \
        len(rd.load_jsonl(tmp_path / "hold.jsonl")) == 30


# ---------------- A6: 单路打标 ----------------

def test_single_label_writes_labeled_single_with_flag(tmp_path):
    """--single 写 labeled_single.jsonl,带 single_source,不进双路数据。"""
    pool = [{"id": "p-0", "text": "给张三发封邮件", "text_orig": "x", "source": "bfcl"}]
    jev = FakeJevClient(lambda s, q: _answers("single", ["comms"], ["send"]))
    out = rd.single_label(pool, ("jev", jev), cost=rd.CostBook(tmp_path / "c.json"), live=True)
    assert len(out["rows"]) == 1
    assert out["rows"][0]["label_source"] == "jev"
    assert out["rows"][0]["single_source"] is True


def test_split_does_not_read_single_source(tmp_path):
    """single_source 条目不进 real_train / real_holdout。"""
    agree = [{"id": "a", "text": "x", "need": "single", "domains": ["comms"],
              "actions": ["send"], "label_source": "jev+haiku"},
             {"id": "b", "text": "y", "need": "none", "domains": [],
              "actions": [], "label_source": "jev+haiku"}]
    single = [{"id": "c", "text": "z", "need": "single", "domains": ["comms"],
               "actions": ["send"], "label_source": "codebuddy", "single_source": True}]
    rd.write_jsonl(tmp_path / "agree.jsonl", agree)
    rd.write_jsonl(tmp_path / "single.jsonl", single)
    code = cli.main(["realdata", "split", "--agree", str(tmp_path / "agree.jsonl"),
                     "--real-train", str(tmp_path / "train.jsonl"),
                     "--holdout", str(tmp_path / "hold.jsonl")])
    assert code == 0
    train = rd.load_jsonl(tmp_path / "train.jsonl")
    hold = rd.load_jsonl(tmp_path / "hold.jsonl")
    assert not any(r.get("single_source") for r in train + hold)
    assert len(train) + len(hold) == 2


def test_evaluate_real_includes_d_variant_when_single_exists(data):
    """有 labeled_single 时 evaluate-real 多报一组 D。"""
    tools, synth, gold = data
    single = [{"id": "s-0", "text": "单路请求", "need": "single", "domains": ["comms"],
               "actions": ["send"], "label_source": "codebuddy", "single_source": True}]
    res = rd.evaluate_real(tools, synth, gold, _fake_real(30), _fake_real(12, offset=100),
                           labeled_single=single)
    variants = {r["variant"] for r in res["rows"]}
    assert "D 合成 + real_train + 单路" in variants


# ---------------- A7: 候选表兼容旧列名 ----------------

def test_load_gold_candidates_reads_old_jev_haiku_columns(tmp_path):
    """旧 CSV 用 jev_* / haiku_* 列名也能读。"""
    lines = ["id,text,text_orig,source,jev_need,jev_domains,haiku_need,haiku_domains,"
             "local_need,local_domains,need,domains,actions,note",
             "x,请求,orig,bfcl,single,comms,single,comms,single,comms,single,comms,send,"]
    path = tmp_path / "old.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    rows = rd.load_gold_candidates(path)
    assert len(rows) == 1 and rows[0]["need"] == "single"
    assert rows[0]["domains"] == ["comms"] and rows[0]["actions"] == ["send"]


def test_load_gold_candidates_ignores_labeler_name_columns(tmp_path):
    """读回时只看 id/need/domains/actions,多余的 labeler1_name 等列不影响。"""
    review, holdout = _review_rows(4), _fake_real(4)
    rows = rd.candidates(review, holdout, lambda t: [("single", ["comms"], ["read"])] * len(t),
                         n=8)
    path = tmp_path / "c.csv"
    rd.write_gold_candidates(path, rows)
    good = rd.load_gold_candidates(path)
    assert len(good) == 8 and all(set(r) >= {"text", "need", "domains", "actions"} for r in good)


# ---------------- T-010a:主域一致判定 ----------------

def _answers_probs(need, probs, actions=("send",)):
    """指定每个域 noul 概率的答案;域那一题缺 noul 就等于这一路没有概率可看。"""
    a = {"need": {"type": "choice", "choice": need,
                  "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": 0.6}}
    for d in DOMAIN_IDS:
        a[f"domain_{d}"] = {"type": "noul", "noul": probs.get(d, 0.1)}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
    return a


def test_primary_domain_picks_the_most_probable_domain():
    ans = _answers_probs("single", {"comms": 0.9, "system_misc": 0.6})
    assert rd.primary_domain(ans, ["comms", "system_misc"]) == "comms"
    # v2 的答案里没有两个新域那两题:主域只在它有的域里挑
    v2 = {f"domain_{d}": {"type": "noul", "noul": 0.2}
          for d in DOMAIN_IDS if d not in ("weather_geo", "entertainment")}
    v2["domain_business"] = {"type": "noul", "noul": 0.8}
    assert rd.primary_domain(v2, []) == "business"


def test_primary_domain_falls_back_to_the_first_domain_without_probabilities():
    answers = {"need": {"type": "choice", "choice": "single"}}      # 一个域的概率都没有
    assert rd.primary_domain(answers, ["comms", "file"]) == "comms"
    assert rd.primary_domain(None, []) == ""


def test_primary_rule_agrees_only_on_the_primary_domain():
    """A2 的五种情形:need 不同 / 都 none / 主域命中 / 主域不中 / 没有概率时退回第一个域。"""
    a = _lab("single", ["comms", "system_misc"], ["send"])
    ans = _answers_probs("single", {"comms": 0.9, "system_misc": 0.4})
    # 主域命中:一致,标签domains 只剩主域
    assert rd.agreement_outcome(a, _lab("single", ["comms"]), ans, None, "primary") == {
        "agreed": True, "domains": ["comms"], "primary_domain": "comms"}
    # 主域不中:不一致
    assert rd.agreement_outcome(a, _lab("single", ["file"]), ans, None, "primary") == {
        "agreed": False, "domains": [], "primary_domain": "comms"}
    # need 不同:不看域
    assert rd.agreement_outcome(a, _lab("multi", ["comms"]), ans, None, "primary")["agreed"] is False
    # 两路都 none:一致,标签为空
    assert rd.agreement_outcome(_lab("none"), _lab("none"), None, None, "primary") == {
        "agreed": True, "domains": [], "primary_domain": ""}
    # 第一路没有域概率:退回 domains 的第一个
    assert rd.agreement_outcome(_lab("single", ["comms", "file"]), _lab("single", ["comms"]),
                                {"need": {"choice": "single"}}, None, "primary") == {
        "agreed": True, "domains": ["comms"], "primary_domain": "comms"}


def test_primary_sym_also_accepts_the_second_labelers_primary():
    a = _lab("single", ["comms", "database"])
    ans_a = _answers_probs("single", {"comms": 0.9})
    ans_b = _answers_probs("single", {"database": 0.7})
    # 只有第二路的主域(database)落在第一路的域里:sym 算一致,primary 不算
    assert rd.agreement_outcome(a, _lab("single", ["file"]), ans_a, ans_b, "primary_sym") == {
        "agreed": True, "domains": ["database"], "primary_domain": "comms"}
    assert rd.agreement_outcome(a, _lab("single", ["file"]), ans_a, ans_b,
                                "primary")["agreed"] is False
    # 两个主域都命中时取第一路的
    both = rd.agreement_outcome(a, _lab("single", ["file", "comms"]), ans_a, ans_b, "primary_sym")
    assert both["agreed"] is True and both["domains"] == ["comms"]


def test_exact_rule_is_the_old_output_row_for_row(tmp_path):
    """A3:exact 口径逐行复现 T-006 的输出 —— need 相同且域集合相同才一致,domains 是整份。"""
    pool = [{"id": f"p-{i}", "text": f"请求{i}", "text_orig": f"r{i}", "source": "bfcl"}
            for i in range(6)]
    spec = {
        "请求0": (("single", ["comms"], ["send"]), ("single", ["comms"], ["send", "read"])),
        "请求1": (("single", ["comms", "file"], ["send"]), ("single", ["file", "comms"], ["read"])),
        "请求2": (("single", ["comms"], ["send"]), ("single", ["file"], ["send"])),
        "请求3": (("single", ["comms"], ["send"]), ("multi", ["comms"], ["send"])),
        "请求4": (("none", [], []), ("none", [], [])),
        "请求5": (("multi", ["comms", "file"], ["send"]), ("multi", ["comms"], ["send"])),
    }
    jev = FakeJevClient(lambda s, q: _answers(*spec[s][0]))
    haiku = FakeJevClient(lambda s, q: _answers(*spec[s][1]))
    out = rd.dual_label(pool, [("jev", jev), ("haiku", haiku)],
                        cost=rd.CostBook(tmp_path / "c.json"), live=True, agreement="exact")
    assert [r["id"] for r in out["agreed"]] == ["p-0", "p-1", "p-4"]
    assert [r["domains"] for r in out["agreed"]] == [["comms"], ["file", "comms"], []]
    assert out["agreed"][0]["actions"] == ["send"]          # 交集
    assert out["agreed"][1]["actions"] == ["send"]          # 交集为空取第一路的
    assert [r["id"] for r in out["review"]] == ["p-2", "p-3", "p-5"]
    # review 行带上的规则与主域是新字段,旧行为本身(谁进 review、标签是什么)没变
    assert all(r["agreement_rule"] == "exact" for r in out["review"])


def test_recompute_from_cache_makes_no_client_calls(tmp_path):
    """A5:不加 --live 时只读缓存重算,客户端一次都不调,结果与联网那次逐行相同。"""
    pool = [{"id": f"p-{i}", "text": f"请求{i}", "text_orig": f"r{i}", "source": "bfcl"}
            for i in range(3)]

    def jev_ans(state, q):
        return _answers_probs("single", {"comms": 0.9, "system_misc": 0.4}) \
            if state == "请求0" else _answers("none")

    def haiku_ans(state, q):
        return _answers_probs("single", {"comms": 0.8, "file": 0.4})

    jev1 = FakeJevClient(jev_ans)
    first = rd.dual_label(pool, [("jev", jev1), ("haiku", FakeJevClient(haiku_ans))],
                          cost=rd.CostBook(tmp_path / "c.json"), live=True)
    jev2, haiku2 = FakeJevClient(jev_ans), FakeJevClient(haiku_ans)
    second = rd.dual_label(pool, [("jev", jev2), ("haiku", haiku2)],
                           cost=rd.CostBook(tmp_path / "c.json"), live=False)
    assert jev2.calls == [] and haiku2.calls == []
    assert second["agreed"] == first["agreed"]
    assert second["review"] == first["review"]
    assert second["agreed"] and second["agreed"][0]["domains"] == ["comms"]   # 主域来自缓存里的概率


def _primary_review_rows():
    """三种待审行:need 相同(主域不一致)、need 不同、以及按 exact 判定的老行。"""
    def row(rid, need_a, need_b, rule, primary):
        return {"id": rid, "text": rid, "text_orig": rid, "source": "bfcl",
                "labelers": ["jev", "haiku"], "agreement_rule": rule, "primary_domain": primary,
                "jev": {"need": need_a, "domains": ["comms", "system_misc"], "actions": ["send"],
                        "ok": True},
                "haiku": {"need": need_b, "domains": ["file"], "actions": ["read"], "ok": True}}

    return [row("same-need", "single", "single", "primary", "comms"),
            row("diff-need", "single", "multi", "primary", "comms"),
            row("exact-old", "single", "single", "exact", "")]


def test_review_row_note_and_domains_follow_the_rule():
    rows = rd.candidates(_primary_review_rows(), [], lambda t: [("single", [], [])] * len(t), n=3)
    by_id = {r["id"]: r for r in rows}
    assert by_id["same-need"]["note"] == "主域不一致"
    assert by_id["same-need"]["domains"] == "comms"               # 只预填第一路的主域
    assert by_id["diff-need"]["note"] == "need 不一致"
    assert by_id["diff-need"]["domains"] == "comms;system_misc"    # need 不同时沿用整份
    assert by_id["exact-old"]["note"] == "两路不一致"               # exact 沿用旧的措辞


def test_evaluate_real_runs_when_the_train_set_has_no_multi(data):
    """A4:训练集一条 multi 都没有时 evaluate-real 不崩,报告写明 multi 无法预测。"""
    tools, synth, gold = data
    real = [r for r in _fake_real(30) if r["need"] != "multi"]
    hold = [r for r in _fake_real(12, offset=100) if r["need"] != "multi"]
    assert real and hold
    res = rd.evaluate_real(tools, synth, gold, real, hold)
    md = rd.render_real_eval(res)
    b_rows = [r for r in res["rows"] if r["variant"].startswith("B 只用 real_train")]
    assert b_rows and all("multi" in r["need_missing"] for r in b_rows)
    assert not [r for r in res["rows"]
                if r["variant"].startswith("A ") and r.get("need_missing")]
    assert "训练集缺 multi" in md and "multi 无法预测" in md
    assert md.count("训练集缺 multi") == 1          # 同一个训练集的两个评估集只写一行


# ---------------- A9: 不改动默认行为 / 不外泄 key ----------------

def test_no_api_key_literal_anywhere():
    # 拼出来而不是直接写,免得这条断言自己把 key 前缀带进仓库
    needle = "sk" + "-" + "or"
    files = list((ROOT / "mcpcls").glob("*.py")) + list((ROOT / "tests").glob("*.py")) \
        + list((ROOT / "scripts").glob("*.py")) + [ROOT / "README.md"]
    assert files
    for path in files:
        assert needle not in path.read_text(encoding="utf-8"), path


def test_evaluate_defaults_are_untouched(data):
    """不加新参数时 evaluate 的 need 头设置与 main 一致(阈值、规则、两级判定都不动)。"""
    tools, synth, gold = data
    router = ev.build_system("tfidf", tools, synth)
    assert router.threshold == 0.55 and router.top_k == 5
    res = ev.eval_requests(router, gold)
    assert res["n"] == len(gold) == 69


def test_review_row_need_uses_labelers_when_they_agree_on_need():
    """两路 need 相同、只差能力域时,建议 need 用两路的共同判断,不能被本地模型覆盖成别的。"""
    review = _review_rows(3)            # jev 的 need 轮流是 none/single/multi,haiku 恒为 single
    rows = rd.candidates(review, [], lambda t: [("none", [], [])] * len(t), n=3)
    by_id = {r["id"]: r for r in rows}
    assert by_id["rev-1"]["need"] == "single"     # 两路都是 single
    assert by_id["rev-0"]["need"] == "none"       # need 本身不一致 -> 本地模型的
