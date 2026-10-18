"""T-010 的测试:server 推断、去重规则、请求→工具映射、可复现性。

全程不联网:raw 目录用 tmp 里的假 BFCL 文件,`--no-sources` 避免动到 data/SOURCES.md。
真实数据的完整性(A4 那种"全表唯一")由 `tests/test_real_tools.py::test_repo_outputs`
读仓库里已生成的 data/*.jsonl 来查,文件不存在就跳过,不作为硬失败。
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_real_tools as brt          # noqa: E402


# ---------------------------------------------------------------- server 推断

@pytest.mark.parametrize("name, server", [
    ("requests.get", "requests"),
    ("acl_api.AclApi.retrieve_projects", "acl_api.AclApi"),
    ("Weather_1_GetWeather", "Weather"),
    ("Flights_4_SearchRoundtripFlights", "Flights"),
    ("get_current_weather", "misc"),
    ("ThinQ_Connect", "misc"),          # 有下划线但没有 _数字_,不算命名空间
    (".foo", "misc"),                   # 点号前是空串,退回 misc
    ("_1_bar", "misc"),
    ("", "misc"),
])
def test_infer_server(name, server):
    assert brt.infer_server(name) == server


def test_infer_server_dot_beats_underscore():
    """同时有点号和 _数字_ 时按点号规则走(任务卡里点号规则写在前面)。"""
    assert brt.infer_server("Flights_2_Search.search") == "Flights_2_Search"


# ---------------------------------------------------------------- 规范化

def test_norm_desc_and_canon_params_are_stable():
    assert brt.norm_desc("  Hello   World ") == "hello world"
    assert brt.norm_desc(None) == ""
    a = {"b": 1, "a": {"x": [1, 2]}}
    b = {"a": {"x": [1, 2]}, "b": 1}
    assert brt.canon_params(a) == brt.canon_params(b)


def test_dedup_key_ignores_description_whitespace():
    f1 = {"name": "f", "description": "Do the thing.", "parameters": {"type": "dict"}}
    f2 = {"name": "f", "description": "  do   the thing. ", "parameters": {"type": "dict"}}
    assert brt.dedup_key(f1) == brt.dedup_key(f2)


# ---------------------------------------------------------------- 造假数据

def _func(name, desc, props=("a",)):
    return {"name": name, "description": desc,
            "parameters": {"type": "dict",
                           "properties": {p: {"type": "string"} for p in props},
                           "required": list(props)}}


def _question(text):
    return [[{"role": "user", "content": text}]]


def write_fake(tmp_path, records_by_file, answers_by_file, pool_ids):
    """在 tmp 里搭一份迷你 BFCL:问题文件 + possible_answer + pool.jsonl。

    records_by_file: {文件名: [{"id":..., "question":..., "function":[...]}, ...]}
    answers_by_file: {文件名: [{"id":..., "ground_truth":[{函数名: {...}}]}, ...]}
    pool_ids: 池子里要保留的请求 id(写成 bfcl-<n>)
    """
    raw = tmp_path / "raw" / "bfcl"
    raw.mkdir(parents=True)
    for name, recs in records_by_file.items():
        with open(raw / name, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if answers_by_file:
        (raw / "possible_answer").mkdir(exist_ok=True)
        for name, recs in answers_by_file.items():
            with open(raw / "possible_answer" / name, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool = tmp_path / "pool.jsonl"
    with open(pool, "w", encoding="utf-8") as f:
        for i in pool_ids:
            f.write(json.dumps({"id": f"bfcl-{i}", "text_orig": "x" * 10,
                                "lang": "en", "source": "bfcl",
                                "source_file": "BFCL_v3_simple.json",
                                "hint": {}}, ensure_ascii=False) + "\n")
    return raw, pool


def run(tmp_path, raw, pool):
    """跑一次 build_real_tools,返回 (工具目录, 请求对应)。"""
    out = tmp_path / "out"
    brt.main(["--raw", str(raw), "--pool", str(pool), "--out-dir", str(out),
              "--stats", str(tmp_path / "stats.md"), "--no-sources"])
    tools = [json.loads(l) for l in open(out / "tools_real.jsonl", encoding="utf-8")]
    reqs = [json.loads(l) for l in open(out / "requests_real_tool_ids.jsonl",
                                        encoding="utf-8")]
    return tools, reqs


# ---------------------------------------------------------------- 去重与 #2

def test_duplicate_and_suffix(tmp_path):
    """同名不同描述 → 两条工具,第二条 tool_id 带 #2;完全相同的只留一条。"""
    f_a = _func("get_x", "Get x.")
    f_b = _func("get_x", "Get x, but different.", props=("a", "b"))
    rec = {"id": "simple_0", "question": _question("Please get x for me."),
           "function": [f_a, f_b]}
    rec2 = {"id": "simple_1", "question": _question("Fetch x again right now."),
            "function": [f_a]}            # 和 f_a 完全相同,不该新增
    raw, pool = write_fake(tmp_path, {"BFCL_v3_simple.json": [rec, rec2]},
                           {"BFCL_v3_simple.json": [
                               {"id": "simple_0", "ground_truth": [{"get_x": {}}]},
                               {"id": "simple_1", "ground_truth": [{"get_x": {}}]}]},
                           [0, 1])
    tools, reqs = run(tmp_path, raw, pool)
    ids = {t["tool_id"] for t in tools}
    assert sorted(ids) == ["misc/get_x", "misc/get_x#2"]
    assert len(ids) == len(tools)
    # 请求 0 的答案 get_x 在候选集里有两条同名,按候选定义精确匹配:两个都算命中
    assert set(reqs[0]["tool_ids"]) == ids
    # 请求 1 只带了 f_a,所以只映射到其中一条
    assert len(reqs[1]["tool_ids"]) == 1
    assert reqs[1]["candidate_tool_ids"] == reqs[1]["tool_ids"]


def test_n_requests_counts_records(tmp_path):
    rec = {"id": "simple_0", "question": _question("Please get x for me."),
           "function": [_func("get_x", "Get x.")]}
    rec2 = {"id": "simple_1", "question": _question("Fetch x again right now."),
            "function": [_func("get_x", "Get x."), _func("get_y", "Get y.")]}
    raw, pool = write_fake(tmp_path, {"BFCL_v3_simple.json": [rec, rec2]},
                           {"BFCL_v3_simple.json": [
                               {"id": "simple_0", "ground_truth": [{"get_x": {}}]},
                               {"id": "simple_1", "ground_truth": [{"get_y": {}}]}]},
                           [0, 1])
    tools, _ = run(tmp_path, raw, pool)
    by_id = {t["tool_id"]: t for t in tools}
    assert by_id["misc/get_x"]["n_requests"] == 2
    assert by_id["misc/get_y"]["n_requests"] == 1
    assert by_id["misc/get_x"]["source_files"] == ["BFCL_v3_simple.json"]


# ---------------------------------------------------------------- 请求映射

def test_request_mapping_needs_candidate_or_unique_name(tmp_path):
    """答案里的名字不在候选集里时,只有目录里唯一同名才采信,否则记失败。"""
    # 第 3 条记录里的 dup 有两个变体,但它不在池子里,所以只进目录、不出对应关系;
    # 第 2 条请求的答案是 dup,dup 不在它自己的候选集里且目录里有两个 → 记失败。
    raw, pool = write_fake(
        tmp_path,
        {"BFCL_v3_simple.json": [
            {"id": "simple_0", "question": _question("Call the only one please."),
             "function": [_func("cand", "A candidate.")]},
            {"id": "simple_1", "question": _question("Use the only one now."),
             "function": [_func("only_one", "The only one.")]},
            {"id": "simple_2", "question": _question("Something else entirely."),
             "function": [_func("other", "Other.")]},
            {"id": "simple_3", "question": _question("Yet another request here."),
             "function": [_func("dup", "First variant."),
                          _func("dup", "Second variant.", props=("a", "b"))]}]},
        {"BFCL_v3_simple.json": [
            {"id": "simple_0", "ground_truth": [{"only_one": {}}]},   # 唯一同名 → 采信
            {"id": "simple_1", "ground_truth": [{"cand": {}}]},
            {"id": "simple_2", "ground_truth": [{"dup": {}}]},        # 二义 → 失败
            {"id": "simple_3", "ground_truth": [{"dup": {}}]}]},
        [0, 1, 2])
    _tools, reqs = run(tmp_path, raw, pool)
    assert [r["id"] for r in reqs] == ["bfcl-0", "bfcl-1", "bfcl-2"]
    assert reqs[0]["tool_ids"] == ["misc/only_one"]
    assert reqs[1]["tool_ids"] == ["misc/cand"]
    assert reqs[2]["tool_ids"] == []        # 映射失败时留空,不猜
    assert reqs[2]["candidate_tool_ids"] == ["misc/other"]


def test_missing_possible_answer_is_skipped(tmp_path):
    rec = {"id": "simple_0", "question": _question("Please get x for me."),
           "function": [_func("get_x", "Get x.")]}
    raw, pool = write_fake(tmp_path, {"BFCL_v3_simple.json": [rec]}, None, [0])
    out = tmp_path / "out"
    brt.main(["--raw", str(raw), "--pool", str(pool), "--out-dir", str(out),
              "--stats", str(tmp_path / "stats.md"), "--no-sources"])
    tools = [json.loads(l) for l in open(out / "tools_real.jsonl", encoding="utf-8")]
    reqs = [json.loads(l) for l in open(out / "requests_real_tool_ids.jsonl",
                                        encoding="utf-8")]
    assert len(tools) == 1                  # 目录不依赖 possible_answer
    assert reqs == []                       # 对应关系跳过


def test_reproducible_bytes(tmp_path):
    """A2:同一份 raw 跑两次,两个输出文件逐字节相同。"""
    recs = [{"id": f"simple_{i}", "question": _question(f"Request number {i} here."),
             "function": [_func(f"f{i % 3}", f"Do {i % 3}.")]} for i in range(6)]
    answers = [{"id": f"simple_{i}",
                "ground_truth": [{f"f{i % 3}": {}}]} for i in range(6)]
    raw, pool = write_fake(tmp_path, {"BFCL_v3_simple.json": recs},
                           {"BFCL_v3_simple.json": answers}, list(range(6)))
    first = tmp_path / "a"
    second = tmp_path / "b"
    for out in (first, second):
        brt.main(["--raw", str(raw), "--pool", str(pool), "--out-dir", str(out),
                  "--stats", str(tmp_path / f"{out.name}.md"), "--no-sources"])
    for name in ("tools_real.jsonl", "requests_real_tool_ids.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


# ---------------------------------------------------------------- 仓库产物

def test_repo_outputs():
    """A4:仓库里已生成的两个 jsonl 自洽(tool_id 唯一、引用都能找到)。

    data/raw 不进版本库,别人 clone 下来可能没有产物,所以文件缺失时跳过。
    """
    tools_path = ROOT / "data" / "tools_real.jsonl"
    reqs_path = ROOT / "data" / "requests_real_tool_ids.jsonl"
    if not (tools_path.exists() and reqs_path.exists()):
        pytest.skip("data/tools_real.jsonl 还没生成(需要 data/raw/bfcl)")
    tools = [json.loads(l) for l in open(tools_path, encoding="utf-8") if l.strip()]
    reqs = [json.loads(l) for l in open(reqs_path, encoding="utf-8") if l.strip()]
    ids = [t["tool_id"] for t in tools]
    assert len(set(ids)) == len(ids), "tool_id 有重复"
    for t in tools:
        assert t["source"] == "bfcl"
        assert t["tool_id"].split("#")[0] == f"{t['server']}/{t['name']}"
        assert t["inputSchema"] is not None
        assert "domains" not in t and "actions" not in t, "A 部分不编标签"
    for r in reqs:
        for tid in r["tool_ids"] + r["candidate_tool_ids"]:
            assert tid in set(ids), f"{r['id']} 引用了目录里没有的 {tid}"
        assert r["answer_source"].startswith("possible_answer/")


def test_sources_has_tool_catalog_section():
    """A5:data/SOURCES.md 里有工具目录一节,带许可、署名和 sha256。"""
    text = (ROOT / "data" / "SOURCES.md").read_text(encoding="utf-8")
    assert "## 工具目录" in text
    assert "Apache-2.0" in text
    assert "gorilla-llm" in text
    assert "possible_answer" in text
    assert "`bfcl/possible_answer/BFCL_v3_simple.json`" in text


# ---------------------------------------------------------------- T-010b 真实工具打标 / 风险 / 检索

import csv

from mcpcls import compare
from mcpcls import evaluate as ev
from mcpcls import tools_real as tr
from mcpcls.jev import FakeJevClient, tool_questions
from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS


def _tool_answers(domains=(), actions=()):
    """为 FakeJevClient 生成一份 20 题的答案。"""
    a = {}
    for d in DOMAIN_IDS:
        a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x in actions else 0.1}
    return a


def _make_tools(n):
    return [{"tool_id": f"misc/t{i}", "name": f"t{i}",
             "description": f"Tool number {i} does things.",
             "server": "misc",
             "inputSchema": {"type": "object", "properties": {}}}
            for i in range(n)]


def test_tool_questions_has_20_questions_from_taxonomy():
    """A2:tool_questions 正好 20 题,域和动作 id 与 taxonomy 一一对应。"""
    q = tool_questions()
    assert len(q) == 20
    assert set(q) == {f"domain_{d}" for d in DOMAIN_IDS} | {f"action_{a}" for a in ACTION_IDS}
    for d in DOMAIN_IDS:
        assert q[f"domain_{d}"]["type"] == "noul"
    for a in ACTION_IDS:
        assert q[f"action_{a}"]["type"] == "noul"


def test_label_tools_second_run_uses_cache_no_calls(monkeypatch, tmp_path):
    """A3:第二次跑 label(不加 --live)时客户端调用 0 次,输出逐字节相同。"""
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "cache")
    tools = _make_tools(3)
    ans = _tool_answers(domains=["system_misc"], actions=["read"])
    client = FakeJevClient(lambda state, questions: ans)

    out1 = tr.label_tools(tools, [("jev", client), ("haiku", client)],
                          live=True, workers=1)
    assert len(client.calls) == 6           # 3 工具 × 2 路,workers=1 仍顺序问

    client.calls.clear()
    out2 = tr.label_tools(tools, [("jev", client), ("haiku", client)],
                          live=False, workers=1)
    assert len(client.calls) == 0
    assert json.dumps(out1["rows"], ensure_ascii=False, sort_keys=True) == \
           json.dumps(out2["rows"], ensure_ascii=False, sort_keys=True)


def test_label_tools_marks_disagreement_and_missing(monkeypatch, tmp_path):
    """A4:不一致行保留,缺答案行单独计数。"""
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "cache")
    tools = _make_tools(2)
    jev = FakeJevClient(lambda state, questions: _tool_answers(
        domains=["system_misc"], actions=["read"]))
    haiku = FakeJevClient(lambda state, questions: _tool_answers(
        domains=["comms"], actions=["send"]))
    out = tr.label_tools(tools, [("jev", jev), ("haiku", haiku)], live=True)
    assert len(out["rows"]) == 2
    assert out["missing"]["jev"] == 0 and out["missing"]["haiku"] == 0
    # 两路域不同:不一致,domains 用基准路 Jev 的
    row = out["rows"][0]
    assert row["agreed"] is False
    assert row["domains"] == ["system_misc"]
    assert set(row["actions"]) == {"read", "send"}

    # 模拟 Haiku 全部失败;换一批文本避免命中上一轮的缓存
    tools2 = [{"tool_id": f"misc/u{i}", "name": f"u{i}",
               "description": f"Other tool {i}.", "server": "misc",
               "inputSchema": {"type": "object", "properties": {}}}
              for i in range(2)]
    bad = FakeJevClient(lambda state, questions: _tool_answers(), fail=True)
    out2 = tr.label_tools(tools2, [("jev", jev), ("haiku", bad)], live=True)
    assert out2["missing"]["haiku"] == 2
    assert all(not r["agreed"] for r in out2["rows"])


def test_labeled_jsonl_has_one_row_per_tool():
    """A4:2405 个工具每个都有一行。"""
    rows = ev.load_jsonl(ROOT / "data" / "tools_real_labeled.jsonl")
    assert len(rows) == 2405
    assert len({r["tool_id"] for r in rows}) == 2405


def test_eval_t010_reports_agreement_and_missing():
    """A4/A6:报告里有一致率、缺答案、中英文过滤表等关键信息。"""
    text = (ROOT / "reports" / "eval_t010.md").read_text(encoding="utf-8")
    assert "两路一致" in text and "缺答案" in text
    sec3 = text.split("## 3. 真实 Recall@k")[1].split("## 4.")[0]
    assert "| 请求头 | 查询语言 | 过滤 | 严格 R@1" in sec3
    assert "中文" in sec3 and "英文" in sec3
    assert "开(金标域)" in sec3
    assert "金标工具被过滤掉的比例" in sec3


def test_assign_folds_excludes_fold_ids_from_training():
    """A5:任何一折的训练集里都不含本折的 id。"""
    ids = [f"id_{i}" for i in range(20)]
    fold_of = tr.assign_folds(ids, folds=5, seed=0)
    for f in range(5):
        test_ids = {i for i, fo in fold_of.items() if fo == f}
        train = tr.fold_train_rows([], [{"id": i} for i in ids], test_ids)
        train_ids = {r["id"] for r in train}
        assert not (train_ids & test_ids)


def test_risk_audit_csv_has_60_rows_with_empty_gu_risk():
    """A7:风险抽检表 60 行,列齐全,gu_risk 为空。"""
    path = ROOT / "data" / "tools_real_risk_audit.csv"
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    assert len(rows) == 61           # 表头 + 60 行
    assert rows[0] == list(tr.AUDIT_COLUMNS)
    for row in rows[1:]:
        assert len(row) == len(tr.AUDIT_COLUMNS)
        assert row[-1] == ""         # gu_risk 留空


def test_risk_metrics_counts_l3_recall_and_under_rate():
    """风险指标对银标和预测做 L3 召回 / 低报率统计。"""
    gold = ["L0", "L3", "L2", "L3"]
    pred = ["L0", "L3", "L0", "L2"]
    m = tr.risk_metrics(gold, pred)
    assert m["accuracy"] == 0.5
    assert m["l3_recall"] == 0.5
    assert m["under_rate"] == 0.5    # L2->L0 和 L3->L2 各低报 1 条 / 4 条
    assert m["confusion"]["L3"]["L2"] == 1


def test_audit_rows_prioritises_disagreements_then_l3():
    """抽检表优先抽两路风险不同的工具,其余从银标 L3 中按固定 seed 抽。"""
    rows = [
        {"tool_id": "a", "agreed": True, "risk_silver": "L3"},
        {"tool_id": "b", "agreed": True, "risk_silver": "L3"},
        {"tool_id": "c", "agreed": True, "risk_silver": "L0"},
        {"tool_id": "d", "agreed": True, "risk_silver": "L3"},
    ]
    risk_by_id = {
        "jev": {"a": "L3", "b": "L3", "c": "L0", "d": "L2"},
        "haiku": {"a": "L3", "b": "L2", "c": "L0", "d": "L3"},
    }
    picked = tr.audit_rows(rows, risk_by_id, n=3, disagree_max=2, seed=0)
    ids = [r["tool_id"] for r in picked]
    assert set(ids[:2]) == {"b", "d"}     # 两路风险不同的优先
    assert len(ids) == 3
    assert all(r["agreed"] and r["risk_silver"] == "L3" for r in picked[2:])
