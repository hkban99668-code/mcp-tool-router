"""T-009 的测试:need 说明 v3、14 域 taxonomy、缓存按版本隔离、v3 输出路径、新旧口径对照。

全程离线:`no_network` 把没被单独 mock 的 urlopen 打成 AssertionError,
需要"两路打标"的地方一律用 `FakeJevClient`。
"""

import json
import pathlib
import urllib.request

import pytest

from mcpcls import cli, compare
from mcpcls import evaluate as ev
from mcpcls import label_compare as lc
from mcpcls import realdata as rd
from mcpcls.jev import (NEED_VERSIONS, FakeJevClient, answers_to_labels, need_question,
                        request_questions)
from mcpcls.taxonomy import ACTION_IDS, DOMAINS, DOMAIN_IDS, NEED_IDS, category_cards

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("测试里不应发出任何网络请求")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch, tmp_path):
    """缓存落到 tmp:任何测试都不会碰到 .cache/answers 里的真实答案。"""
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "answers")


# ---------------- A2: v3 的 need 说明逐字照抄任务卡 ----------------

_NEED_V3_INSTRUCTIONS = (
    "判断这条用户请求能否只靠语言模型自身的知识完成。只要需要读取、查询或改动用户的外部数据或系统，"
    "或者需要实时、最新、会变化的外部信息（价格、汇率、天气、新闻、榜单、库存、营业时间、航班车次、"
    "比分等），就需要工具。数学计算、换算、解释概念、写作翻译不需要工具。只按请求本身判断，"
    "不要替用户设想没说出来的需求。")
_NEED_V3_CRITERIA = {
    "none": "不需要任何外部工具：解释概念、常识或知识问答、数学计算与单位换算、写作、翻译、润色、"
            "出主意、闲聊或表达观点，模型凭自身知识就能直接回答，不依赖实时信息，"
            "也不读取或改动用户的任何外部数据。",
    "single": "需要一个外部工具完成一件事：要对外部数据或系统做一件事（发消息、查改日程、读写文件、"
              "查数据库、操作网页、查日志或部署、改订单或客户记录），或者需要查一次实时、最新的外部信息"
              "（价格、天气、新闻、榜单、库存、航班等）。即使请求很短、语气很随意，也是 single。",
    "multi": "需要先后或组合使用两个及以上不同的工具或系统，例如先查日程再发邮件、先查天气再订车、"
             "先转写录音再存成文件。",
}


def test_need_v3_text_matches_the_task_card():
    q = need_question("v3")
    assert q["type"] == "choice"
    assert q["instructions"] == _NEED_V3_INSTRUCTIONS
    assert q["criteria"] == _NEED_V3_CRITERIA
    assert request_questions("v3")["need"] == q
    assert NEED_VERSIONS == ("v1", "v2", "v3")


def test_need_v3_is_a_real_rewrite_of_v2_and_v1_is_untouched():
    v1, v2, v3 = need_question("v1"), need_question("v2"), need_question("v3")
    assert v3["instructions"] != v2["instructions"]
    assert "会变化的外部信息" in v3["instructions"] and "数学计算" in v3["instructions"]
    for key in NEED_IDS:
        assert v3["criteria"][key] != v2["criteria"][key]
    assert v1["instructions"].startswith("判断这条用户请求是否需要调用外部工具")


# ---------------- A3: 14 个域与题数 ----------------

_EXPECTED_IDS = ("file", "code", "database", "search_web", "browser", "comms", "calendar_task",
                 "knowledge", "cloud_ops", "business", "media", "weather_geo", "entertainment",
                 "system_misc")


def test_domain_ids_are_fourteen_in_the_order_the_card_asked_for():
    assert DOMAIN_IDS == _EXPECTED_IDS
    # 两个新域插在 media 之后、system_misc 之前
    assert DOMAIN_IDS.index("media") + 1 == DOMAIN_IDS.index("weather_geo")
    assert DOMAIN_IDS.index("entertainment") + 1 == DOMAIN_IDS.index("system_misc")


def test_new_categories_have_the_text_from_the_card():
    by_id = {c.id: c for c in DOMAINS}
    wg, ent = by_id["weather_geo"], by_id["entertainment"]
    assert wg.name == "天气地图与出行"
    assert wg.definition == "天气预报、地图与地点查询、导航路线、航班火车与打车等出行查询和预订"
    assert wg.examples == ("北京明天天气怎么样", "查一下从机场到酒店怎么走")
    assert wg.boundary == "订酒店、买票付款的订单管理归 business;只是查信息归这里"
    assert ent.name == "娱乐资讯与推荐"
    assert ent.definition == "影视、音乐、体育、游戏、演出的信息查询与推荐(查片单、找歌、看比分、查演出)"
    assert ent.examples == ("推荐几部最近的科幻片", "湖人昨晚比分多少")
    assert ent.boundary == "生成或处理图片音视频归 media;购票付款归 business"


def test_system_misc_only_lost_the_weather_example_and_words():
    misc = {c.id: c for c in DOMAINS}["system_misc"]
    assert misc.definition == "Shell、本地系统、设备控制等以上都不属于的通用工具"
    assert misc.examples == ("运行 ls -la", "把空调调到 26 度")
    assert "天气" not in misc.definition and "地图" not in misc.definition


def test_the_other_eleven_domains_are_untouched():
    names = {c.id: c for c in DOMAINS}
    assert [names[d].name for d in _EXPECTED_IDS[:11]] == [
        "文件与文档", "代码与开发", "数据库与数据分析", "搜索与网页", "浏览器与自动化", "通讯协作",
        "日程与任务", "知识库与笔记", "云与运维", "业务系统", "多媒体"]
    assert ACTION_IDS == ("read", "create", "update", "delete", "execute", "send")
    assert NEED_IDS == ("none", "single", "multi")


def test_category_cards_and_questions_follow_the_new_domains():
    cards = category_cards()
    assert "weather_geo" in cards and "entertainment" in cards
    q = request_questions("v3")
    assert len(q) == 1 + len(DOMAIN_IDS) + len(ACTION_IDS) == 1 + 14 + 6
    assert q["domain_weather_geo"]["type"] == "noul"
    assert "天气" in q["domain_weather_geo"]["instructions"]
    assert {c.id: c for c in DOMAINS}["entertainment"].boundary in q["domain_entertainment"]["instructions"]
    assert len(request_questions("v2")) == len(q)      # 题数与 need 版本无关


# ---------------- A4: v3 的缓存与 v2 分开 ----------------

_TEXT = "北京明天天气怎么样"


def _answers(need="single", domains=("system_misc",), with_new_domains=True):
    """with_new_domains=False 造出 v2 时代的答案:没有 weather_geo / entertainment 那两题。"""
    a = {"need": {"type": "choice", "choice": need,
                  "probabilities": {n: 0.3 for n in NEED_IDS}, "confidence": 0.6}}
    for d in DOMAIN_IDS:
        if not with_new_domains and d in ("weather_geo", "entertainment"):
            continue
        a[f"domain_{d}"] = {"type": "noul", "noul": 0.9 if d in domains else 0.1}
    for x in ACTION_IDS:
        a[f"action_{x}"] = {"type": "noul", "noul": 0.9 if x == "read" else 0.1}
    return a


def _write_cache(client_name, need_version, answers):
    path = compare.cache_path(client_name, need_version, "pool")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": _TEXT, "answers": answers, "latency_ms": 1.0,
                            "cost": 0.0, "ok": True, "error": ""}, ensure_ascii=False) + "\n")


def test_v3_reads_its_own_cache_and_ignores_the_v2_one():
    items = [{"text": _TEXT}]
    _write_cache("jev@typesafe-jev", "v2", _answers("none", with_new_domains=False))
    v3 = compare.fetch_answers(None, "jev@typesafe-jev", items, "v3", "pool", live=False)
    assert v3[0]["ok"] is False and v3[0]["answers"] is None      # 有 v2 答案也不读
    v2 = compare.fetch_answers(None, "jev@typesafe-jev", items, "v2", "pool", live=False)
    assert v2[0]["ok"] is True
    assert sorted(p.name for p in compare.CACHE_DIR.glob("jev_at_typesafe-jev-*")) == [
        "jev_at_typesafe-jev-v2-pool.jsonl"]


def test_v2_answers_have_no_question_for_the_new_domains():
    """这就是不能把 v2 的缓存当 v3 用的原因:里面根本没有新域那两题的答案。"""
    labels = answers_to_labels(_answers("single", domains=("weather_geo",), with_new_domains=False))
    assert "weather_geo" not in labels["domains"]      # 缺失的 key 一律按 0.0 处理
    assert answers_to_labels(_answers("single", domains=("weather_geo",)))["domains"] == ["weather_geo"]


# ---------------- A5: v3 的输出全部写进新目录 ----------------

def _pool_row(i, source="bfcl"):
    return {"id": f"{source}-{i}", "text": f"真实请求 {i} 号", "text_orig": f"request {i}",
            "source": source, "hint": {"math_like": i % 3 == 0}}


def _fake_labelers():
    """两路给出同样的答案 -> 6 条全进 agree;三种 need 都要有,训练集才不会因为单类报错。"""

    def ans(state, q):
        i = int(state.split("请求 ")[1].split(" 号")[0])
        need = ("none", "single", "multi")[i % 3]
        return _answers(need, ("weather_geo",) if need != "none" else ())

    return iter([FakeJevClient(ans), FakeJevClient(ans)])


def _snapshot(d):
    return {p.name: p.stat().st_mtime_ns for p in d.iterdir()} if d.exists() else None


def test_realdata_steps_write_everything_into_the_v3_dir(monkeypatch, tmp_path):
    before = _snapshot(DATA / "v3")
    v3 = tmp_path / "v3"
    pool_zh = tmp_path / "pool_zh.jsonl"
    rd.write_jsonl(pool_zh, [_pool_row(i) for i in range(12)])
    clients = _fake_labelers()
    monkeypatch.setattr(rd, "open_client", lambda name, **kw: next(clients))

    assert cli.main(["realdata", "label", "--live", "--pool-zh", str(pool_zh),
                     "--agree", str(v3 / "labeled_agree.jsonl"),
                     "--review", str(v3 / "review_queue.jsonl"),
                     "--cost-file", str(tmp_path / "cost.json")]) == 0
    agree = rd.load_jsonl(v3 / "labeled_agree.jsonl")
    assert len(agree) == 12
    assert {r["need"] for r in agree} == {"none", "single", "multi"}
    assert all(r["domains"] == ["weather_geo"] for r in agree if r["need"] != "none")

    assert cli.main(["realdata", "split", "--agree", str(v3 / "labeled_agree.jsonl"),
                     "--real-train", str(v3 / "real_train.jsonl"),
                     "--holdout", str(v3 / "real_holdout.jsonl")]) == 0
    assert (v3 / "real_train.jsonl").exists() and (v3 / "real_holdout.jsonl").exists()

    assert cli.main(["realdata", "candidates", "--review", str(v3 / "review_queue.jsonl"),
                     "--holdout", str(v3 / "real_holdout.jsonl"),
                     "--candidates", str(v3 / "gold_candidates.csv")]) == 0
    assert (v3 / "gold_candidates.csv").exists() and (v3 / "gold_candidates_README.md").exists()

    reports = tmp_path / "reports"
    assert cli.main(["evaluate-real", "--tools", str(DATA / "tools.jsonl"),
                     "--requests", str(DATA / "requests_train.jsonl"),
                     "--gold", str(DATA / "requests_gold.jsonl"),
                     "--real-train", str(v3 / "real_train.jsonl"),
                     "--holdout", str(v3 / "real_holdout.jsonl"),
                     "--report-md", str(reports / "eval_v3.md"),
                     "--report-json", str(reports / "eval_v3.json")]) == 0
    assert "C 合成 + real_train" in (reports / "eval_v3.md").read_text(encoding="utf-8")

    assert {p.name for p in v3.iterdir()} == {"labeled_agree.jsonl", "review_queue.jsonl",
                                              "real_train.jsonl", "real_holdout.jsonl",
                                              "gold_candidates.csv", "gold_candidates_README.md"}
    # 仓库里的 data/v3 不能被测试写入(Claude 的真实 v3 打标也写在那里,所以只看内容有没有变)
    assert _snapshot(DATA / "v3") == before


def test_missing_files_are_reported_not_silently_counted_as_zero(tmp_path):
    side = lc.load_side(tmp_path / "nope")
    assert side["missing"] == list(lc.FILES.values())
    assert side["agree"] == []


# ---------------- A6: compare-labels 在两套小数据上跑通 ----------------

_OLD = ("none", "single", "multi", "none", "single", "multi",
        "none", "single", "multi", "none", "single", "multi")
_NEW = ("single", "none", "none", "multi", "single", "multi",
        "none", "multi", "multi", "single", "single", "multi")


def _agree_rows(needs, v3=False):
    out = []
    for i, need in enumerate(needs):
        out.append({"id": f"bfcl-{i}" if i < 6 else f"apibank-{i}",
                    "text": f"请求 {i}", "text_orig": f"request {i}",
                    "source": "bfcl" if i < 6 else "apibank", "need": need,
                    "domains": [] if need == "none" else (["weather_geo"] if v3 else ["system_misc"]),
                    "actions": [] if need == "none" else ["read"],
                    "label_source": "jev+haiku", "hint": {"math_like": i % 3 == 0}})
    return out


def _review_rows(n, ok=True):
    return [{"id": f"rev-{i}", "text": f"待人工 {i}", "text_orig": f"review {i}", "source": "bfcl",
             "labelers": ["jev", "haiku"],
             "jev": {"need": "single", "domains": ["file"], "actions": ["read"], "ok": ok},
             "haiku": {"need": "none", "domains": [], "actions": [], "ok": True}} for i in range(n)]


def _write_side(d, v3=False):
    d.mkdir(parents=True, exist_ok=True)
    agree = _agree_rows(_NEW if v3 else _OLD, v3=v3)
    rd.write_jsonl(d / "labeled_agree.jsonl", agree)
    rd.write_jsonl(d / "review_queue.jsonl", _review_rows(3) + _review_rows(1, ok=False))
    rd.write_jsonl(d / "real_train.jsonl", agree[:8])
    rd.write_jsonl(d / "real_holdout.jsonl", agree[8:])


@pytest.fixture(scope="module")
def sides(tmp_path_factory):
    old = tmp_path_factory.mktemp("old")
    new = tmp_path_factory.mktemp("v3")
    _write_side(old), _write_side(new, v3=True)
    tools = ev.load_jsonl(DATA / "tools.jsonl")
    train = ev.load_jsonl(DATA / "requests_train.jsonl")
    return old, new, tools, train


def test_compare_labels_report_has_every_section(tmp_path, sides):
    old_dir, new_dir, tools, train = sides
    md_path, json_path = tmp_path / "cmp.md", tmp_path / "cmp.json"
    code = cli.main(["compare-labels", "--old", str(old_dir), "--new", str(new_dir),
                     "--tools", str(DATA / "tools.jsonl"),
                     "--requests", str(DATA / "requests_train.jsonl"),
                     "--report-md", str(md_path), "--report-json", str(json_path)])
    assert code == 0
    md = md_path.read_text(encoding="utf-8")
    for head in ("## 1. 数据规模与双路一致率", "## 2. 一致集里的 need 分布与能力域样本数",
                 "## 3. 留出集里", "## 4. need 转移矩阵", "## 5. C 方案"):
        assert head in md
    assert "只能看趋势,不能逐点比较" in md
    assert "weather_geo" in md and "entertainment" in md      # 新域单独一行

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    # 每侧 12 条一致 + 4 条待人工,其中 1 条有一路没答 -> 有效 15 条、一致 12 条
    for key in ("old", "new"):
        a = payload["agreement"][key]
        assert a["n"] == 15 and a["agreed"] == 12 and a["skipped"] == 1
        assert set(a["by_source"]) == {"bfcl", "apibank"}
    assert payload["domain_counts"]["new"]["weather_geo"] > 0
    assert payload["domain_counts"]["old"]["weather_geo"] == 0
    assert payload["holdout"]["old"]["n"] == payload["holdout"]["new"]["n"] == 4
    assert payload["eval"]["old"] and payload["eval"]["new"]
    for key in ("old", "new"):
        assert 0.0 <= payload["eval"][key]["need_accuracy"] <= 1.0
        assert payload["eval"][key]["domain"]["micro_f1"] is not None


def test_transition_matrix_counts_match_the_fake_data(sides):
    old_dir, new_dir, tools, train = sides
    tr = lc.compare_labels(old_dir, new_dir, tools, train)["transitions"]
    assert tr["n_compared"] == 12 and tr["only_old"] == tr["only_new"] == 0
    rows = tr["groups"]["全部"]
    assert rows.get("none->single") == 2        # 第 0、9 条
    assert rows.get("single->none") == 1        # 第 1 条
    assert rows.get("multi->none") == 1         # 第 2 条
    assert rows.get("none->multi") == 1         # 第 3 条
    assert rows.get("single->multi") == 1       # 第 7 条
    assert rows.get("multi->single") is None
    assert "source=bfcl" in tr["groups"] and "math_like=True" in tr["groups"]


def test_eval_version_is_none_without_train_or_holdout(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert lc.eval_version("tfidf", [], [], lc.load_side(d)) is None


# ---------------- T-010a:多套口径同时比 + multi 统计 ----------------

def test_compare_sides_takes_more_than_two(tmp_path, sides):
    """一个口径目录一套四份 jsonl;`compare_sides` 比几套都行,report 里每套一行。"""
    old_dir, new_dir, tools, train = sides
    third = tmp_path / "v3-primary"
    _write_side(third, v3=True)
    res = lc.compare_sides([("v2-exact", old_dir), ("v3-exact", new_dir),
                            ("v3-primary", third)], tools, train)
    assert res["keys"] == ["old", "new", "v3-primary"]
    assert res["labels"]["new"] == "v3-exact" and res["labels"]["v3-primary"] == "v3-primary"
    for key in res["keys"]:
        assert res["agreement"][key]["n"] == 15
        assert set(res["need_distribution"][key]) == set(NEED_IDS)
    md = lc.render(res)
    assert md.count("| v3-primary |") >= 3          # 总表、第一节、第五节各一行
    assert "| v2-exact |" in md and "| v3-exact |" in md


def test_multi_stats_counts_each_labeler_and_the_mixed_rows(sides):
    old_dir, _, _, _ = sides
    m = lc.multi_stats(lc.load_side(old_dir))
    assert m["labelers"] == ["jev", "haiku"]
    # 一致集里 4 条 multi(两路的判断相同),review 行两路是 single / none,不算 mixed
    assert m["jev_multi"] == m["haiku_multi"] == m["both_multi"] == 4
    assert m["multi_vs_single"] == {"total": 0, "by_source": {}}

    side = {"agree": [{"need": "multi"}],
            "review": [{"labelers": ["jev", "haiku"], "source": src,
                        "jev": {"need": "multi", "ok": True},
                        "haiku": {"need": "single", "ok": True}}
                       for src in ("bfcl", "apibank")]}
    got = lc.multi_stats(side)
    assert (got["jev_multi"], got["haiku_multi"], got["both_multi"]) == (3, 1, 1)
    assert got["multi_vs_single"]["total"] == 2
    assert got["multi_vs_single"]["by_source"] == {"bfcl": 1, "apibank": 1}


def test_cli_compare_labels_accepts_extra_sides(tmp_path, sides):
    """`目录=标签` 的写法:一套口径一个目录,名字写进报告的行首。"""
    old_dir, new_dir, tools, train = sides
    md_path, json_path = tmp_path / "cmp3.md", tmp_path / "cmp3.json"
    code = cli.main(["compare-labels", "--old", f"{old_dir}=v2-exact",
                     "--new", f"{new_dir}=v3-exact", "--extra", f"{old_dir}=v2-copy",
                     "--tools", str(DATA / "tools.jsonl"),
                     "--requests", str(DATA / "requests_train.jsonl"),
                     "--report-md", str(md_path), "--report-json", str(json_path)])
    assert code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["keys"] == ["old", "new", "v2-copy"]
    text = md_path.read_text(encoding="utf-8")
    assert "| v2-exact |" in text and "| v2-copy |" in text
    assert "## 6. multi 的标注情况" in text
