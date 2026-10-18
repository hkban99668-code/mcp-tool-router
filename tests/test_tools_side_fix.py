"""T-011 工具侧修复的回归测试。

A2 开发/测试划分;A3 翻译可缓存;A4 rules.py 只加词;A5 catalog.py 默认行为不变。
"""

import json
import pathlib
from collections import Counter

import pytest

import mcpcls.compare as compare
from mcpcls import evaluate as ev
from mcpcls import realdata as rd
from mcpcls import tools_real as tr
from mcpcls.catalog import ToolCatalog
from mcpcls.jev import FakeJevClient
from mcpcls.rules import PARAM_ACTIONS, VERB_ACTIONS
from mcpcls.taxonomy import ACTION_IDS, DOMAIN_IDS

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- A2 划分

def test_request_dev_test_split_is_disjoint():
    """请求按折分,0-2 是开发,3-4 是测试,两份无交集。"""
    ids = [f"id_{i}" for i in range(100)]
    dev, test, fold_of = tr._dev_test_ids(ids, folds=5, seed=0)
    assert not (dev & test)
    assert dev | test == set(ids)
    assert all(fold_of[i] in (0, 1, 2) for i in dev)
    assert all(fold_of[i] in (3, 4) for i in test)


def test_tool_dev_test_split_by_function_name_is_disjoint():
    """工具按函数名分组 6:4 切分,开发/测试无交集,同一函数名不跨两边。"""
    tools = [{"tool_id": f"misc/f{i}", "name": f"f{i % 10}", "server": "misc"} for i in range(50)]
    dev, test = tr.tool_dev_test_split(tools, by="name")
    assert dev and test
    assert not (dev & test)
    assert dev | test == {t["tool_id"] for t in tools}
    for name in {t["name"] for t in tools}:
        ids = {t["tool_id"] for t in tools if t["name"] == name}
        assert ids <= dev or ids <= test


def test_tool_dev_test_split_by_server_is_disjoint():
    """工具按 server 分组 6:4 切分;同一 server 整体进同一边,misc 仍按函数名分组。"""
    tools = ([{"tool_id": f"S{i}/a{i}", "name": f"a{i}", "server": f"S{i}"} for i in range(20)]
             + [{"tool_id": f"misc/m{i}", "name": f"m{i % 5}", "server": "misc"} for i in range(20)])
    dev, test = tr.tool_dev_test_split(tools, by="server")
    assert dev and test
    assert not (dev & test)
    assert dev | test == {t["tool_id"] for t in tools}
    for server in {f"S{i}" for i in range(20)}:
        ids = {t["tool_id"] for t in tools if t["server"] == server}
        assert ids <= dev or ids <= test
    # misc 里同一个函数名也整块进同一边
    for name in {f"m{i}" for i in range(5)}:
        ids = {t["tool_id"] for t in tools if t["server"] == "misc" and t["name"] == name}
        assert ids <= dev or ids <= test


# ---------------------------------------------------------------- A3 翻译缓存

def _tool_answers_zh():
    return {"translation": {"type": "text", "text": "中文描述。"}}


def test_translate_tools_second_run_is_cached(monkeypatch, tmp_path):
    """第二次不加 --live 调用 0 次,输出逐字节相同。"""
    monkeypatch.setattr(compare, "CACHE_DIR", tmp_path / "cache")
    tools = [{"tool_id": f"misc/t{i}", "description": f"Tool number {i}."} for i in range(3)]
    client = FakeJevClient(lambda state, questions: _tool_answers_zh())

    out1 = rd.translate_tools(tools, client, "haiku", live=True, workers=1)
    assert out1["stats"].asked == 3
    assert len(out1["rows"]) == 3

    client.calls.clear()
    out2 = rd.translate_tools(tools, client, "haiku", live=False, workers=1)
    assert out2["stats"].asked == 0
    assert len(out2["rows"]) == 3
    assert json.dumps(out1["rows"], ensure_ascii=False, sort_keys=True) == \
           json.dumps(out2["rows"], ensure_ascii=False, sort_keys=True)


def test_translate_tools_empty_description_is_preserved():
    """空描述的条目写空串,不计入失败。"""
    tools = [{"tool_id": "misc/t1", "description": ""}, {"tool_id": "misc/t2", "description": "x"}]
    client = FakeJevClient(lambda state, questions: _tool_answers_zh())
    out = rd.translate_tools(tools, client, "haiku", live=True, workers=1)
    assert out["rows"][0]["description_zh"] == ""
    assert out["rows"][1]["description_zh"] == "中文描述。"


# ---------------------------------------------------------------- A4 rules.py 只加词

def test_rules_verbs_added_not_removed():
    """T-011 只加动作动词,没有删原有词;服务名/名词不在 send 里。"""
    # 这些词是 T-011 补漏加的(都出现在开发份"银标 L3 但旧规则判低"的函数名里)
    added_send = {"buy", "order", "reserve", "book", "booking"}
    added_execute = {"play"}
    assert added_send <= set(VERB_ACTIONS["send"])
    assert added_execute <= set(VERB_ACTIONS["execute"])
    # 第 2 轮删掉的 5 个服务名 / 名词不得再出现在 send 里
    assert not ({"hotel", "hotels", "ride", "uber", "eat"} & set(VERB_ACTIONS["send"]))
    # 没有删除原有核心词
    assert "send" in VERB_ACTIONS["send"]
    assert "run" in VERB_ACTIONS["execute"]
    # PARAM_ACTIONS 没动
    assert set(PARAM_ACTIONS) == {"execute", "send", "delete"}


# ---------------------------------------------------------------- A5 catalog 默认行为

def test_catalog_retrieve_default_params_unchanged():
    """不传新参数时 ToolCatalog.retrieve 用原来的默认 min_prob/prior_weight/action_bonus。"""
    from mcpcls.evaluate import fit_encoder, train_tool_head

    tools = ev.load_jsonl(ROOT / "data" / "tools.jsonl")
    train = ev.load_jsonl(ROOT / "data" / "requests_train.jsonl")
    enc = fit_encoder("tfidf", tools, train)
    head = train_tool_head(enc, tools)
    cat = ToolCatalog(head)
    cat.add_tools(tools, use_labels=True)
    # 默认参数调用不应抛出,且返回列表
    res = cat.retrieve("query")
    assert isinstance(res, list)


# ---------------------------------------------------------------- 双语索引(T-011)

def test_bilingual_index_changes_description_not_head_input():
    """双语索引只改 record.description 并重新编码,不改动工具侧分类头输入。"""
    tools = [{"tool_id": "misc/t1", "name": "t1", "description": "English.",
              "server": "misc", "inputSchema": {"type": "object", "properties": {}}},
             {"tool_id": "misc/t2", "name": "t2", "description": "English two.",
              "server": "misc", "inputSchema": {"type": "object", "properties": {}}}]
    labeled_tools = [{**t, "domains": ["system_misc"], "actions": ["read"]} for t in tools]
    zh_by_id = {"misc/t1": "中文一", "misc/t2": ""}

    _enc, _head, cat = tr.build_fold_index(labeled_tools, labeled_tools, [],
                                           bilingual=True, zh_by_id=zh_by_id)
    # t1 的描述变成双语,t2 保持原样
    by_id = {r.tool_id: r for r in cat.records}
    assert "中文一" in by_id["misc/t1"].description
    assert "中文一" not in by_id["misc/t2"].description


# ---------------------------------------------------------------- T-011 报告生成

def _fake_summary(r5_zh, r5_en):
    """一个假 summary:只填 render_t011 会读到的 off/strict/name 单元。"""
    cells = {}
    for lang, val in (("zh", r5_zh), ("en", r5_en)):
        for suf in ("strict", "name"):
            cells[(lang, "off", suf)] = {"r1": val / 2, "r5": val}
    return {"cells": cells, "diag": {"cand_count": 100.0, "gold_filtered_out": 0.0}, "n": 10}


def _fake_rule(n=100, l3_n=10, rec_old=0.2, rec_new=0.5, acc_old=0.8, acc_new=0.85,
               under_old=0.1, under_new=0.05):
    return {"n": n, "l3_n": l3_n, "accuracy_old": acc_old, "accuracy_new": acc_new,
            "l3_recall_old": rec_old, "l3_recall_new": rec_new,
            "under_old": under_old, "under_new": under_new}


def _fake_t011_res():
    """一份够 render_t011 走完全部小节的假结果。"""
    ids = [f"q{i}" for i in range(10)]
    fold_of = tr.assign_folds(ids, folds=5, seed=0)
    per_query = {i: {"zh": {"off": {"r5": 0.5}}, "en": {"off": {"r5": 0.6}}} for i in ids}
    grid = {
        "grid": [
            {"filter_on": False, "min_prob": 0.1, "prior_weight": 0.3, "zh_strict_r5": 0.5,
             "diag": {"cand_count": 100.0, "gold_filtered_out": 0.0}},
            {"filter_on": True, "min_prob": 0.1, "prior_weight": 0.0, "zh_strict_r5": 0.4,
             "diag": {"cand_count": 50.0, "gold_filtered_out": 0.1}},
        ],
        "best": (False, 0.1, 0.3), "best_score": 0.5,
        "test_rows": [
            {"label": "默认参数", "cfg": (True, 0.1, 0.3), "zh_strict_r5": 0.3,
             "en_strict_r5": 0.5, "zh_name_r5": 0.4, "en_name_r5": 0.6,
             "diag": {"cand_count": 150.0, "gold_filtered_out": 0.13}},
            {"label": "开发集最优", "cfg": (False, 0.1, 0.3), "zh_strict_r5": 0.43,
             "en_strict_r5": 0.6, "zh_name_r5": 0.55, "en_name_r5": 0.7,
             "diag": {"cand_count": 240.0, "gold_filtered_out": 0.0}},
            {"label": "过滤关", "cfg": (False, 0.1, 0.0), "zh_strict_r5": 0.46,
             "en_strict_r5": 0.63, "zh_name_r5": 0.58, "en_name_r5": 0.76,
             "diag": {"cand_count": 240.0, "gold_filtered_out": 0.0}},
        ],
    }
    risk = {mode: {
        "silver_n": 100,
        "head_f1": {"domain": {"micro_f1": 0.8, "macro_f1": 0.7},
                    "action": {"micro_f1": 0.9, "macro_f1": 0.85}},
        "paths": {f"a 工具侧分类头({mode} 训练)": {"l3_recall": 0.7, "under_rate": 0.05}},
    } for mode in ("synth", "real", "both")}
    return {
        "n_tools": 100, "n_requests": len(ids), "untranslated": 0, "fold_of": fold_of,
        "mono": {"summary": {tr.T011_OOF: _fake_summary(0.1, 0.65)},
                 "variants": {tr.T011_OOF: per_query}},
        "bi": {"summary": {tr.T011_OOF: _fake_summary(0.48, 0.64)},
               "variants": {tr.T011_OOF: per_query}},
        "grid": grid, "risk": risk,
        "rule": {"by_name": {"开发份": _fake_rule(), "测试份": _fake_rule()},
                 "by_server": {"开发份": _fake_rule(), "测试份": _fake_rule(rec_new=0.22)}},
        "rule_round1": {"by_name": {"开发份": _fake_rule(rec_new=0.6),
                                    "测试份": _fake_rule(rec_new=0.5)},
                        "by_server": {"开发份": _fake_rule(rec_new=0.5),
                                      "测试份": _fake_rule(rec_new=0.22)}},
        "verbs": Counter({"order": 5, "book": 4, "play": 2}),
        "n_low": 20,
    }


def test_render_t011_runs_on_fake_data():
    """用 Fake 数据跑通报告生成函数,关键小节和数字都出现。"""
    report = tr.render_t011(_fake_t011_res())
    for section in ("# 真实工具目录修复评估(T-011)", "## 1. 双语工具索引", "## 2. 软过滤选参",
                    "## 3. 风险规则补漏", "## 4. 用真实数据训练工具侧分类头",
                    "## 5. 建议设成默认的改动"):
        assert section in report
    # 双语索引:改成「600 条上」,并补测试折各自的数字
    assert "600 条上中文严格 R@5" in report
    assert "测试折:第 3 折" in report and "第 4 折" in report
    # 3.1 删词说明 + 11 词 / 6 词对照
    assert "第 2 轮删掉的词" in report and "过拟合" in report
    # 3.2 两种划分各一张表
    assert "按函数名分组 6:4" in report and "按 server 分组 6:4" in report
    # 2.2 过滤关那行的「金标被过滤」写成 -
    line = next(l for l in report.splitlines() if l.startswith("| 过滤关(关) |"))
    assert line.rstrip().endswith("| - |")


def test_t011_metrics_json_is_jsonable():
    """机器可读版把 tuple 键转成字符串,可直接 json.dump。"""
    data = tr.t011_metrics_json(_fake_t011_res())
    json.dumps(data, ensure_ascii=False)
    assert any(k.startswith("zh/off/") for k in data["mono_summary"][tr.T011_OOF]["cells"])


def test_open_client_or_cache_falls_back_offline(monkeypatch):
    """只读缓存时,客户端造不出来也不该退出,退回占位客户端。"""
    import mcpcls.realdata as rd

    def boom(name, timeout=30.0, model=None):
        raise rd.Stopped("没有 key", rd.EXIT_CLIENT)

    monkeypatch.setattr(rd, "open_client", boom)
    client = rd.open_client_or_cache("haiku", live=False)
    assert client.cache_name == "haiku"
    with pytest.raises(rd.Stopped):
        rd.open_client_or_cache("haiku", live=True)
