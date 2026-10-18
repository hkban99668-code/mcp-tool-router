"""T-023 两类风险信号(扫描外部主机 / 控制实体设备)的回归测试。

- A2:词表只按开发份选,测试份 / 参考集只报一次 —— 用 `compute_t023` 的 "改前 / 改后"
  计数核对(测试份 L3 召回不下降、低判成 L3 不增加,参考集 L3 判低 2→0)。
- 两个漏判 L3(Crawler.scan、oven_preheat)和开发份同类例子判 L3;只读设备查询不判 L3。
- 合成工具(`data/tools.jsonl`)不受影响,`cli evaluate` 输出不变。
- 报告能生成、可 json 序列化、可复跑逐字节一致。
"""

import json
import pathlib
import re

from mcpcls import evaluate as ev
from mcpcls import tools_real as tr
from mcpcls.rules import (DEVICE_CONTROL_VERBS, DEVICE_NOUNS, RISK_KEYWORD_SOURCES,
                          SCAN_HOST_TOKENS, risk_signals, rule_risk, rule_risk_base,
                          split_identifier, taxonomy_vocabulary)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _tools():
    return {t["tool_id"]: t for t in ev.load_jsonl(DATA / "tools_real.jsonl")}


def _tool(name, desc, **kw):
    d = {"tool_id": f"misc/{name}", "name": name, "server": "misc", "description": desc,
         "inputSchema": {"type": "object", "properties": {}}}
    d.update(kw)
    return d


# ---------------------------------------------------------------- 两个漏判 + 同类例子

def test_missed_l3_tools_now_are_l3():
    """参考集里漏判的两个 L3:Crawler.scan(扫描外部主机)、oven_preheat(操作设备)。"""
    tools = _tools()
    assert rule_risk(tools["Crawler/Crawler.scan"], ["search_web"]) == "L3"
    assert rule_risk(tools["misc/oven_preheat"], ["system_misc"]) == "L3"
    # 改前分别是 L1 / L0
    assert rule_risk_base(tools["Crawler/Crawler.scan"], ["search_web"]) == "L1"
    assert rule_risk_base(tools["misc/oven_preheat"], ["system_misc"]) == "L0"


def test_dev_split_same_class_examples_are_l3():
    """开发份里同类的 3 个例子(扫描)也判 L3;设备类开发份例子是 ControlAppliance。"""
    tools = _tools()
    for tid in ("XSS_Scanner/XSS_Scanner.scan", "network_scanner/network_scanner.scan",
                "misc/nuclei_http"):
        assert rule_risk(tools[tid]) == "L3", tid
    assert rule_risk(tools["ControlAppliance/ControlAppliance.execute"]) == "L3"
    # 测试份的 ThinQ_Connect(控制家电)也由设备信号命中
    assert rule_risk(tools["misc/ThinQ_Connect"]) == "L3"


# ---------------------------------------------------------------- 只读设备查询不升 L3

def test_read_only_device_queries_stay_below_l3():
    """纯读取的设备状态查询不升 L3:名词 + 只读动词不该命中设备信号。"""
    tools = _tools()
    # 真实目录里的读温度 / 读传感器历史
    assert risk_signals(tools["misc/get_temperature"]) == set()
    assert rule_risk(tools["misc/get_temperature"]) != "L3"
    assert rule_risk(tools["misc/get_sensor_readings_history"]) != "L3"
    # 合成几个 list / get 开头、带设备名词的只读工具
    for t in (_tool("list_devices", "List all devices in the account."),
              _tool("get_thermostat_status", "Read the current thermostat status."),
              _tool("list_lights", "List all lights and their on/off state.")):
        assert risk_signals(t) == set(), t["name"]
        assert rule_risk(t) != "L3", t["name"]


def test_device_signal_needs_noun_and_control_verb():
    """设备信号要求名词 + 控制动词同时出现;只有名词或只有动词都不命中。"""
    noun_only = _tool("device_listing", "Show devices.", server="misc")
    verb_only = _tool("turn_on_thing", "Turn on a thing.")
    assert risk_signals(noun_only) == set()
    assert risk_signals(verb_only) == set()          # 没有设备名词
    both = _tool("set_thermostat", "Set the thermostat target temperature.")
    assert risk_signals(both) == {"device_control"}
    assert rule_risk(both) == "L3"


def test_scan_signal_matches_name_or_description():
    """扫描信号命中函数名 token 或描述词,不命中同形的无关词(mapping 不含 ping)。"""
    assert risk_signals(_tool("port_scan", "x")) == {"scan_host"}
    assert risk_signals(_tool("nuclei_http", "Scan a host for HTTP vulnerabilities.")) == {"scan_host"}
    assert risk_signals(_tool("add_mapping", "Add a mapping.")) == set()
    assert risk_signals(_tool("support_export", "Export support data.")) == set()


# ---------------------------------------------------------------- 词来源标注

def test_t023_keyword_sources_match_the_catalog():
    """每个新词都标了来源;train 词出现在真实工具目录里,prior 词不在,taxonomy 词在 taxonomy。"""
    words = set(SCAN_HOST_TOKENS) | set(DEVICE_NOUNS) | set(DEVICE_CONTROL_VERBS)
    assert words == set(RISK_KEYWORD_SOURCES), "新词与来源标注要一一对应"
    catalog = set()
    for t in ev.load_jsonl(DATA / "tools_real.jsonl"):
        catalog |= set(split_identifier(t.get("name", "")))
        catalog |= set(re.findall(r"[a-z0-9]+", (t.get("description") or "").lower()))
    tax = taxonomy_vocabulary()
    for w, src in RISK_KEYWORD_SOURCES.items():
        assert src in ("train", "taxonomy", "prior"), w
        if src == "train":
            assert w in catalog, f"{w}: 标成 train 但工具目录里没有"
        elif src == "taxonomy":
            assert w in tax, f"{w}: 标成 taxonomy 但 taxonomy.py 里没有"
        else:
            assert w not in catalog and w not in tax, f"{w}: 标成 prior 但数据里有"


# ---------------------------------------------------------------- 不影响合成侧

def test_synthetic_tools_unaffected_by_new_signals():
    """合成 53 个工具里没有一个命中两类信号,规则风险逐条等于改前。"""
    for t in ev.load_jsonl(DATA / "tools.jsonl"):
        assert risk_signals(t) == set(), t["name"]
        assert rule_risk(t) == rule_risk_base(t), t["name"]


# ---------------------------------------------------------------- 开发 / 测试指标(A3)

def test_test_split_recall_up_and_over_not_increased():
    """测试份(两种划分)L3 召回不下降、低判成 L3 不增加;参考集 L3 判低 2→0。"""
    tools = ev.load_jsonl(DATA / "tools_real.jsonl")
    rows = ev.load_jsonl(DATA / "tools_real_labeled.jsonl")
    risk_ref = tr.load_risk_reference(DATA / "tools_real_risk_audit_claude.csv")
    res = tr.compute_t023(tools, rows, risk_ref)
    for split_key in ("by_name", "by_server"):
        m = res["rule"][split_key]["测试份"]
        assert m["l3_recall_new"] >= m["l3_recall_old"], split_key
        assert m["over_new"] <= m["over_old"], split_key
        assert m["over_new"] - m["over_old"] <= 10      # A3 上限
    ref = res["reference"]
    assert len(ref["under_old_ids"]) == 2
    assert ref["under_new_ids"] == []
    assert set(ref["under_old_ids"]) == {"Crawler/Crawler.scan", "misc/oven_preheat"}
    assert ref["over_new"] == ref["over_old"]           # 多打扰不增加


# ---------------------------------------------------------------- 报告生成

def _fixtures():
    tools = [
        _tool("scan_host", "Scan an external host.", server="S1"),
        _tool("list_files", "List files.", server="S1"),
        _tool("set_thermostat", "Set the thermostat.", server="S2"),
    ]
    rows = [
        {"tool_id": "misc/scan_host", "domains": ["system_misc"], "agreed": True,
         "risk_silver": "L3"},
        {"tool_id": "misc/list_files", "domains": ["file"], "agreed": True, "risk_silver": "L0"},
        {"tool_id": "misc/set_thermostat", "domains": ["system_misc"], "agreed": True,
         "risk_silver": "L3"},
    ]
    risk_ref = [{"tool_id": "misc/scan_host", "truth": "L3", "truth_source": "claude"},
                {"tool_id": "misc/list_files", "truth": "L0", "truth_source": "gu"},
                {"tool_id": "misc/set_thermostat", "truth": "L3", "truth_source": "claude"}]
    return tools, rows, risk_ref


def test_compute_t023_renders_and_repeats():
    fx = _fixtures()
    res = tr.compute_t023(*fx)
    assert res["n_tools"] == 3
    assert set(res["rule"]) == {"by_name", "by_server"}
    report = tr.render_t023(res)
    for sec in ("# 风险规则补两类信号", "## 0. 参考集", "## 1. 词表",
                "### 1.0 两类工具概览", "### 1.1 宽候选词排序",
                "## 2. 开发份 / 测试份规则指标", "## 3. 结论"):
        assert sec in report, sec
    json.dumps(tr.t023_metrics_json(res), ensure_ascii=False)     # 可序列化

    res2 = tr.compute_t023(*fx)
    assert (json.dumps(res, ensure_ascii=False, sort_keys=True)
            == json.dumps(res2, ensure_ascii=False, sort_keys=True))
    assert tr.render_t023(res) == tr.render_t023(res2)
