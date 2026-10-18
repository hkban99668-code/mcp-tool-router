"""T-022:D-005 参考集的加载与评估口径。

- 请求参考集:按 `id` 对回请求池取正文,两路 need 一致时给 `silver_need`,非法类别报错。
- 风险参考集:真值取 `gu_risk`,为空回退 `claude_risk`。
- 混淆矩阵与两个风险计数用已知 gold/pred 精确核对;`reference_eval` 在假数据上跑通、
  能渲染、两次跑逐字节相同。
"""

import csv
import json

import pytest

from mcpcls import realdata as rd
from mcpcls import tools_real as tr


def _write_csv(path, cols, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------- 加载

def test_load_request_reference_maps_pool(tmp_path):
    """标签读 CSV,正文/来源按 id 对回请求池;两路不一致时 silver_need 为空。"""
    pool = _write_jsonl(tmp_path / "pool_zh.jsonl", [
        {"id": "a-1", "text": "帮我订酒店", "text_orig": "book a hotel", "source": "apibank"},
        {"id": "b-2", "text": "你好呀", "text_orig": "hi", "source": "bfcl"}])
    csvp = _write_csv(tmp_path / "ref.csv",
                      ["id", "need", "domains", "actions", "labeler1_need", "labeler2_need"],
                      [{"id": "a-1", "need": "single", "domains": "business", "actions": "send",
                        "labeler1_need": "single", "labeler2_need": "single"},
                       {"id": "b-2", "need": "none", "domains": "", "actions": "",
                        "labeler1_need": "none", "labeler2_need": "single"}])
    rows = rd.load_request_reference(csvp, pool)
    assert [r["id"] for r in rows] == ["a-1", "b-2"]
    assert rows[0]["text"] == "帮我订酒店" and rows[0]["text_orig"] == "book a hotel"
    assert rows[0]["source"] == "apibank"
    assert rows[0]["domains"] == ["business"] and rows[0]["actions"] == ["send"]
    assert rows[0]["silver_need"] == "single"       # 两路一致
    assert rows[1]["silver_need"] is None            # 两路不一致


def test_load_request_reference_rejects_bad_need(tmp_path):
    """need 不在 none/single/multi 里时按行号报错。"""
    pool = _write_jsonl(tmp_path / "pool_zh.jsonl",
                        [{"id": "a-1", "text": "x", "text_orig": "x", "source": "bfcl"}])
    csvp = _write_csv(tmp_path / "ref.csv", ["id", "need", "domains", "actions"],
                      [{"id": "a-1", "need": "maybe", "domains": "", "actions": ""}])
    with pytest.raises(ValueError, match="need"):
        rd.load_request_reference(csvp, pool)


def test_load_risk_reference_gu_fallback(tmp_path):
    """真值取 gu_risk;为空回退 claude_risk,并记录真值来源。"""
    csvp = _write_csv(tmp_path / "risk.csv",
                      ["tool_id", "gu_risk", "claude_risk", "risk_silver", "risk_rule", "risk_head"],
                      [{"tool_id": "s/a", "gu_risk": "", "claude_risk": "L2",
                        "risk_silver": "L3", "risk_rule": "L0", "risk_head": "L2"},
                       {"tool_id": "s/b", "gu_risk": "L1", "claude_risk": "L3",
                        "risk_silver": "L3", "risk_rule": "L0", "risk_head": "L0"}])
    rows = tr.load_risk_reference(csvp)
    assert rows[0]["truth"] == "L2" and rows[0]["truth_source"] == "claude"
    assert rows[1]["truth"] == "L1" and rows[1]["truth_source"] == "gu"
    assert rows[1]["csv_risk_rule"] == "L0"          # 旧列原样带回,不参与打分


# ---------------------------------------------------------------- 混淆矩阵 / 计数

def test_need_block_confusion_and_recall():
    gold = ["none", "none", "single", "single", "multi"]
    pred = ["none", "single", "single", "none", "single"]
    b = tr._need_block(gold, pred)
    assert b["confusion"]["none"] == {"none": 1, "single": 1, "multi": 0}
    assert b["confusion"]["single"] == {"none": 1, "single": 1, "multi": 0}
    assert b["confusion"]["multi"] == {"none": 0, "single": 1, "multi": 0}
    assert b["per_class"]["single"]["recall"] == 0.5
    assert b["per_class"]["none"]["recall"] == 0.5
    assert b["per_class"]["multi"]["recall"] == 0.0
    assert b["per_class"]["multi"]["precision"] is None    # 没有 multi 预测,精确率无定义
    assert b["need_tool_recall"] == pytest.approx(2 / 3)   # 单+多 3 条,命中 2
    assert b["need_tool_precision"] == pytest.approx(2 / 3)


def test_risk_block_under_and_over_counts():
    gold = ["L3", "L3", "L3", "L0", "L2", "L1"]
    pred = ["L3", "L2", "L0", "L3", "L3", "L1"]
    b = tr._risk_block(gold, pred)
    assert b["l3_total"] == 3
    assert b["l3_under"] == 2       # L3->L2, L3->L0(漏确认)
    assert b["low_over"] == 2       # L0->L3, L2->L3(多打扰)
    assert b["confusion"]["L3"]["L0"] == 1 and b["confusion"]["L3"]["L2"] == 1
    assert b["accuracy"] == pytest.approx(2 / 6)


# ---------------------------------------------------------------- 能力域两个口径

def test_domain_block_argmax_when_pred_domains_empty():
    """预测 `domains` 为空时,主口径仍从 `domain_probs` 取最大域判对;旧口径全判错。"""
    refs = [{"domains": ["comms", "file"]}, {"domains": ["file"]}]
    preds = [("single", [], []), ("single", [], [])]     # 两行预测 domains 都为空
    probs = [{"comms": 0.30, "file": 0.10}, {"file": 0.40, "comms": 0.05}]
    b = tr._domain_block(refs, preds, probs)
    assert b["n"] == 2
    assert b["empty_pred"] == 2
    assert b["argmax_accuracy"] == 1.0     # 空 domains,但最大概率域落在参考域里
    assert b["primary_accuracy"] == 0.0    # 旧口径取 domains 第一个(空)→ 全错


def test_argmax_domain_empty_probs():
    """`domain_probs` 全空时 argmax 记空串;非空时取概率最大者。"""
    assert tr._argmax_domain({}) == ""
    assert tr._argmax_domain({"comms": 0.30, "file": 0.10}) == "comms"


# ---------------------------------------------------------------- 端到端(假数据)

def _tool(tid, name, desc, **kw):
    d = {"tool_id": tid, "name": name, "server": tid.split("/")[0], "description": desc,
         "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}
    d.update(kw)
    return d


def _fixtures():
    tools = [_tool("S/send", "send_mail", "Send an email.", domains=["comms"], actions=["send"]),
             _tool("S/ls", "list_files", "List files.", domains=["file"], actions=["read"]),
             _tool("S/x", "get_x", "Read something.", domains=["file"], actions=["read"])]
    labeled = [
        {"tool_id": "S/send", "domains": ["comms"], "actions": ["send"], "agreed": True,
         "risk_silver": "L3", "domains_jev": ["comms"], "actions_jev": ["send"],
         "domains_haiku": ["comms"], "actions_haiku": ["send"]},
        {"tool_id": "S/ls", "domains": ["file"], "actions": ["read"], "agreed": True,
         "risk_silver": "L0", "domains_jev": ["file"], "actions_jev": ["read"],
         "domains_haiku": ["file"], "actions_haiku": ["read"]},
        {"tool_id": "S/x", "domains": ["file"], "actions": ["read"], "agreed": False,
         "risk_silver": "L0", "domains_jev": ["file"], "actions_jev": ["read"],
         "domains_haiku": ["file"], "actions_haiku": ["delete"]}]
    zh = [{"tool_id": t["tool_id"], "description_zh": "中文描述"} for t in tools]
    synth_tools = [dict(_tool("synth/s", "send_email", "Send email."),
                        domains=["comms"], actions=["send"]),
                   dict(_tool("synth/l", "list_files", "List files."),
                        domains=["file"], actions=["read"])]
    synth_train = [{"text": "给张三发邮件", "need": "single", "domains": ["comms"],
                    "actions": ["send"], "tool_ids": []},
                   {"text": "列出文件", "need": "single", "domains": ["file"],
                    "actions": ["read"], "tool_ids": []},
                   {"text": "你好", "need": "none", "domains": [], "actions": [], "tool_ids": []}]
    real_train = [{"id": "rt1", "text": "整理报表", "need": "single", "domains": ["file"],
                   "actions": ["read"]},
                  {"id": "rt2", "text": "闲聊", "need": "none", "domains": [], "actions": []}]
    req_ref = [{"id": "rt1", "text": "整理报表", "source": "bfcl", "need": "single",
                "domains": ["file"], "actions": ["read"], "silver_need": "single"},
               {"id": "q2", "text": "写一篇博客", "source": "apibank", "need": "none",
                "domains": [], "actions": [], "silver_need": "none"},
               {"id": "q3", "text": "同时发邮件并列文件", "source": "bfcl", "need": "multi",
                "domains": ["comms", "file"], "actions": ["send", "read"], "silver_need": None}]
    risk_ref = [{"tool_id": "S/send", "truth": "L3", "truth_source": "claude",
                 "csv_risk_silver": "L3", "csv_risk_rule": "L3", "csv_risk_head": "L3"},
                {"tool_id": "S/ls", "truth": "L0", "truth_source": "gu",
                 "csv_risk_silver": "L0", "csv_risk_rule": "L0", "csv_risk_head": "L0"},
                {"tool_id": "S/x", "truth": "L2", "truth_source": "claude",
                 "csv_risk_silver": "L0", "csv_risk_rule": "L0", "csv_risk_head": "L0"}]
    return tools, labeled, zh, synth_tools, synth_train, real_train, req_ref, risk_ref


def test_reference_eval_runs_renders_and_repeats():
    fx = _fixtures()
    res = tr.reference_eval(*fx)
    assert res["request"]["n"] == 3
    assert res["request"]["dropped_train"] == 1          # rt1 与参考集同 id
    assert res["request"]["dropped_ids"] == ["rt1"]
    dom = res["request"]["domain"]
    assert dom["n"] == 2 and set(dom) >= {"primary_accuracy", "argmax_accuracy", "empty_pred"}
    blocks = [res["request"]["overall"]] + list(res["request"]["by_source"].values())
    for b in blocks:
        assert sum(sum(row.values()) for row in b["confusion"].values()) == b["n"]
    assert res["risk"]["n"] == 3
    assert set(res["risk"]["paths"]) == {"rule", "head", "silver"}
    for key in ("rule", "head", "silver"):
        b = res["risk"]["paths"][key]
        assert sum(sum(row.values()) for row in b["confusion"].values()) == b["n"]
        assert b["l3_under"] + b["low_over"] <= b["n"]
    report = tr.render_reference_eval(res)
    for sec in ("# 用定稿参考集评估当前默认配置", "## 1. 请求 need(整体)",
                "## 2. 请求 need(按 source)", "## 3. 能力域 / 动作",
                "## 4. 银标对照", "## 5. 风险侧"):
        assert sec in report, sec
    json.dumps(tr.reference_metrics_json(res), ensure_ascii=False)   # 可序列化

    res2 = tr.reference_eval(*fx)
    assert (json.dumps(res, ensure_ascii=False, sort_keys=True)
            == json.dumps(res2, ensure_ascii=False, sort_keys=True))
    assert tr.render_reference_eval(res) == tr.render_reference_eval(res2)
