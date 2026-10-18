"""T-028:能力域投票改用请求级银标重评。

- 去重、主域统计、多数类基线、bootstrap 置信区间都是纯函数,直接测;
- `t028_eval` / `render_t028` 用一份小型合成数据跑通全流程:选参集 5 折折外、留出集与参考集
  各只报一次、报告与 json 都出得来。不加载任何语义模型 / 不走 HF 缓存。
"""

import json

import pytest

from mcpcls import tools_real as tr


# ---------------------------------------------------------------- 纯函数

def test_norm_text_nfkc_whitespace_lower():
    assert tr._norm_text("  给张三 发邮件 ") == "给张三发邮件"
    assert tr._norm_text("ＡＢ\tＣ\n") == "abc"          # 全角 + 空白 + 大小写
    assert tr._norm_text(None) == ""


def test_t028_dedupe_by_id_and_normalized_text():
    ref_ids, ref_texts = {"R1"}, {tr._norm_text("给张三发邮件")}
    rows = [{"id": "R1", "text": "别的话"},                # id 命中
            {"id": "X", "text": " 给张三 发邮件 "},        # 规范化文本命中(空白不同)
            {"id": "Y", "text": "完全无关的话"},
            {"id": "Z", "text": "另一句"}]
    kept, dropped = tr._t028_dedupe(rows, ref_ids, ref_texts)
    assert dropped == ["R1", "X"]
    assert [r["id"] for r in kept] == ["Y", "Z"]


def test_t028_ref_sets_drops_empty_text():
    ids, texts = tr._t028_ref_sets([{"id": "a", "text": "aaa"}, {"id": "b", "text": ""}])
    assert ids == {"a", "b"} and "" not in texts


def test_t028_domain_stats_majority_baseline():
    rows = [{"domains": ["file"]}, {"domains": ["file"]}, {"domains": ["comms"]},
            {"domains": ["comms", "file"]}, {"domains": []}]
    st = tr._t028_domain_stats(rows)
    assert st["n"] == 5 and st["empty"] == 1
    assert st["multi"] == 1 and st["single"] == 3
    # 多数类 file 出现 3 次(file,file,file) → 主口径 3/5;comms 只 2 次
    assert st["majority"] == "file" and st["majority_hits"] == 3
    assert tr._t028_majority_hits(rows, "") == 0


def test_t028_bootstrap_ci_bounds_and_determinism():
    flags = [1] * 8 + [0] * 2
    lo, hi = tr._t028_bootstrap_ci(flags)
    assert lo <= 0.8 <= hi and 0.0 <= lo <= hi <= 1.0
    assert (lo, hi) == tr._t028_bootstrap_ci(flags)          # 固定种子 → 可复现
    assert tr._t028_bootstrap_ci([]) == (None, None)


# ---------------------------------------------------------------- t028_eval / render(合成)

def _tool(tid, name, desc):
    return {"tool_id": tid, "name": name, "server": "Svc", "description": desc,
            "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}


def _synth_tools():
    return [dict(_tool("synth/send", "send_email", "Send an email."), domains=["comms"],
                 actions=["send"]),
            dict(_tool("synth/ls", "list_files", "List files."), domains=["file"],
                 actions=["read"])]


def _train():
    return [{"text": "给张三发邮件", "need": "single", "domains": ["comms"], "actions": ["send"],
             "tool_ids": ["synth/send"]},
            {"text": "列出全部文件", "need": "single", "domains": ["file"], "actions": ["read"],
             "tool_ids": ["synth/ls"]},
            {"text": "你好呀", "need": "none", "domains": [], "actions": [], "tool_ids": []}]


def _fake_data():
    """真实目录 3 个工具 + 20 条选参集 + 10 条留出集 + 6 条参考集。"""
    real = [_tool("Svc/mail#1", "mail_svc", "发邮件 send mail"),
            _tool("Svc/ls#1", "ls_svc", "列目录 list directory"),
            _tool("Svc/cal#1", "cal_svc", "日历 calendar event")]
    labeled = [{"tool_id": "Svc/mail#1", "domains": ["comms"], "actions": ["send"]},
               {"tool_id": "Svc/ls#1", "domains": ["file"], "actions": ["read"]},
               {"tool_id": "Svc/cal#1", "domains": ["calendar_task"], "actions": ["create"]}]
    zh = [{"tool_id": r["tool_id"], "description_zh": f"中文 {r['name']}"} for r in real]
    doms = ["file", "comms", "calendar_task"]
    real_train = [{"id": f"rt{i}", "text": f"整理第 {i} 份报表并归档",
                   "need": "single", "domains": [doms[i % 3]],
                   "actions": ["read"], "tool_ids": ["Svc/ls#1"]} for i in range(20)]
    real_holdout = [{"id": f"hh{i}", "text": f"处理第 {i} 件私事",
                     "need": "single", "domains": [doms[i % 3]],
                     "actions": ["read"], "tool_ids": ["Svc/ls#1"]} for i in range(10)]
    req_ref = [{"id": f"q{i}", "text": f"帮我处理第 {i} 件事", "source": "fake",
                "need": "single" if i % 3 else "none",
                "domains": ["comms"] if i % 2 else ["file"], "actions": ["send"],
                "silver_need": None} for i in range(6)]
    return real, labeled, zh, req_ref, real_train, real_holdout


def test_t028_eval_smoke_selection_and_single_report():
    real, labeled, zh, req_ref, real_train, real_holdout = _fake_data()
    res = tr.t028_eval(real, labeled, zh, _synth_tools(), _train(), real_train, real_holdout,
                       req_ref)
    keys = [c["key"] for c in res["candidates"]]
    assert keys[0] == "D0"
    assert len(keys) == 1 + len(tr.T027_KS) + len(tr.T027_KS) * len(tr.T027_BETAS)
    for c in res["candidates"]:
        d = res["dev"][c["key"]]
        assert 0.0 <= d["main"] <= 1.0 and d["p95_ms"] <= tr.T028_P95_MAX_MS
        assert res["dev_ci"][c["key"]][0] <= d["main"] <= res["dev_ci"][c["key"]][1]
    # 选法:满足延迟约束里选参集主口径最高
    best = max((c for c in res["candidates"] if c["key"] != "D0"),
               key=lambda c: res["dev"][c["key"]]["main"])
    assert res["selected_key"] == best["key"]
    assert res["dev_gain"] == pytest.approx(
        res["dev"][res["selected_key"]]["main"] - res["dev"]["D0"]["main"])
    # 留出集与参考集只含基线与主选
    assert set(res["holdout"]["rows"]) - {"n"} == {"D0", res["selected_key"]}
    assert set(res["reference"]["rows"]) - {"n"} == {"D0", res["selected_key"]}
    assert res["sel_n"] == 20 and res["holdout"]["n"] == 10
    assert res["reference"]["n"] == 4              # need != none 的 4 条
    assert len(res["dev_retrieval_sha1"]) == 40
    assert res["recommend"] in (True, False)
    assert res["stats"]["majority"] in {"file", "comms", "calendar_task"}


def test_t028_eval_dedupes_holdout_against_reference():
    """参考集里的 id/文本也从留出集剔除;选参集无重叠时不受影响。"""
    real, labeled, zh, req_ref, real_train, real_holdout = _fake_data()
    # 让 hh0 与参考集 q0 同 id、hh1 与 q1 同文本(空文本另算)
    real_holdout[0]["id"] = "q0"
    real_holdout[1]["text"] = req_ref[1]["text"]
    res = tr.t028_eval(real, labeled, zh, _synth_tools(), _train(), real_train, real_holdout,
                       req_ref)
    assert res["holdout_dropped"] == ["q0", "hh1"]
    assert res["holdout"]["n"] == 8                # 10 - 2
    assert res["train_dropped"] == []


def test_t028_eval_no_e2e_skips_holdout_and_reference():
    real, labeled, zh, req_ref, real_train, real_holdout = _fake_data()
    res = tr.t028_eval(real, labeled, zh, _synth_tools(), _train(), real_train, real_holdout,
                       req_ref, e2e=False)
    assert "holdout" not in res and "reference" not in res
    assert res["selected_key"] in [c["key"] for c in res["candidates"]]


def test_render_t028_sections_and_json():
    real, labeled, zh, req_ref, real_train, real_holdout = _fake_data()
    res = tr.t028_eval(real, labeled, zh, _synth_tools(), _train(), real_train, real_holdout,
                       req_ref)
    report = tr.render_t028(res)
    for section in ("# 能力域投票重评:改用请求级银标选参(T-028)", "## 0. 调查结论",
                    "## 1. 候选对照", "## 2. 留出集", "## 3. 参考集", "## 4. 结论"):
        assert section in report, section
    json.dumps(tr.t028_metrics_json(res), ensure_ascii=False)
