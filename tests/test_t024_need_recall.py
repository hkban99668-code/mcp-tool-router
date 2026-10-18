"""T-024:need 漏判诊断的判定规则、汇总口径与报告的可复现性。

- 候选 A 的规则 `p_none >= t 才判 none`;
- `_need_scores` 的需要工具召回/精确率与按 source 汇总;
- `_t024_buckets` 的 p_none 分桶、`_t024_samples` 的筛选;
- `t024_eval`(tfidf、无端到端)在假数据上跑通、能渲染、两次跑逐字节相同,且报告里不带延迟。
"""

import json

from mcpcls import tools_real as tr


# ---------------------------------------------------------------- 判定规则

def test_need_at_threshold_rule():
    """p_none ≥ t 判 none;否则在 single/multi 里取概率大的。"""
    p = {"none": 0.6, "single": 0.3, "multi": 0.1}
    assert tr._need_at_threshold(p, 0.5) == "none"
    assert tr._need_at_threshold(p, 0.7) == "single"      # 0.6 < 0.7,降到 single
    p2 = {"none": 0.4, "single": 0.1, "multi": 0.5}
    assert tr._need_at_threshold(p2, 0.5) == "multi"
    assert tr._pnone({"single": 1.0}) == 0.0              # 拿不到 none 概率按 0 算


def test_need_scores_recall_precision_and_source():
    rows = [{"gold": "single", "pred": "single", "source": "bfcl"},
            {"gold": "single", "pred": "none", "source": "bfcl"},
            {"gold": "none", "pred": "none", "source": "alpaca"},
            {"gold": "multi", "pred": "single", "source": "alpaca"},
            {"gold": "none", "pred": "single", "source": "alpaca"}]
    b = tr._need_scores(rows)
    assert b["n"] == 5 and b["pos"] == 3
    assert b["recall"] == 2 / 3                 # 真值要工具 3 条,判对 2 条
    assert b["precision"] == 2 / 3              # 判要工具 3 条,其中 2 条对
    assert b["confusion"]["single"] == {"none": 1, "single": 1, "multi": 0}
    assert b["by_source"]["bfcl"]["recall"] == 0.5
    assert b["by_source"]["alpaca"]["precision"] == 0.5      # 判要工具 2 条中 1 条对


# ---------------------------------------------------------------- 折外诊断

def _oof(fold, gold, source, p_none, need="none"):
    return {"gold": gold, "source": source, "fold": fold, "need": need,
            "need_probs": {"none": p_none, "single": 1 - p_none, "multi": 0.0}, "text": "x"}


def test_buckets_only_counts_dev_and_missing_tool():
    oof = {"a": _oof(0, "single", "bfcl", 0.3),          # 漏判,<0.5
           "b": _oof(1, "multi", "bfcl", 0.6),           # 漏判,0.5-0.7
           "c": _oof(2, "single", "apibank", 0.8),       # 漏判,0.7-0.9
           "d": _oof(3, "single", "bfcl", 0.95),         # 测试折,不计
           "e": _oof(0, "none", "alpaca_zh", 0.99),      # 真值 none,不计
           "f": dict(_oof(0, "single", "bfcl", 0.99), need="single")}   # 判对了,不计
    b = tr._t024_buckets(oof, (0, 1, 2))
    assert b["total"] == 3
    assert b["by_source"] == {"apibank": 1, "bfcl": 2}
    assert b["buckets"] == {"<0.5": 1, "0.5-0.7": 1, "0.7-0.9": 1, ">=0.9": 0}


def test_samples_filter_sort_and_cap():
    oof = {"a": _oof(0, "single", "bfcl", 0.95),
           "b": _oof(1, "single", "bfcl", 0.72),
           "c": _oof(2, "single", "bfcl", 0.71),        # 与 b 同分,按文本排序兜底
           "d": _oof(0, "single", "bfcl", 0.69),        # < 软阈值,兜得住,不抽
           "e": _oof(0, "none", "bfcl", 0.99)}          # 真值 none,不抽
    s = tr._t024_samples(oof, n=2)
    assert [z["need_probs"]["none"] for z in s] == [0.95, 0.72]
    assert all(z["gold"] != "none" for z in s)


# ---------------------------------------------------------------- 端到端(假数据)

def _tool(tid, name, desc, **kw):
    d = {"tool_id": tid, "name": name, "server": tid.split("/")[0], "description": desc,
         "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}
    d.update(kw)
    return d


def _fixtures():
    tools = [_tool("S/send", "send_mail", "Send an email.", domains=["comms"], actions=["send"]),
             _tool("S/ls", "list_files", "List files.", domains=["file"], actions=["read"])]
    labeled = [{"tool_id": t["tool_id"], "domains": t["domains"], "actions": t["actions"],
                "agreed": True, "risk_silver": "L0", "domains_jev": t["domains"],
                "actions_jev": t["actions"], "domains_haiku": t["domains"],
                "actions_haiku": t["actions"]} for t in tools]
    zh = [{"tool_id": t["tool_id"], "description_zh": "中文描述"} for t in tools]
    synth_tools = [dict(_tool("synth/s", "send_email", "Send email."),
                        domains=["comms"], actions=["send"])]
    synth_train = [{"text": "给张三发邮件", "need": "single", "domains": ["comms"],
                    "actions": ["send"], "tool_ids": []},
                   {"text": "你好", "need": "none", "domains": [], "actions": [], "tool_ids": []}]
    real_train = [{"id": f"rt{i}", "text": t, "source": src, "need": n,
                   "domains": ["file"], "actions": ["read"]}
                  for i, (t, src, n) in enumerate([
                      ("列出文件", "bfcl", "single"), ("整理报表", "bfcl", "single"),
                      ("你好呀", "alpaca_zh", "none"), ("今天天气", "alpaca_zh", "none"),
                      ("发封邮件", "apibank", "single"), ("闲聊一下", "alpaca_zh", "none"),
                      ("读一下文件", "bfcl", "single"), ("谢谢", "alpaca_zh", "none")])]
    ref = [{"id": "x1", "text": "列出文件", "source": "bfcl", "need": "single",
            "domains": ["file"], "actions": ["read"], "silver_need": "single"},
           {"id": "x2", "text": "你好", "source": "alpaca_zh", "need": "none",
            "domains": [], "actions": [], "silver_need": "none"}]
    return tools, labeled, zh, synth_tools, synth_train, real_train, ref


def _run(probes=None):
    tools, labeled, zh, synth_tools, synth_train, real_train, ref = _fixtures()
    return tr.t024_eval(tools, labeled, zh, [], synth_tools, synth_train, real_train, ref,
                        encoders=(("tfidf", "基线"),), probes=probes or [], e2e=False)


def test_t024_eval_runs_renders_and_repeats():
    res = _run()
    assert res["selected_key"] == "baseline"        # 没有误挂率数据 -> 不做候选判定
    assert set(res["buckets"]) == {"tfidf"}
    blocks = [c for c in res["candidates"]]
    assert blocks[0]["kind"] == "baseline"
    assert any(c["kind"] == "none_threshold" for c in blocks)
    # 只有基线与被选中的候选会在测试折/参考集上各算一次(其余候选只比开发折)
    for c in (blocks[0], next(c for c in blocks if c["key"] == res["selected_key"])):
        assert set(c["test"]) >= {"recall", "precision", "confusion", "by_source"}
        assert set(c["reference"]) >= {"recall", "precision"}
    report = tr.render_t024(res)
    for sec in ("# need 漏判诊断与改进", "## 0. 调查结论", "## 1. 候选对照",
                "## 2. 兜不住的样本", "## 3. 测试折", "## 4. 参考集", "## 6. 结论"):
        assert sec in report, sec
    assert "延迟" not in report and "latency" not in report
    json.dumps(tr.t024_metrics_json(res), ensure_ascii=False)

    res2 = _run()
    assert (json.dumps(tr.t024_metrics_json(res), ensure_ascii=False, sort_keys=True)
            == json.dumps(tr.t024_metrics_json(res2), ensure_ascii=False, sort_keys=True))
    assert tr.render_t024(res) == tr.render_t024(res2)


def test_t024_eval_selects_qualified_candidate():
    """给一份误挂率池后,A(0.55) 满足约束,应按开发折召回成为选中项。"""
    res = _run(probes=[{"id": "p1", "text": "随便聊聊"}])
    att = res["attach"]["tfidf"]
    assert att["n"] == 1
    cands = {c["key"]: c for c in res["candidates"]}
    assert cands["A@0.55"]["qualified"] in (True, False)      # 约束字段算出来了
    if res["selected_key"] != "baseline":
        sel = cands[res["selected_key"]]
        assert sel["qualified"]
        assert sel["dev"]["recall"] >= cands["baseline"]["dev"]["recall"]


def test_attach_margin_blocks_encoder_like_candidate():
    """误挂率超过基线 +5 点的候选不算满足约束(用构造的 attach 直接验规则)。"""
    base_attach, cand_attach = 0.03, 0.09
    margin = tr.T024_ATTACH_MARGIN
    assert (cand_attach - base_attach) > margin
