"""T-021:当前默认配置的端到端快照。

- A2 报告头部的配置行由 `build_real_router` 的默认参数生成:用 `inspect` 独立读一遍默认参数,
    和 `snapshot_signature()` 逐项核对,再确认已落盘的报告里就是这一行。
- 快照行定义:`current` 必须等于 `cli train` 的 `--real-train` 默认 + `SOFT_NEED_DEFAULT`;
    `t012` 是同一份请求头但软 need 关。
- `snapshot_eval` 在假数据上跑通:结构齐全、软 need 关的行从不软触发、报告与 json 都出得来。
"""

import inspect
import json
import pathlib

from mcpcls import tools_real as tr

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT_MD = ROOT / "reports" / "eval_current_default.md"


# ---------------------------------------------------------------- A2 配置一致

def test_snapshot_signature_matches_build_real_router_defaults():
    """报告头部那一行必须逐项等于 `build_real_router` 的默认参数。"""
    p = inspect.signature(tr.build_real_router).parameters
    expected = ("build_real_router 默认参数:"
                f"encoder={p['encoder'].default} threshold={p['threshold'].default} "
                f"top_k={p['top_k'].default} use_filter={p['use_filter'].default} "
                f"dedupe_by_name={p['dedupe_by_name'].default} "
                f"retrieval_encoder={p['retrieval_encoder'].default} "
                f"soft_need_threshold={p['soft_need_threshold'].default}")
    assert tr.snapshot_signature() == expected
    assert tr.snapshot_config()["soft_need_threshold"] == tr.SOFT_NEED_DEFAULT
    assert SNAPSHOT_MD.exists(), f"快照报告还没生成:{SNAPSHOT_MD}"
    assert expected in SNAPSHOT_MD.read_text(encoding="utf-8")


def test_snapshot_rows_follow_code_defaults():
    """当前默认行 = cli train 的请求头 + SOFT_NEED_DEFAULT;T-012 行是同一请求头、软 need 关。"""
    by_key = {row[0]: row for row in tr.SNAPSHOT_ROWS}
    cur, old = by_key["current"], by_key["t012"]
    assert str(cur[2]) == str(tr.SNAPSHOT_CODE_TRAIN)
    assert cur[3] == tr.SOFT_NEED_DEFAULT
    assert str(old[2]) == str(tr.SNAPSHOT_CODE_TRAIN)
    assert old[3] is None


# ---------------------------------------------------------------- 假数据

def _tool(tid, name, desc):
    return {"tool_id": tid, "name": name, "server": "Svc", "description": desc,
            "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}


def _real_tools():
    return [_tool("Svc/hi#2", "hi", "Say hi."), _tool("Svc/hello", "hello", "Say hello."),
            _tool("Svc/ls", "list_all", "List everything.")]


def _labeled_rows():
    return [{"tool_id": "Svc/hi#2", "domains": ["system_misc"], "actions": ["read"], "agreed": True},
            {"tool_id": "Svc/hello", "domains": ["system_misc"], "actions": ["read"], "agreed": True},
            {"tool_id": "Svc/ls", "domains": ["file"], "actions": ["read"], "agreed": True}]


def _zh_rows():
    return [{"tool_id": "Svc/hi#2", "description_zh": "打招呼"},
            {"tool_id": "Svc/hello", "description_zh": "问好"},
            {"tool_id": "Svc/ls", "description_zh": "列出全部"}]


def _synth_tools():
    return [dict(_tool("synth/send", "send_email", "Send an email."),
                 domains=["comms"], actions=["send"]),
            dict(_tool("synth/ls", "list_files", "List files."),
                 domains=["file"], actions=["read"])]


def _synth_train():
    return [{"text": "给张三发邮件", "need": "single", "domains": ["comms"], "actions": ["send"],
             "tool_ids": ["synth/send"]},
            {"text": "列出文件", "need": "single", "domains": ["file"], "actions": ["read"],
             "tool_ids": ["synth/ls"]},
            {"text": "你好呀", "need": "none", "domains": [], "actions": [], "tool_ids": []}]


def _real_train():
    return [{"id": f"rt{i}", "text": f"整理第 {i} 份报表", "need": "single" if i % 2 else "none",
             "domains": ["file"], "actions": ["read"], "tool_ids": ["Svc/ls"]} for i in range(8)]


def _requests():
    return [{"id": f"q{i}", "text": f"请求 {i}", "text_orig": f"request {i}",
             "tool_ids": ["Svc/hi#2"] if i % 2 == 0 else ["Svc/hello"],
             "candidate_tool_ids": ["Svc/hi#2", "Svc/hello", "Svc/ls"],
             "need": "single" if i % 3 else "none"} for i in range(10)]


def test_snapshot_eval_structure_and_render():
    """两行(软 need 开 / 关)在假数据上跑通:结构齐全、关的行不软触发、报告能渲染。"""
    train = _real_train()
    rows = (("current", "当前默认", train, 1.01, "x"),
            ("t012", "T-012 默认", train, None, "y"))
    probes = [{"id": "p1", "text": "你好呀"}, {"id": "p2", "text": "写一篇博客"}]
    res = tr.snapshot_eval(_real_tools(), _labeled_rows(), _zh_rows(), _requests(),
                           _synth_tools(), _synth_train(), rows=rows, folds=5, seed=0,
                           probes=probes)
    req_ids = {r["id"] for r in _requests()}
    assert set(res["per"]["current"]) == req_ids
    assert res["rows"][0]["request_train"] == "(内存数据)"
    # 软 need 关的行从不软触发;开的行有触发(阈值 1.01 = 判 none 就附带)
    assert all(not res["per"]["t012"][i]["zh_soft"] for i in req_ids)
    assert any(res["per"]["current"][i]["zh_soft"] for i in req_ids)
    assert res["deploy"]["t012"]["false_attach"] == 0.0
    assert res["deploy"]["current"]["n_probes"] == 2
    assert res["deploy"]["current"]["pkl_bytes"] > 0
    # need 判定与软 need 无关,两行应逐条一致
    assert tr.snapshot_need_metrics(res, "current", res["dev_ids"], "zh") == \
        tr.snapshot_need_metrics(res, "t012", res["dev_ids"], "zh")
    report = tr.render_snapshot(res)
    for section in ("# 当前默认配置的端到端快照", "## 0. 本报告的配置", "## 1. 检索:主口径",
                    "## 2. 检索:上限", "## 3. need 判定", "## 4. 软 need 触发率",
                    "## 5. 延迟", "## 6. top-1", "## 7. `router.pkl` 大小", "## 8. 从 T-012",
                    "## 9. 与既有报告的一致性核对"):
        assert section in report, section
    assert tr.snapshot_signature() in report
    assert "T-012 默认" in report and "当前默认" in report
    json.dumps(tr.snapshot_metrics_json(res), ensure_ascii=False)
