"""T-027:能力域预测的检索投票。

- `domainvote` 的三个纯函数 + policy:投票权重、融合、阈值挑域,都不依赖模型;
- `t027_eval` / `render_t027` 用一份小型合成目录跑通全流程:候选齐全、开发折选、测试折与
  参考集只报一次、报告与 json 都出得来。不加载任何语义模型 / 不走 HF 缓存。
"""

import json

import pytest

from mcpcls import domainvote as dv
from mcpcls import evaluate as ev
from mcpcls import tools_real as tr


class _Rec:
    """假的 ToolRecord:投票只看 name / domains。"""

    def __init__(self, name, domains):
        self.name = name
        self.domains = list(domains)


# ---------------------------------------------------------------- 纯函数

def test_vote_probs_rank_weights_and_normalization():
    """rank 权重 = 1/名次,分布归一化;同名次分数大小不影响结果。"""
    hits = [(_Rec("a", ["file"]), 0.9), (_Rec("b", ["comms"]), 0.1),
            (_Rec("c", ["file"]), 0.05)]
    dist = dv.vote_probs(hits, weight="rank")
    assert dist["file"] == pytest.approx((1.0 + 1.0 / 3) / (1.0 + 1.0 / 2 + 1.0 / 3))
    assert dist["comms"] == pytest.approx((1.0 / 2) / (1.0 + 1.0 / 2 + 1.0 / 3))
    assert sum(dist.values()) == pytest.approx(1.0)
    assert set(dist) >= {"file", "comms", "business"}


def test_vote_probs_score_weight_uses_scores():
    hits = [(_Rec("a", ["file"]), 0.8), (_Rec("b", ["comms"]), 0.2)]
    dist = dv.vote_probs(hits, weight="score")
    assert dist["file"] == pytest.approx(0.8)
    assert dist["comms"] == pytest.approx(0.2)


def test_vote_probs_empty_and_unknown_domains():
    assert sum(dv.vote_probs([]).values()) == 0.0
    # 工具域不在 taxonomy 里时被忽略,不报错
    dist = dv.vote_probs([(_Rec("a", ["not_a_domain"]), 1.0)])
    assert sum(dist.values()) == 0.0
    with pytest.raises(ValueError):
        dv.vote_probs([], weight="nope")


def test_fuse_probs_endpoints_and_validation():
    head = {"file": 0.9, "comms": 0.1}
    vote = {"file": 0.0, "comms": 1.0}
    assert dv.fuse_probs(head, vote, 1.0)["file"] == pytest.approx(0.9)
    assert dv.fuse_probs(head, vote, 0.0)["comms"] == pytest.approx(1.0)
    mid = dv.fuse_probs(head, vote, 0.5)
    assert mid["file"] == pytest.approx(0.45) and mid["comms"] == pytest.approx(0.55)
    assert set(mid) == set(dv.DOMAIN_IDS)
    with pytest.raises(ValueError):
        dv.fuse_probs(head, vote, 1.5)


def test_pick_domains_threshold_vs_floor():
    probs = {"file": 0.4, "comms": 0.3}
    assert dv.pick_domains(probs, 0.5, floor_top1=False) == []
    assert dv.pick_domains(probs, 0.5, floor_top1=True) == ["file"]
    assert dv.pick_domains({"file": 0.9, "comms": 0.6}, 0.5) == ["file", "comms"]
    # 分布全 0(没有候选)时 floor 也不能凭空造一个域
    assert dv.pick_domains({"file": 0.0, "comms": 0.0}, 0.5, floor_top1=True) == []


def test_argmax_domain_tie_breaks_by_taxonomy_order():
    assert dv.argmax_domain({"file": 0.5, "comms": 0.5}) == dv.DOMAIN_IDS[0]
    assert dv.argmax_domain({"comms": 0.9, "file": 0.1}) == "comms"
    assert dv.argmax_domain({}) == ""


def test_policy_resolve_fuses_and_picks():
    pol = dv.DomainVotePolicy(k=2, beta=0.5, weight="rank", threshold=0.5, floor_top1=True)
    head = {"file": 0.2, "comms": 0.1}
    hits = [(_Rec("a", ["comms"]), 0.9), (_Rec("b", ["file"]), 0.1), (_Rec("c", ["file"]), 0.0)]
    probs, domains = pol.resolve(head, hits)
    # vote(rank) = comms 1, file 1/2 → 归一 comms 2/3, file 1/3;融合 0.5/0.5
    assert probs["comms"] == pytest.approx(0.5 * 0.1 + 0.5 * (2 / 3))
    # 两个域都没过 0.5,floor_top1=True 时至少给分最高的 comms
    assert domains == ["comms"]
    assert dv.DomainVotePolicy(beta=1.0).resolve(head, hits)[0] == head


# ---------------------------------------------------------------- t027_eval / render(合成)

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
    real = [_tool("Svc/mail#1", "mail_svc", "发邮件 send mail"),
            _tool("Svc/ls#1", "ls_svc", "列目录 list directory"),
            _tool("Svc/cal#1", "cal_svc", "日历 calendar event")]
    labeled = [{"tool_id": "Svc/mail#1", "domains": ["comms"], "actions": ["send"]},
               {"tool_id": "Svc/ls#1", "domains": ["file"], "actions": ["read"]},
               {"tool_id": "Svc/cal#1", "domains": ["calendar_task"], "actions": ["create"]}]
    zh = [{"tool_id": r["tool_id"], "description_zh": f"中文 {r['name']}"} for r in real]
    requests = [{"id": f"q{i}", "text": f"帮我处理第 {i} 件事",
                 "text_orig": f"handle item {i}",
                 "tool_ids": ["Svc/mail#1" if i % 2 else "Svc/ls#1"],
                 "candidate_tool_ids": [r["tool_id"] for r in real],
                 "need": "none" if i % 5 == 4 else "single"} for i in range(10)]
    real_train = [{"id": f"rt{i}", "text": f"整理第 {i} 份报表",
                   "need": "single" if i % 2 else "none", "domains": ["file"],
                   "actions": ["read"], "tool_ids": ["Svc/ls#1"]} for i in range(8)]
    req_ref = [{"id": f"q{i}", "text": f"帮我处理第 {i} 件事", "source": "fake",
                "need": "single" if i % 3 else "none",
                "domains": ["comms"] if i % 2 else ["file"], "actions": ["send"],
                "silver_need": None} for i in range(6)]
    return real, labeled, zh, requests, req_ref, real_train


def test_t027_eval_smoke_and_selection():
    """候选齐全;选的是满足延迟约束里开发折主口径最高的那个;测试折/参考集各一次。"""
    real, labeled, zh, requests, req_ref, real_train = _fake_data()
    res = tr.t027_eval(real, labeled, zh, requests, _synth_tools(), _train(), real_train,
                       req_ref=req_ref)
    keys = [c["key"] for c in res["candidates"]]
    assert keys[0] == "D0"
    assert len(keys) == 1 + len(tr.T027_KS) + len(tr.T027_KS) * len(tr.T027_BETAS)
    for c in res["candidates"]:
        d = res["dev"][c["key"]]
        assert 0.0 <= d["main"] <= 1.0 and d["p95_ms"] <= tr.T027_P95_MAX_MS
    # 选法:满足约束的候选里开发折主口径最高
    best = max((c for c in res["candidates"] if c["key"] != "D0"),
               key=lambda c: res["dev"][c["key"]]["main"])
    assert res["selected_key"] == best["key"]
    assert res["dev_gain"] == pytest.approx(
        res["dev"][res["selected_key"]]["main"] - res["dev"]["D0"]["main"])
    # 测试折与参考集只含基线与主选
    assert set(res["test"]["rows"]) - {"n"} == {"D0", res["selected_key"]}
    assert set(res["reference"]["rows"]) - {"n"} == {"D0", res["selected_key"]}
    assert res["reference"]["n"] == 4            # need != none 的 4 条
    assert len(res["dev_retrieval_sha1"]) == 40


def test_t027_eval_no_e2e_skips_test_and_reference():
    real, labeled, zh, requests, _ref, real_train = _fake_data()
    res = tr.t027_eval(real, labeled, zh, requests, _synth_tools(), _train(), real_train,
                       e2e=False)
    assert "test" not in res and "reference" not in res
    assert res["selected_key"] in [c["key"] for c in res["candidates"]]


def test_render_t027_sections_and_json():
    real, labeled, zh, requests, req_ref, real_train = _fake_data()
    res = tr.t027_eval(real, labeled, zh, requests, _synth_tools(), _train(), real_train,
                       req_ref=req_ref)
    report = tr.render_t027(res)
    for section in ("# 能力域预测:分类头 + 检索结果投票(T-027)", "## 0. 调查结论",
                    "## 1. 候选对照", "## 2. 测试折", "## 3. 参考集", "## 4. 结论"):
        assert section in report, section
    json.dumps(tr.t027_metrics_json(res), ensure_ascii=False)


def test_t027_gold_domains_ordered_union():
    dom_of = {"A/1": ["file", "comms"], "A/2": ["comms", "business"]}
    got = tr._t027_gold_domains({"tool_ids": ["A/1", "A/2"]}, dom_of)
    assert got == ["file", "comms", "business"]      # 保序去重


def test_t027_candidates_count_and_labels():
    cands = tr._t027_candidates()
    kinds = {c["kind"] for c in cands}
    assert kinds == {"head", "vote", "fuse"}
    assert sum(1 for c in cands if c["kind"] == "fuse") == len(tr.T027_KS) * len(tr.T027_BETAS)
    assert [c["k"] for c in cands if c["kind"] == "vote"] == list(tr.T027_KS)
