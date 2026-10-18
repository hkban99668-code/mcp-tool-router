"""T-012:真实工具目录作为线上默认 + 端到端评估的回归测试。

全程不联网、不读仓库里的 2405 个工具:用小号假数据跑通同一条代码路径。
覆盖 A2(synth 行为不变)、A3(use_filter=False 传 None)、A5(e2e 报告)、A6(README 来源)。
"""

import argparse
import json
import pathlib
import pickle

from mcpcls import cli
from mcpcls import evaluate as ev
from mcpcls import tools_real as tr
from mcpcls.heads import HeadOutput
from mcpcls.router import Router

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 小号假数据

def _tool(tid, name, server="Svc", desc="Do the thing.", props=("a",)):
    return {"tool_id": tid, "name": name, "server": server, "description": desc,
            "inputSchema": {"type": "object",
                            "properties": {p: {"type": "string"} for p in props}}}


def _synth_tools():
    rows = [("send_email", "comms", "send"), ("list_files", "file", "read"),
            ("run_script", "system_misc", "execute"), ("get_weather", "weather_geo", "read")]
    return [dict(_tool(f"synth/{n}", n, server="synth", desc=f"{n} tool."),
                 domains=[d], actions=[a]) for n, d, a in rows]


def _synth_requests():
    return [
        {"text": "给张三发一封邮件", "need": "single", "domains": ["comms"], "actions": ["send"],
         "tool_ids": ["synth/send_email"]},
        {"text": "帮我列出我的文件", "need": "single", "domains": ["file"], "actions": ["read"],
         "tool_ids": ["synth/list_files"]},
        {"text": "跑一下这段脚本", "need": "single", "domains": ["system_misc"], "actions": ["execute"],
         "tool_ids": ["synth/run_script"]},
        {"text": "解释一下什么是梯度下降", "need": "none", "domains": [], "actions": [],
         "tool_ids": []},
        {"text": "今天天气怎么样", "need": "single", "domains": ["weather_geo"], "actions": ["read"],
         "tool_ids": ["synth/get_weather"]},
        {"text": "你好呀", "need": "none", "domains": [], "actions": [], "tool_ids": []},
    ]


def _real_tools():
    return [_tool("Svc/hi#2", "hi", desc="Say hi."), _tool("Svc/hello", "hello", desc="Say hello.")]


def _labeled_rows():
    return [{"tool_id": "Svc/hi#2", "domains": ["system_misc"], "actions": ["read"], "agreed": True},
            {"tool_id": "Svc/hello", "domains": ["system_misc"], "actions": ["read"], "agreed": True}]


def _zh_rows():
    return [{"tool_id": "Svc/hi#2", "description_zh": "打招呼"},
            {"tool_id": "Svc/hello", "description_zh": ""}]


def _e2e_requests():
    return [{"id": f"q{i}", "text": f"请求 {i}", "text_orig": f"request {i}",
             "tool_ids": ["Svc/hi#2"] if i % 2 == 0 else ["Svc/hello"],
             "candidate_tool_ids": ["Svc/hi#2", "Svc/hello"],
             "need": "single" if i % 3 else "none"}
            for i in range(6)]


# ---------------------------------------------------------------- A3 use_filter

class _RecordingCatalog:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, domains=None, k=5, actions=None, domain_probs=None, **kw):
        self.calls.append({"domains": domains, "actions": actions, "domain_probs": domain_probs})
        return []


class _FakeHead:
    def predict(self, texts):
        return [HeadOutput(need="single", domains=["comms"], actions=["send"], confidence=0.9,
                           domain_probs={"comms": 0.9})]


def test_use_filter_false_passes_none_domains():
    """A3:use_filter=False 时 retrieve 收到的 domains 是 None(不传动作与概率)。"""
    cat = _RecordingCatalog()
    router = Router(_FakeHead(), cat, use_filter=False)
    router.route("发封邮件")
    assert cat.calls == [{"domains": None, "actions": None, "domain_probs": None}]


def test_use_filter_true_is_the_default_and_filters():
    """A3:默认 True,retrieve 收到预测的能力域 / 动作 / 概率。"""
    cat = _RecordingCatalog()
    Router(_FakeHead(), cat).route("发封邮件")
    call = cat.calls[0]
    assert call["domains"] == ["comms"]
    assert call["actions"] == ["send"]
    assert call["domain_probs"] == {"comms": 0.9}


# ---------------------------------------------------------------- A2 synth 行为不变

def _strip_latency(d):
    return {k: v for k, v in d.items() if k != "latency_ms"}


def test_synth_train_router_behavior_unchanged(tmp_path, monkeypatch):
    """A2:cli train --catalog synth 与改前的 ev.build_system 在同一组输入上路由结果相同。"""
    tools, train = _synth_tools(), _synth_requests()
    tp, rp = tmp_path / "tools.jsonl", tmp_path / "requests.jsonl"
    tp.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tools), encoding="utf-8")
    rp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    monkeypatch.setattr(cli, "MODELS", tmp_path / "models")

    a = argparse.Namespace(encoder="tfidf", tools=str(tp), requests=str(rp), threshold=0.55,
                           top_k=5, catalog="synth")
    assert cli.cmd_train(a) is None
    with open(tmp_path / "models" / "router.pkl", "rb") as f:
        router = pickle.load(f)

    assert router.catalog_name == "synth"
    assert router.use_filter is True
    base = ev.build_system("tfidf", tools, train, threshold=0.55, top_k=5)
    for text in ("给张三发一封邮件", "解释一下什么是梯度下降", "跑一下这段脚本"):
        assert _strip_latency(router.route(text).to_dict()) == \
               _strip_latency(base.route(text).to_dict()), text


def test_cli_route_output_shows_catalog(tmp_path, monkeypatch, capsys):
    """route 的输出里带上用了哪个目录。"""
    tools, train = _synth_tools(), _synth_requests()
    tp, rp = tmp_path / "tools.jsonl", tmp_path / "requests.jsonl"
    tp.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tools), encoding="utf-8")
    rp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    monkeypatch.setattr(cli, "MODELS", tmp_path / "models")
    cli.cmd_train(argparse.Namespace(encoder="tfidf", tools=str(tp), requests=str(rp),
                                     threshold=0.55, top_k=5, catalog="synth"))
    capsys.readouterr()          # 丢掉训练那行输出,只看 route 的
    args = argparse.Namespace(text="给张三发一封邮件", llm_fallback=False, jev_fallback=False,
                              fallback_client=None, jev_timeout=30.0, jev_model=None,
                              fallback_strategy="replace", fallback_trigger="low_conf",
                              need_version="v2", model="claude-sonnet-5-5")
    cli.cmd_route(args)
    out = json.loads(capsys.readouterr().out)
    assert out["catalog"] == "synth"
    assert "candidates" in out


# ---------------------------------------------------------------- 真实目录 Router

def test_build_real_router_bilingual_and_no_filter():
    """真实目录 Router:保留 #N 的 tool_id、索引拼中文、软过滤关、风险用 final_risk。"""
    router = tr.build_real_router(_real_tools(), _labeled_rows(), _zh_rows(),
                                  _synth_requests(), _synth_tools(), use_filter=False)
    assert router.use_filter is False
    ids = [r.tool_id for r in router.catalog.records]
    assert set(ids) == {"Svc/hi#2", "Svc/hello"}
    by_id = {r.tool_id: r for r in router.catalog.records}
    assert "打招呼" in by_id["Svc/hi#2"].description
    assert "打招呼" not in by_id["Svc/hello"].description
    assert all(r.risk in {"L0", "L1", "L2", "L3"} for r in router.catalog.records)


# ---------------------------------------------------------------- e2e

def test_e2e_eval_structure_and_render():
    """e2e 在假数据上跑通:三行目录 + 中英文 + 主口径与上限,报告与 json 都出得来。"""
    res = tr.e2e_eval(_real_tools(), _labeled_rows(), _zh_rows(), _e2e_requests(),
                      _synth_tools(), _synth_requests(), [], folds=5, seed=0)
    assert res["catalogs"] == ["synth", "real_mono", "real_bi"]
    for cat in res["catalogs"]:
        assert set(res["per"][cat]) == {r["id"] for r in _e2e_requests()}
        for lang in ("zh", "en"):
            item = res["per"][cat]["q0"][lang]
            for key in ("main_r1", "main_r5", "main_r1_name", "main_r5_name",
                        "upper_r1", "upper_r5", "latency_ms", "risk_top1", "pred_none"):
                assert key in item

    summary = tr.e2e_summary(res)
    for cat in res["catalogs"]:
        assert 0.0 <= summary[cat]["zh_main_r5_strict"] <= 1.0
        assert summary[cat]["latency_p95"] >= summary[cat]["latency_p50"] >= 0.0
    need = tr.e2e_need_metrics(res, "zh")
    assert need["n"] == sum(1 for r in _e2e_requests() if r["need"])
    assert 0.0 <= need["recall"] <= 1.0

    report = tr.render_e2e(res)
    for section in ("# 端到端评估(T-012)", "## 1. need 判定", "## 2. 检索(三行对照)",
                    "## 3. 延迟", "## 4. top-1 工具风险分布"):
        assert section in report
    for cat in res["catalogs"]:
        assert res["labels"][cat] in report
    assert "假设 need 全对" in report and "need 误判计 0" in report
    json.dumps(tr.e2e_metrics_json(res), ensure_ascii=False)


# ---------------------------------------------------------------- A6 README 来源

def test_readme_cites_reports_and_default_catalog():
    """A6:README 的指标都标了来源报告,快速开始用真实目录(默认 train)。"""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for report in ("reports/eval_t010.md", "reports/eval_t011.md", "reports/eval_e2e.md"):
        assert report in text
    assert "48.4%" in text and "10.2%" in text       # T-011 双语/单语中文 R@5
    assert "python -m mcpcls.cli train" in text
    assert "帮我订一张明天去上海的火车票" in text
