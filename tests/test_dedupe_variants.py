"""T-013:检索结果按函数名去重的回归测试。

全程不联网:用小号假数据走同一条代码路径。
覆盖 A2(关去重逐条不变、合成目录开去重也不变)、A3(去重后无同名、留最高分变体)、
A4(dedupe-eval 报告结构)、A6(真实目录默认去重与开发折一致)。
"""

import json
import pathlib

from mcpcls import evaluate as ev
from mcpcls import tools_real as tr
from mcpcls.heads import HeadOutput
from mcpcls.router import Router, dedupe_by_name

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 假目录 / 假头

class _Rec:
    def __init__(self, tool_id, name, risk="L0"):
        self.tool_id, self.name, self.risk = tool_id, name, risk
        self.domains = []


class _RankedCatalog:
    """固定排序的目录:retrieve 只按给定顺序返回前 k 条,用来把去重行为钉死。"""

    def __init__(self, hits):
        self._hits = list(hits)
        self.records = [r for r, _ in self._hits]
        self.calls = []

    def retrieve(self, query, domains=None, k=5, actions=None, domain_probs=None, **kw):
        self.calls.append(k)
        return self._hits[:k]


class _FakeHead:
    def predict(self, texts):
        return [HeadOutput(need="single", domains=["system_misc"], actions=["read"],
                           confidence=0.9, domain_probs={"system_misc": 0.9})]


def _same_name_hits():
    """前两条同名(`weather` 的两个变体),后面各一个名字。"""
    return [(_Rec("Svc/weather#7", "weather"), 0.90),
            (_Rec("Svc/weather#6", "weather"), 0.80),
            (_Rec("Svc/news", "news"), 0.70),
            (_Rec("Svc/weather#5", "weather"), 0.60),
            (_Rec("Svc/mail", "mail"), 0.50)]


# ---------------------------------------------------------------- dedupe_by_name 纯函数

def test_dedupe_keeps_highest_scoring_variant():
    """A3:同名只留一条,且留的是分数最高的那个变体。"""
    out = dedupe_by_name(_same_name_hits(), 10)
    assert [r.name for r, _ in out] == ["weather", "news", "mail"]
    assert out[0][0].tool_id == "Svc/weather#7"


# ---------------------------------------------------------------- A3 Router 去重

def test_router_dedupe_true_has_no_duplicate_names():
    """A3:打开去重后 top_k 里没有重复函数名,且留下最高分变体。"""
    cat = _RankedCatalog(_same_name_hits())
    router = Router(_FakeHead(), cat, top_k=3, use_filter=False, dedupe_by_name=True)
    cands = router.route("查天气").candidates
    names = [c["tool_id"] for c in cands]
    assert names == ["Svc/weather#7", "Svc/news", "Svc/mail"]
    # 先多取候选:top_k*10 取满目录(这里只有 5 条)
    assert cat.calls == [5]


def test_router_dedupe_false_keeps_old_topk():
    """关去重时按原样取 top_k,不放大候选池。"""
    cat = _RankedCatalog(_same_name_hits())
    router = Router(_FakeHead(), cat, top_k=3, use_filter=False, dedupe_by_name=False)
    cands = router.route("查天气").candidates
    assert [c["tool_id"] for c in cands] == ["Svc/weather#7", "Svc/weather#6", "Svc/news"]
    assert cat.calls == [3]


def test_dedupe_on_unique_names_is_identical_to_off():
    """A2:目录里没有同名变体时,开去重与关去重逐条一致(合成目录就是这种情况)。"""
    hits = [(_Rec("Svc/a", "a"), 0.9), (_Rec("Svc/b", "b"), 0.8), (_Rec("Svc/c", "c"), 0.7)]
    off = Router(_FakeHead(), _RankedCatalog(hits), top_k=3, use_filter=False, dedupe_by_name=False)
    on = Router(_FakeHead(), _RankedCatalog(hits), top_k=3, use_filter=False, dedupe_by_name=True)
    assert on.route("x").candidates == off.route("x").candidates


# ---------------------------------------------------------------- A2 默认关去重 = 旧行为

def _synth_tools():
    rows = [("send_email", "comms", "send"), ("list_files", "file", "read"),
            ("get_weather", "weather_geo", "read")]
    return [{"tool_id": f"synth/{n}", "name": n, "server": "synth", "description": f"{n} tool.",
             "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}},
             "domains": [d], "actions": [a]} for n, d, a in rows]


def _synth_requests():
    return [
        {"text": "给张三发一封邮件", "need": "single", "domains": ["comms"], "actions": ["send"],
         "tool_ids": ["synth/send_email"]},
        {"text": "帮我列出我的文件", "need": "single", "domains": ["file"], "actions": ["read"],
         "tool_ids": ["synth/list_files"]},
        {"text": "今天天气怎么样", "need": "single", "domains": ["weather_geo"], "actions": ["read"],
         "tool_ids": ["synth/get_weather"]},
        {"text": "解释一下什么是梯度下降", "need": "none", "domains": [], "actions": [],
         "tool_ids": []},
    ]


def _strip_latency(d):
    return {k: v for k, v in d.items() if k != "latency_ms"}


def test_dedupe_false_matches_baseline_router():
    """A2:dedupe_by_name=False 的路由结果和改前(ev.build_system)逐条一致。"""
    tools, train = _synth_tools(), _synth_requests()
    base = ev.build_system("tfidf", tools, train, threshold=0.55, top_k=5)
    router = Router(base.head, base.catalog, threshold=0.55, top_k=5, use_filter=True,
                    dedupe_by_name=False)
    for r in train:
        assert _strip_latency(router.route(r["text"]).to_dict()) == \
               _strip_latency(base.route(r["text"]).to_dict()), r["text"]


def test_synth_catalog_dedupe_on_is_noop():
    """A2:合成目录没有同名变体,打开去重也不应有任何变化。"""
    tools, train = _synth_tools(), _synth_requests()
    base = ev.build_system("tfidf", tools, train, threshold=0.55, top_k=5)
    on = Router(base.head, base.catalog, threshold=0.55, top_k=5, use_filter=True,
                dedupe_by_name=True)
    for r in train:
        assert _strip_latency(on.route(r["text"]).to_dict()) == \
               _strip_latency(base.route(r["text"]).to_dict()), r["text"]


def test_build_real_router_dedupe_defaults_off():
    """A6:真实目录 router 默认不打开去重(开发折上严格 R@5 下降)。"""
    tools = [{"tool_id": "Svc/hi#2", "name": "hi", "server": "Svc", "description": "Say hi.",
              "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}]
    labeled = [{"tool_id": "Svc/hi#2", "domains": ["system_misc"], "actions": ["read"],
                "agreed": True}]
    zh = [{"tool_id": "Svc/hi#2", "description_zh": "打招呼"}]
    router = tr.build_real_router(tools, labeled, zh, _synth_requests(), _synth_tools())
    assert router.dedupe_by_name is False
    assert router.use_filter is False


# ---------------------------------------------------------------- A4 去重评估报告

def _real_tools():
    """两个同名变体 + 一个别的名字:让去重有可见效果。"""
    return [{"tool_id": "Svc/weather#2", "name": "weather", "server": "Svc",
             "description": "Get weather.", "inputSchema": {"type": "object",
                                                             "properties": {"a": {"type": "string"}}}},
            {"tool_id": "Svc/weather#9", "name": "weather", "server": "Svc",
             "description": "Get the weather forecast.", "inputSchema": {"type": "object",
                                                                         "properties": {"a": {"type": "string"}}}},
            {"tool_id": "Svc/news", "name": "news", "server": "Svc", "description": "Read news.",
             "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}}}]


def _labeled_rows():
    return [{"tool_id": t["tool_id"], "domains": ["system_misc"], "actions": ["read"],
             "agreed": True} for t in _real_tools()]


def _zh_rows():
    return [{"tool_id": "Svc/weather#2", "description_zh": "查询天气"},
            {"tool_id": "Svc/weather#9", "description_zh": "查询天气"},
            {"tool_id": "Svc/news", "description_zh": "看新闻"}]


def _e2e_requests():
    return [{"id": f"q{i}", "text": f"查询天气 {i}", "text_orig": f"get weather {i}",
             "tool_ids": ["Svc/weather#2"], "candidate_tool_ids": ["Svc/weather#2", "Svc/news"],
             "need": "single"} for i in range(6)]


def test_dedupe_eval_structure_and_render():
    """A4:dedupe-eval 在假数据上跑通,开发折 / 测试折、去重关 / 开、中英文、R@1/3/5 都在。"""
    res = tr.dedupe_eval(_real_tools(), _labeled_rows(), _zh_rows(), _e2e_requests(),
                         _synth_tools(), _synth_requests(), [], folds=5, seed=0)
    assert set(res["per"]) == {"off", "on"}
    assert res["dev_ids"] and res["test_ids"]
    assert set(res["per"]["on"]) == {r["id"] for r in _e2e_requests()}
    sample = res["per"]["on"][res["dev_ids"][0]]
    for k in tr.T013_RATIOS:
        for key in (f"zh_main_r{k}_strict", f"zh_upper_r{k}_name"):
            assert key in sample

    report = tr.render_t013(res)
    for section in ("# 检索按函数名去重(T-013)", "## 1. 开发折", "## 2. 测试折",
                    "## 3. 金标变体被同名变体顶掉"):
        assert section in report
    assert "去重" in report and "严格" in report and "同名" in report
    assert "R@1 | R@3 | R@5" in report
    json.dumps(tr.t013_metrics_json(res), ensure_ascii=False)


def test_dropped_gold_counts_low_scoring_variant():
    """A4:金标偏偏是低分变体时,统计到"被顶掉";金标是最高分变体时不算。"""
    cat = _RankedCatalog([(_Rec("Svc/weather#7", "weather"), 0.9),
                          (_Rec("Svc/weather#5", "weather"), 0.6)])
    name_of = {"Svc/weather#5": "weather", "Svc/weather#7": "weather"}
    assert tr._dropped_gold(cat, "天气", ["Svc/weather#5"], name_of, 5) == (1, True)
    assert tr._dropped_gold(cat, "天气", ["Svc/weather#7"], name_of, 5) == (0, False)


# ---------------------------------------------------------------- A6 README 来源

def test_readme_cites_t013_report():
    """A6:README 的建议一节引用 eval_t013.md,并给出开发折结论的数字。"""
    text = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "reports/eval_t013.md" in text
    assert "51.0%" in text and "43.9%" in text     # 开发折中文上限严格 R@5:关 → 开
