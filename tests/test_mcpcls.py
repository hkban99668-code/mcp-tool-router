import json
import pathlib
import types

import pytest
from sklearn.model_selection import StratifiedKFold

from mcpcls import evaluate as ev
from mcpcls import rules
from mcpcls.encoder import build_encoder
from mcpcls.heads import MultiHead
from mcpcls.labeler import LLMLabeler, _parse
from mcpcls.rules import (final_risk, request_features, request_need_features, rule_actions, rule_risk,
                          split_identifier)
from mcpcls.taxonomy import DOMAIN_IDS, category_cards, max_risk

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def seed():
    return (ev.load_jsonl(DATA / "tools.jsonl"), ev.load_jsonl(DATA / "requests_train.jsonl"),
            ev.load_jsonl(DATA / "requests_gold.jsonl"))


@pytest.fixture(scope="module")
def router(seed):
    tools, train, _ = seed
    return ev.build_system("tfidf", tools, train)


def tool(name, params=(), ann=None, desc=""):
    return {"server": "s", "name": name, "description": desc,
            "inputSchema": {"properties": {p: {} for p in params}}, "annotations": ann or {}}


def test_split_identifier():
    assert split_identifier("getFileContents") == ["get", "file", "contents"]
    assert split_identifier("delete-pod_now") == ["delete", "pod", "now"]


def test_rule_actions_from_verbs_params_and_annotations():
    assert rule_actions(tool("list_files")) == {"read"}
    assert "send" in rule_actions(tool("do_it", ["recipient"]))
    assert "execute" in rule_actions(tool("helper", ["command"]))
    assert "delete" in rule_actions(tool("sync", ann={"destructiveHint": True}))


def test_risk_never_trusts_readonly_hint_over_dangerous_schema():
    t = tool("quick_pay", ["payee", "amount"], ann={"readOnlyHint": True})
    assert rule_risk(t, ["business"]) == "L3"
    # 模型判成只读,规则兜底仍是 L3
    assert final_risk(t, ["read"], ["business"]) == "L3"


def test_risk_levels():
    assert rule_risk(tool("get_x"), ["database"]) == "L0"
    assert rule_risk(tool("search_web"), ["search_web"]) == "L1"
    assert rule_risk(tool("create_page"), ["knowledge"]) == "L2"
    assert max_risk("L1", "L3", "L0") == "L3"


def test_new_domains_are_not_open_world():
    """T-009 的两个新域不在 OPEN_WORLD_DOMAINS 里:只读仍是 L0,要抬到 L1 只能靠
    工具的 openWorldHint —— 和它们在 12 域时代归 system_misc 时的口径完全相同。
    """
    t = tool("get_weather")
    assert rule_risk(t, ["weather_geo"]) == "L0"
    assert rule_risk(t, ["entertainment"]) == "L0"
    assert rule_risk(tool("get_weather", ann={"openWorldHint": True}), ["weather_geo"]) == "L1"


def test_request_features():
    assert "multi_step_cues" in request_features("查一下订单,然后发到群里")
    assert "entity=url" in request_features("抓取 https://a.com")


def test_category_cards_cover_all_domains():
    cards = category_cards()
    assert all(d in cards for d in DOMAIN_IDS)


def test_router_end_to_end(router):
    r = router.route("给张三发封邮件说合同签了")
    assert r.need != "none"
    assert "comms" in r.domains
    assert r.candidates and r.candidates[0]["tool_id"] == "gmail/send_email"
    assert r.requires_confirmation  # 发邮件是 L3
    assert router.route("解释一下精确率和召回率").need == "none"


def test_router_llm_fallback_used_when_low_confidence(router):
    router.threshold, old = 1.01, router.threshold  # 强制全部走兜底
    router.llm_fallback = lambda text: {"need": "single", "domains": ["media"], "actions": ["create"]}
    try:
        r = router.route("随便说点什么")
        assert r.source == "llm_fallback" and r.domains == ["media"]
        assert all("media" in c["domains"] for c in r.candidates)
    finally:
        router.threshold, router.llm_fallback = old, None


def test_original_head_passthrough(router):
    router.original_head = lambda text: {"intent": "chitchat"}
    try:
        assert router.route("你好").original == {"intent": "chitchat"}
    finally:
        router.original_head = None


def test_head_save_load_roundtrip(router, tmp_path, seed):
    p = tmp_path / "tool_head.pkl"
    router.catalog.head.save(p)
    loaded = MultiHead.load(p)
    tools = seed[0][:5]
    from mcpcls.rules import tool_text
    a = router.catalog.head.predict([tool_text(t) for t in tools], items=tools)
    b = loaded.predict([tool_text(t) for t in tools], items=tools)
    assert [x.domains for x in a] == [x.domains for x in b]


def test_old_model_is_rejected_after_the_taxonomy_grew(router, tmp_path, seed, monkeypatch):
    """T-009 把能力域从 12 个加到 14 个:旧模型必须报清楚的错,而不是静默少算两个域的概率。

    静默的后果比报错坏:`predict` 里 `zip(DOMAIN_IDS, probs)` 会被截短,
    新域的概率永远是"少两列"的那份,没人看得出来。
    """
    from mcpcls import heads
    tools, train, _ = seed
    p = tmp_path / "request_head.pkl"
    monkeypatch.setattr(heads, "DOMAIN_IDS", heads.DOMAIN_IDS[:12])     # 假装还在 12 域时代
    old = ev.train_request_head(ev.fit_encoder("tfidf", tools, train), train)
    old.save(p)
    monkeypatch.undo()                                                  # 现在是 14 个域

    with pytest.raises(ValueError, match="12 个能力域上训练"):
        old.check_taxonomy()
    with pytest.raises(ValueError, match="当前 taxonomy 有 14 个"):       # 加载时也要挡住
        heads.MultiHead.load(p)

    # 同一个 taxonomy 下训出来的模型不受影响
    fresh = ev.train_request_head(ev.fit_encoder("tfidf", tools, train), train)
    assert len(fresh.predict(["给张三发封邮件"])) == 1


def test_offline_metrics_meet_smoke_thresholds(router, seed):
    tools, train, gold = seed
    req = ev.eval_requests(router, gold)
    assert req["need"]["recall_needs_mcp"] >= 0.9
    assert req["routing"]["filtered@5"] >= req["routing"]["unfiltered@5"] - 0.02
    adv = ev.eval_tools_adversarial(router, ev.load_jsonl(DATA / "tools_adversarial.jsonl"))
    assert adv["risk_under_report_rate"] == 0


def test_labeler_dual_label_with_fake_client():
    replies = iter([
        '{"need": "single", "domains": ["comms"], "actions": ["send"], "reason": "发邮件", "confidence": 0.9}',
        '好的 {"need": "single", "domains": ["comms"], "actions": ["send"]}',
        '{"need": "single", "domains": ["comms"], "actions": ["send"]}',
        '{"need": "single", "domains": ["calendar_task"], "actions": ["create"]}',
    ])

    def create(**kw):
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=next(replies))])

    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    lab = LLMLabeler(client=client)
    assert lab.dual_label("给张三发邮件")["agreed"] is True
    assert lab.dual_label("约张三开会")["agreed"] is False


def test_parse_drops_unknown_ids():
    d = _parse('{"need": "maybe", "domains": ["comms", "bogus"], "actions": ["fly"]}')
    assert d == {"need": "single", "domains": ["comms"], "actions": []}


def test_jev_encoder_plugs_in(seed):
    tools, train, _ = seed
    r = ev.build_system("jev:tests.fake_jev:encode", tools, train)
    assert r.route("给张三发封邮件").need in {"none", "single", "multi"}


# ---------------- T-001 A 段:评估改造 ----------------


def _small_train(k=None):
    """小样本集:每类取几条,够分层就行,让交叉验证测试跑得快。"""
    train = ev.load_jsonl(DATA / "requests_train.jsonl")
    pick = k or {"none": 6, "single": 12, "multi": 4}
    out = []
    for need, n in pick.items():
        out += [r for r in train if r["need"] == need][:n]
    return out


def test_cross_validate_requests_is_reproducible(seed):
    """A3:折数与分层固定后,交叉验证结果可复现。"""
    tools = seed[0]
    small = _small_train()
    first = ev.cross_validate_requests("tfidf", tools, small, folds=4, random_state=0)
    again = ev.cross_validate_requests("tfidf", tools, small, folds=4, random_state=0)

    assert first["folds"] == 4 and len(first["per_fold"]) == 4
    assert first["per_fold"] == again["per_fold"]
    assert first["recall_none"] == again["recall_none"]

    # 折的划分就是声明的分层切分:每折测试集大小一致,且三类都在
    labels = [r["need"] for r in small]
    split = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(labels, labels))
    assert [f["n"] for f in first["per_fold"]] == [len(te) for _, te in split]
    for _, te in split:
        assert {labels[i] for i in te} >= {"none", "multi"}


def test_summarized_cv_reports_mean_and_std(seed):
    tools = seed[0]
    cv = ev.cross_validate_requests("tfidf", tools, _small_train(), folds=4, random_state=0)
    for key in ("recall_none", "macro_f1_3class", "recall_needs_mcp", "precision_needs_mcp"):
        assert set(cv[key]) == {"mean", "std"}
        assert 0.0 <= cv[key]["mean"] <= 1.0 and cv[key]["std"] >= 0.0


# ---------------- T-001 B 段:need 判定 ----------------


def test_request_need_features_separate_knowledge_from_user_data():
    """知识/写作意图要认出来;"解释一下我仓库里的 X"这种要读数据的句式不能被误判。"""
    knowledge = request_need_features("解释一下什么是闭包")
    assert knowledge["none_intent=explain"] > 0
    assert knowledge["no_object_intent"] == 1.0

    assert request_need_features("写一首关于秋天的诗")["no_object_intent"] == 1.0

    # 反例:同样是"解释一下",但对象是用户自己的仓库
    user_data = request_need_features("解释一下我仓库里 setup.py 做了什么")
    assert user_data["has_object_hint"] == 1.0
    assert "no_object_intent" not in user_data

    assert "no_object_intent" not in request_need_features("总结一下昨天群里聊了啥")
    assert "no_object_intent" not in request_need_features("把 test 数据删了,然后在 #dev 说一声")


def test_need_head_keeps_shared_features_when_no_need_feature_fn():
    """回归:不传 need_feature_fn 时,need 头必须和能力域头共用 request_features。

    曾经在这里把 need 头的输入写成"只有编码器特征",multi_step_cues 整个丢掉,
    gold 的 multi 判定当场塌掉(macro-F1 84%→53%)。这里用特征宽度把它钉住。
    """
    texts = ["写一首关于秋天的诗", "解释一下我仓库里 setup.py 做了什么", "把这个文件删了,然后发邮件"]
    enc = build_encoder("tfidf").fit(texts)
    domains = [["file"], ["code"], ["file"]]
    actions = [["read"], ["read"], ["delete", "send"]]

    shared = MultiHead(enc, with_need=True, feature_fn=request_features)
    shared.fit(texts, domains, actions, needs=["none", "single", "multi"], items=texts)
    xd, _, xn = shared._features(texts, texts)
    assert xn.shape == xd.shape, "无 need_feature_fn 时 need 头不应丢掉 request_features"

    extra = MultiHead(enc, with_need=True, feature_fn=request_features,
                      need_feature_fn=request_need_features)
    extra.fit(texts, domains, actions, needs=["none", "single", "multi"], items=texts)
    xd2, _, xn2 = extra._features(texts, texts)
    assert xn2.shape[1] > xd2.shape[1], "need 专用特征应是叠加,不是替换"


def test_default_head_uses_selected_need_options(router):
    """默认上线的就是交叉验证上选定的组合(规则特征 + 两级判定)。"""
    assert router.head.need_two_stage is True
    assert router.head.need_feature_fn is request_need_features


def test_two_stage_need_head_survives_a_missing_class(seed):
    """T-010a:训练集缺 multi 时不能崩(第二段只有一个类),且要把"multi 预测不出来"说出来。

    v3 按主域判定重算之前,一致集里一条 multi 都没有,evaluate-real 当场死在
    LogisticRegression 的 "only one class: 'single'" 上。
    """
    tools, train, gold = seed
    rows = [r for r in train if r["need"] != "multi"]
    assert rows and not any(r["need"] == "multi" for r in rows)
    head = ev.train_request_head(ev.fit_encoder("tfidf", tools, rows), rows)
    assert head.need_missing == {"multi"}
    outs = head.predict(["先查日程再发邮件", "解释一下什么是闭包"])
    for o in outs:
        assert set(o.need_probs) == {"none", "single", "multi"}
        assert abs(sum(o.need_probs.values()) - 1.0) < 1e-6
        assert o.need_probs["multi"] == 0.0 and o.need != "multi"


def test_two_stage_need_probs_stay_a_distribution(router):
    """两级判定的合成概率仍要落在单纯形上,否则 Router 的置信度语义会坏。"""
    outs = router.head.predict(["解释一下什么是闭包", "把 test 数据删了,然后在 #dev 说一声", "给我发封邮件"])
    for o in outs:
        assert set(o.need_probs) == {"none", "single", "multi"}
        assert abs(sum(o.need_probs.values()) - 1.0) < 1e-6
        assert 0.0 <= o.confidence <= 1.0
        assert o.need in {"none", "single", "multi"}


# ---------------- T-001 第 2 轮:词来源标注与泄漏检查 ----------------


def test_keyword_sources_cover_every_keyword():
    """每个词都要标来源,而且不能一个词标两个来源、也不能标了词表里没有的词。"""
    declared = [w for words in rules.KEYWORD_SOURCES.values() for w in words]
    assert len(declared) == len(set(declared)), "同一个词被标了多个来源"
    assert set(declared) == {w for _, w in rules.all_keywords()}


def test_keyword_sources_match_the_data():
    """来源标注要与数据自洽:train 的词必须在训练集里、taxonomy 的词要在 taxonomy.py 里、
    prior 的词不能在训练集里出现(出现了就应该标 train)。"""
    train_text = " ".join(r["text"] for r in ev.load_jsonl(DATA / "requests_train.jsonl")).lower()
    assert rules.keyword_source_problems(train_text, rules.taxonomy_vocabulary()) == []


def test_gold_only_keywords_are_pinned(seed):
    """训练集 0 次、gold 至少 1 次的词必须与代码里声明的集合一致。

    新增这类词必须显式改 GOLD_ONLY_KEYWORDS,否则测试失败——
    它们对交叉验证不可见,悄悄加进来就等于按 gold 调参。
    """
    tools, train, gold = seed
    leak = ev.keyword_leak_check(train, gold)
    assert leak["matches_declared"], f"实际 {leak['words']} 与声明 {leak['declared']} 不一致"
    assert leak["source_problems"] == []


def test_cv_is_blind_to_gold_only_words(seed):
    """交叉验证看不见 gold-only 词:去掉它们前后每折的每个 need 指标都逐位相同。

    这些词在训练集里 0 次,任何一折的训练段和测试段都不会触发,
    所以 5 折交叉验证无论多好都不能作为它们的证据。
    """
    tools, train, _ = seed
    full = ev.cross_validate_requests("tfidf", tools, train, folds=5, random_state=0,
                                      need_feature_fn=rules.request_need_features,
                                      need_two_stage=True)
    dropped = ev.cross_validate_requests("tfidf", tools, train, folds=5, random_state=0,
                                         need_feature_fn=rules.request_need_features_ablated,
                                         need_two_stage=True)
    assert full["per_fold"] == dropped["per_fold"]


def test_ablation_drops_exactly_the_gold_only_words():
    """消融只去掉 gold-only 词:命中它们的意图信号消失,其他词的信号不变。"""
    full = rules.request_need_features("写一首关于秋天的诗")
    ablated = rules.request_need_features_ablated("写一首关于秋天的诗")
    assert full["no_object_intent"] == 1.0  # "写一首" 命中 authoring
    assert "no_object_intent" not in ablated  # 消融后没有意图词了

    kept = rules.request_need_features("解释一下什么是闭包")
    assert rules.request_need_features_ablated("解释一下什么是闭包") == kept
