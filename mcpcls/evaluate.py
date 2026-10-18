"""离线评估,对应方案"评估方式"一节。

请求侧(金标集):need 召回/精确率、能力域与动作 micro/macro-F1、路由 Recall@k(过滤 vs 不过滤)、
候选缩减比例、p95 延迟。
工具侧:按 server 留一交叉验证(测试 server 不出现在训练集),报能力域/动作 F1、风险准确率、
L3 召回率与低报率;另跑对抗集(描述无害但 schema 危险、描述夹带指令)。

调参纪律:`cross_validate_requests` 在训练集上做分层交叉验证,是唯一的选择依据;
gold 与探针集(`eval_probe`)只用于报告最终结果。
"""

import copy
import json
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

from .catalog import ToolCatalog, tool_id
from .encoder import build_encoder
from .heads import MultiHead
from .jev import DEFAULT_JEV_MODEL, jev_fallback
from .router import Router
from .rules import (GOLD_ONLY_KEYWORDS, all_keywords, keyword_source_problems, request_features,
                    request_need_features, request_need_features_ablated, risk_from, rule_risk,
                    source_of, taxonomy_vocabulary, tool_action_features, tool_features, tool_text)
from .taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS, risk_rank

# B 段在训练集 5 折交叉验证上选定的 need 头方案(见 reports/eval_report.md 的三列对照):
# 叠加"无外部对象"规则特征 + 先判要不要工具的两级判定。两个开关都可以关掉做对照。
NEED_HEAD_DEFAULTS = {"need_feature_fn": request_need_features, "need_two_stage": True}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_risk(tool):
    return risk_from(tool["actions"], tool["domains"], (tool.get("annotations") or {}).get("openWorldHint"))


def need_metrics(gold_need, pred_need) -> dict:
    """need 三分类的核心指标:需要 MCP 的召回/精确率、none 召回、macro-F1。"""
    tp = sum(1 for g, p in zip(gold_need, pred_need) if g != "none" and p != "none")
    pos = sum(1 for g in gold_need if g != "none")
    ppos = sum(1 for p in pred_need if p != "none")
    none_idx = [i for i, g in enumerate(gold_need) if g == "none"]
    return {
        "n": len(gold_need),
        "recall_needs_mcp": tp / pos if pos else 0.0,
        "precision_needs_mcp": tp / ppos if ppos else 0.0,
        # none 只有十几条,单看召回更能暴露闲聊误报
        "recall_none": float(np.mean([pred_need[i] == "none" for i in none_idx])) if none_idx else None,
        "macro_f1_3class": float(f1_score(gold_need, pred_need, average="macro", zero_division=0)),
        "accuracy_3class": float(np.mean([g == p for g, p in zip(gold_need, pred_need)])),
    }


def request_metrics(items, preds) -> dict:
    """逐条 (need, domains, actions) 预测的汇总指标。

    `eval_requests` 需要 tool_ids 才能算路由指标,银标留出集没有 tool_ids,
    所以留出集与 `compare` 共用这一份只算分类指标的口径。
    """
    gold_need = [r["need"] for r in items]
    pred_need = [p[0] for p in preds]
    m = need_metrics(gold_need, pred_need)
    rows = [(r, p) for r, p in zip(items, preds) if r["need"] != "none"]
    if rows:
        domain = multilabel_f1([r["domains"] for r, _ in rows], [p[1] for _, p in rows], DOMAIN_IDS)
        action = multilabel_f1([r["actions"] for r, _ in rows], [p[2] for _, p in rows], ACTION_IDS)
    else:
        domain = action = {"micro_f1": 0.0, "macro_f1": 0.0}
    none_total = sum(1 for g in gold_need if g == "none")
    return {
        "need_accuracy": m["accuracy_3class"],
        "need_macro_f1": m["macro_f1_3class"],
        "recall_needs_mcp": m["recall_needs_mcp"],
        "none_correct": sum(1 for g, p in zip(gold_need, pred_need) if g == "none" and p == "none"),
        "none_total": none_total,
        "domain": domain,
        "action": action,
    }


def confusion_3(gold_need, pred_need) -> dict:
    """3×3 混淆矩阵,行是金标、列是预测。"""
    return {g: {p: sum(1 for a, b in zip(gold_need, pred_need) if a == g and b == p) for p in NEED_IDS}
            for g in NEED_IDS}


def multilabel_f1(gold, pred, classes):
    mlb = MultiLabelBinarizer(classes=list(classes))
    g, p = mlb.fit_transform(gold), mlb.transform(pred)
    present = g.sum(0) > 0  # macro 只在金标里出现过的类别上平均
    return {
        "micro_f1": float(f1_score(g, p, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(g[:, present], p[:, present], average="macro", zero_division=0)),
    }


# ---------------- 训练 ----------------

def fit_encoder(spec, tools, requests):
    enc = build_encoder(spec)
    texts = [tool_text(t) for t in tools] + [f"{t['name'].replace('_', ' ')} {t['description']}" for t in tools]
    texts += [r["text"] for r in requests]
    return enc.fit(texts)


def train_tool_head(enc, tools):
    return MultiHead(enc, with_need=False, feature_fn=tool_features,
                     action_feature_fn=tool_action_features).fit(
        [tool_text(t) for t in tools], [t["domains"] for t in tools], [t["actions"] for t in tools], items=tools)


def train_request_head(enc, requests, need_feature_fn=NEED_HEAD_DEFAULTS["need_feature_fn"],
                       need_two_stage=NEED_HEAD_DEFAULTS["need_two_stage"]):
    """请求侧分类头。两个开关是 B 段用来在交叉验证上比方案的:
    need_feature_fn 只在 need 头叠加"无外部对象"特征;need_two_stage 先判要不要工具。
    默认值就是交叉验证上选定的组合,传 None/False 可以关掉做对照。"""
    return MultiHead(enc, with_need=True, feature_fn=request_features,
                     need_feature_fn=need_feature_fn, need_two_stage=need_two_stage).fit(
        [r["text"] for r in requests], [r["domains"] for r in requests],
        [r["actions"] for r in requests], needs=[r["need"] for r in requests])


def example_queries(requests) -> dict:
    out = {}
    for r in requests:
        if r["need"] == "single":
            for tid in r["tool_ids"]:
                out.setdefault(tid, []).append(r["text"])
    return out


def build_system(spec, tools, requests, threshold=0.55, top_k=5,
                 need_feature_fn=NEED_HEAD_DEFAULTS["need_feature_fn"],
                 need_two_stage=NEED_HEAD_DEFAULTS["need_two_stage"],
                 retrieval_encoder=None, reranker=None):
    enc = fit_encoder(spec, tools, requests)
    tool_head = train_tool_head(enc, tools)
    req_head = train_request_head(enc, requests, need_feature_fn=need_feature_fn,
                                  need_two_stage=need_two_stage)
    # retrieval_encoder(T-019):检索编码器 spec,和分类头分开;None=沿用分类头编码器
    renc = fit_encoder(retrieval_encoder, tools, requests) if retrieval_encoder else None
    catalog = ToolCatalog(tool_head, retrieval_encoder=renc)
    # 已标注(审核过)的工具直接用标注入索引;未标注的新工具由工具侧分类头归类。
    # 训练集里的请求作为每个工具的示例查询,扩充索引文本(新 server 接入时由 LLM 合成)
    catalog.add_tools(tools, use_labels=True, example_queries=example_queries(requests))
    # reranker(T-025):语义重排器实例,None=不重排,行为与改前一致
    return Router(req_head, catalog, threshold=threshold, top_k=top_k, reranker=reranker)


# ---------------- 请求侧 ----------------

def eval_requests(router, gold, ks=(1, 3, 5)):
    outs = router.head.predict([r["text"] for r in gold])
    need_gold = [r["need"] for r in gold]
    need_pred = [o.need for o in outs]
    res = {"n": len(gold), "need": need_metrics(need_gold, need_pred)}
    res["need"]["confusion"] = confusion_3(need_gold, need_pred)
    tool_rows = [(r, o) for r, o in zip(gold, outs) if r["need"] != "none"]
    res["domain"] = multilabel_f1([r["domains"] for r, _ in tool_rows], [o.domains for _, o in tool_rows], DOMAIN_IDS)
    res["action"] = multilabel_f1([r["actions"] for r, _ in tool_rows], [o.actions for _, o in tool_rows], ACTION_IDS)

    cat = router.catalog
    total = len(cat.records)
    recall = {f"filtered@{k}": [] for k in ks} | {f"unfiltered@{k}": [] for k in ks}
    cand = []
    for r, o in tool_rows:
        gold_ids = set(r["tool_ids"])
        for k in ks:
            f_ids = {rec.tool_id for rec, _ in cat.retrieve(r["text"], o.domains, k, actions=o.actions,
                                                             domain_probs=o.domain_probs)}
            u_ids = {rec.tool_id for rec, _ in cat.retrieve(r["text"], None, k)}
            recall[f"filtered@{k}"].append(len(gold_ids & f_ids) / len(gold_ids))
            recall[f"unfiltered@{k}"].append(len(gold_ids & u_ids) / len(gold_ids))
        cand.append(cat.candidate_count(o.domains, o.domain_probs))
    res["routing"] = {k: float(np.mean(v)) for k, v in recall.items()}
    res["routing"]["catalog_size"] = total
    res["routing"]["avg_candidates_after_filter"] = float(np.mean(cand))
    res["routing"]["candidate_reduction"] = 1 - float(np.mean(cand)) / total

    lat = [router.route(r["text"]).latency_ms for r in gold]
    res["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    confs = [o.confidence for o in outs]
    res["llm_fallback_rate"] = float(np.mean([c < router.threshold for c in confs])) if not isinstance(router.threshold, dict) else None

    errors = []
    for r, o in zip(gold, outs):
        if r["need"] != o.need or (r["need"] != "none" and set(r["domains"]) != set(o.domains)):
            errors.append({"text": r["text"], "gold": [r["need"], r["domains"]], "pred": [o.need, o.domains],
                           "conf": round(o.confidence, 3)})
    res["errors"] = errors
    return res


def eval_jev(router, gold, client, mode="fallback", noul_threshold=0.5) -> dict:
    """在金标上跑一组接 Jev 的对照,用来和 `eval_requests` 的本地基线并排看。

    mode="fallback": 只有本地置信度低于阈值时才问 Jev(线上要用的方式);
    mode="full":     每条都问 Jev,作为"全部交给 Jev"的对照基线。

    调用次数、失败次数、费用、Jev 延迟都取自 `client.calls` 里每次 decide 的实际记录,
    不是按阈值估计的(估计值见 `eval_requests` 的 llm_fallback_rate,两者不同,报告里分开写)。
    """
    if mode not in ("fallback", "full"):
        raise ValueError(f"mode 只能是 fallback 或 full,收到 {mode!r}")
    r = copy.copy(router)                 # 不改动传进来的 router
    r.llm_fallback = jev_fallback(client, noul_threshold)
    if mode == "full":
        r.threshold = float("inf")        # 置信度永远够不着 -> 每条都走 Jev
    results = [r.route(item["text"]) for item in gold]

    need_gold = [g["need"] for g in gold]
    need_pred = [x.need for x in results]
    res = {"n": len(gold), "mode": mode, "need": need_metrics(need_gold, need_pred)}
    res["need"]["confusion"] = confusion_3(need_gold, need_pred)
    rows = [(g, x) for g, x in zip(gold, results) if g["need"] != "none"]
    res["domain"] = multilabel_f1([g["domains"] for g, _ in rows], [x.domains for _, x in rows], DOMAIN_IDS)
    res["action"] = multilabel_f1([g["actions"] for g, _ in rows], [x.actions for _, x in rows], ACTION_IDS)

    calls = client.calls
    e2e = [x.latency_ms for x in results] or [0.0]
    jev_lat = [c.latency_ms for c in calls] or [0.0]
    res["jev"] = {
        # 结果真的被 Jev 改写过的比例;Jev 失败回退到本地的条数不算在这里
        "fallback_rate": float(np.mean([x.source == "jev_fallback" for x in results])),
        "called_rate": len(calls) / len(gold) if gold else 0.0,
        "calls": len(calls),
        "failures": sum(1 for c in calls if not c.ok),
        "cost_usd": float(sum(c.cost for c in calls)),
        "input_tokens": int(sum(c.input_tokens for c in calls)),
        "jev_latency_ms": {"p50": float(np.percentile(jev_lat, 50)),
                           "p95": float(np.percentile(jev_lat, 95))},
    }
    res["latency_ms"] = {"p50": float(np.percentile(e2e, 50)), "p95": float(np.percentile(e2e, 95))}
    return res


def cv_splits(train, folds=5, random_state=0):
    """按 need 分层的折划分,产出 (训练下标, 测试下标) 序列。

    `cross_validate_requests` 和 `compare.local_outputs` 共用这一份划分,
    离线对比里"训练集的折外预测"才和评估报告里的交叉验证是同一批预测。
    """
    texts = [r["text"] for r in train]
    labels = [r["need"] for r in train]
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state).split(texts, labels)


def cross_validate_requests(spec, tools, train, folds=5, random_state=0,
                            need_feature_fn=NEED_HEAD_DEFAULTS["need_feature_fn"],
                            need_two_stage=NEED_HEAD_DEFAULTS["need_two_stage"]) -> dict:
    """在训练集上做按 need 分层的 k 折交叉验证:每折用其余折训练请求侧分类头。

    gold 只有 69 条(none 11、multi 5),单条样本就能让指标动几个百分点;
    所以模型与规则的选择只看这里的均值和标准差。
    """
    per_fold = []
    for tr_idx, te_idx in cv_splits(train, folds, random_state):
        tr = [train[i] for i in tr_idx]
        te = [train[i] for i in te_idx]
        enc = fit_encoder(spec, tools, tr)
        head = train_request_head(enc, tr, need_feature_fn=need_feature_fn,
                                  need_two_stage=need_two_stage)
        outs = head.predict([r["text"] for r in te])
        per_fold.append(need_metrics([r["need"] for r in te], [o.need for o in outs]))

    out = {"folds": folds, "random_state": random_state, "n": len(train), "per_fold": per_fold}
    for key in ("recall_needs_mcp", "precision_needs_mcp", "recall_none", "macro_f1_3class",
                "accuracy_3class"):
        vals = [f[key] for f in per_fold if f[key] is not None]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
    return out


def keyword_leak_check(train, gold) -> dict:
    """词表泄漏检查:列出"训练集 0 次、gold >=1 次"的词。

    这类词在 5 折交叉验证的任何一折里都不会触发,所以交叉验证衡量不到它们,
    gold 与探针上的提升里混着它们的贡献。判断规则改动是否真的有效时,
    以"去掉这些词"的消融结果(`--drop-gold-only-words`)作为依据。
    """
    train_text = " ".join(r["text"] for r in train).lower()
    gold_text = " ".join(r["text"] for r in gold).lower()
    rows = []
    for group, word in all_keywords():
        t, g = train_text.count(word.lower()), gold_text.count(word.lower())
        if t == 0 and g >= 1:
            rows.append({"word": word, "group": group, "source": source_of(word), "train": t, "gold": g})
    counts = {src: 0 for src in ("train", "taxonomy", "prior")}
    for _, word in all_keywords():
        counts[source_of(word)] = counts.get(source_of(word), 0) + 1
    return {
        "total_words": len(all_keywords()),
        "source_counts": counts,
        "rows": rows,
        "words": sorted(r["word"] for r in rows),
        "declared": sorted(GOLD_ONLY_KEYWORDS),
        "matches_declared": sorted(r["word"] for r in rows) == sorted(GOLD_ONLY_KEYWORDS),
        "source_problems": keyword_source_problems(train_text, taxonomy_vocabulary()),
    }


def eval_probe(router, probe) -> dict:
    """探针集评估:事先冻结、写完不看,用来检查改动是不是只记住了 gold 的错例。"""
    outs = router.head.predict([r["text"] for r in probe])
    gold_need = [r["need"] for r in probe]
    pred_need = [o.need for o in outs]
    res = need_metrics(gold_need, pred_need)
    res["confusion"] = confusion_3(gold_need, pred_need)
    res["none_total"] = sum(1 for g in gold_need if g == "none")
    res["none_predicted_none"] = sum(1 for g, p in zip(gold_need, pred_need) if g == "none" and p == "none")
    res["needs_mcp_total"] = sum(1 for g in gold_need if g != "none")
    res["needs_mcp_predicted_none"] = sum(
        1 for g, p in zip(gold_need, pred_need) if g != "none" and p == "none")
    res["errors"] = [{"text": r["text"], "gold": r["need"], "pred": o.need,
                      "conf": round(o.confidence, 3), "need_probs": {k: round(v, 3) for k, v in o.need_probs.items()}}
                     for r, o in zip(probe, outs) if r["need"] != o.need]
    return res


# ---------------- 工具侧 ----------------

def _tool_metrics(tools, recs):
    gold_r = [gold_risk(t) for t in tools]
    pred_r = [r.risk for r in recs]
    l3 = [i for i, g in enumerate(gold_r) if g == "L3"]
    under = [i for i, (g, p) in enumerate(zip(gold_r, pred_r)) if risk_rank(p) < risk_rank(g)]
    return {
        "n": len(tools),
        "domain": multilabel_f1([t["domains"] for t in tools], [r.domains for r in recs], DOMAIN_IDS),
        "action": multilabel_f1([t["actions"] for t in tools], [r.actions for r in recs], ACTION_IDS),
        "risk_accuracy": float(np.mean([g == p for g, p in zip(gold_r, pred_r)])),
        "l3_recall": float(np.mean([pred_r[i] == "L3" for i in l3])) if l3 else None,
        "risk_under_report_rate": len(under) / len(tools),
        "under_reported": [recs[i].tool_id for i in under],
    }


def eval_tools_by_server(spec, tools, requests):
    """按 server 留一:每折把一个 server 的工具全部拿出来当测试集。"""
    servers = sorted({t["server"] for t in tools})
    all_tools, all_recs = [], []
    for s in servers:
        train = [t for t in tools if t["server"] != s]
        test = [t for t in tools if t["server"] == s]
        enc = fit_encoder(spec, train, requests)
        head = train_tool_head(enc, train)
        cat = ToolCatalog(head)
        all_recs += cat.classify_tools([{k: v for k, v in t.items() if k not in ("domains", "actions")} for t in test])
        all_tools += test
    res = _tool_metrics(all_tools, all_recs)
    # 只用规则的风险基线
    rule_only = [rule_risk(t, t["domains"]) for t in all_tools]
    gold_r = [gold_risk(t) for t in all_tools]
    res["rule_only_risk_accuracy"] = float(np.mean([g == p for g, p in zip(gold_r, rule_only)]))
    res["servers"] = len(servers)
    res["review_queue"] = sum(r.needs_review for r in all_recs)
    return res


def eval_tools_adversarial(router, adv_tools):
    cat = router.catalog
    recs = cat.classify_tools([{k: v for k, v in t.items() if k not in ("domains", "actions")} for t in adv_tools])
    out = _tool_metrics(adv_tools, recs)
    out["items"] = [{"tool_id": r.tool_id, "gold_risk": gold_risk(t), "pred_risk": r.risk,
                     "pred_actions": r.actions} for t, r in zip(adv_tools, recs)]
    return out


# ---------------- 报告 ----------------

def _pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def _onoff(flag) -> str:
    return "开" if flag else "关"


def _cv_cell(cv, key) -> str:
    stat = cv.get(key)
    return f"{_pct(stat['mean'])} ± {stat['std'] * 100:.1f}%" if stat else "-"


_NEED_METRIC_ROWS = (("需要 MCP 召回率", "recall_needs_mcp"), ("需要 MCP 精确率", "precision_needs_mcp"),
                    ("none 召回率", "recall_none"), ("need macro-F1", "macro_f1_3class"),
                    ("need 准确率", "accuracy_3class"))

# 两个词表口径:是否保留"训练集 0 次、只在 gold 出现"的 13 个词
VARIANT_KEPT, VARIANT_DROPPED = "含 gold-only 词", "去掉 gold-only 词"


def render_report(req, cv, probe, tools_cv, adv, encoder_spec, need_head=None, leak=None,
                  variants=None) -> str:
    L = [f"# MCP 分类评估报告\n", f"编码器:`{encoder_spec}`;数据:手写合成种子数据(见 data/),仅用于冒烟验证。\n"]
    L.append("> 调参只看训练集分层交叉验证;gold 与探针集只用于报告最终结果。\n")
    if need_head:
        L.append(f"> need 头设置:无外部对象规则特征 {_onoff(need_head.get('need_rules'))};"
                 f"两级判定 {_onoff(need_head.get('two_stage'))};"
                 f"gold-only 词 {_onoff(not need_head.get('drop_gold_only'))}。\n")
    if variants:
        L.append("> gold 主数字为去掉 gold-only 词的口径;含词口径在同一张表里并列,"
                 "两种口径的差别就是这些词带来的全部增益。\n")
    L.append(f"## 请求侧(金标 {req['n']} 条)\n")
    L.append("| 指标 | 数值 |\n| --- | --- |")
    n = req["need"]
    L += [f"| 需要 MCP 召回率 | {_pct(n['recall_needs_mcp'])} |",
          f"| 需要 MCP 精确率 | {_pct(n['precision_needs_mcp'])} |",
          f"| need 三分类 macro-F1 | {_pct(n['macro_f1_3class'])} |",
          f"| 能力域 micro-F1 / macro-F1 | {_pct(req['domain']['micro_f1'])} / {_pct(req['domain']['macro_f1'])} |",
          f"| 动作 micro-F1 / macro-F1 | {_pct(req['action']['micro_f1'])} / {_pct(req['action']['macro_f1'])} |",
          f"| LLM 兜底比例(低于阈值) | {_pct(req['llm_fallback_rate'])} |",
          f"| 延迟 p50 / p95 | {req['latency_ms']['p50']:.2f} ms / {req['latency_ms']['p95']:.2f} ms |"]

    L.append(f"\n**need 混淆矩阵**(行=金标,列=预测)\n")
    L.append("| 金标 \\ 预测 | none | single | multi |\n| --- | --- | --- | --- |")
    for g in NEED_IDS:
        row = req["need"]["confusion"][g]
        L.append(f"| {g} | {row['none']} | {row['single']} | {row['multi']} |")

    L.append(f"\n**need 指标三列对照**(交叉验证为训练集 {cv['folds']} 折,"
             f"`random_state={cv['random_state']}`,`n={cv['n']}`)\n")
    dropped, kept = (variants or {}).get(VARIANT_DROPPED), (variants or {}).get(VARIANT_KEPT)
    if dropped and kept:
        L.append(f"| 指标 | gold({req['n']} 条,去词) | gold({req['n']} 条,含词) "
                 f"| 交叉验证(均值 ± 标准差) | 探针({probe['n']} 条,去词) | 探针({probe['n']} 条,含词) |\n"
                 f"| --- | --- | --- | --- | --- | --- |")
        for label, key in _NEED_METRIC_ROWS:
            L.append(f"| {label} | {_pct(dropped['requests']['need'][key])} "
                     f"| {_pct(kept['requests']['need'][key])} | {_cv_cell(cv, key)} "
                     f"| {_pct(dropped['probe'][key])} | {_pct(kept['probe'][key])} |")
        L.append(f"\n去词与含词两种口径的交叉验证完全相同:这些词在训练集里 0 次,任何一折都不会触发,"
                 f"所以 gold 与探针上的差别里有多少是它们的贡献,交叉验证衡量不到"
                 f"(由 `tests/test_mcpcls.py::test_cv_is_blind_to_gold_only_words` 钉住)。"
                 f"判断收益时以 gold(去词)为 gold 主数字。")
    else:
        L.append(f"| 指标 | gold({req['n']} 条) | 训练集交叉验证(均值 ± 标准差) | 探针集({probe['n']} 条) |\n"
                 f"| --- | --- | --- | --- |")
        for label, key in _NEED_METRIC_ROWS:
            L.append(f"| {label} | {_pct(req['need'][key])} | {_cv_cell(cv, key)} | {_pct(probe[key])} |")
    L.append(f"\n探针集:{probe['none_total']} 条 none 中判对 {probe['none_predicted_none']} 条;"
             f"{probe['needs_mcp_total']} 条需要 MCP 的请求中被判 none {probe['needs_mcp_predicted_none']} 条。")
    if probe["errors"]:
        L.append("\n| 探针文本 | 金标 | 预测 | 置信度 |\n| --- | --- | --- | --- |")
        for e in probe["errors"]:
            L.append(f"| {e['text']} | {e['gold']} | {e['pred']} | {e['conf']} |")

    if leak:
        counts = leak["source_counts"]
        L.append(f"\n**词表来源与泄漏检查**:共 {leak['total_words']} 个词,"
                 f"来源 train={counts.get('train', 0)} / taxonomy={counts.get('taxonomy', 0)} / "
                 f"prior={counts.get('prior', 0)};来源标注自洽 {_onoff(not leak['source_problems'])}。\n")
        L.append("| 词 | 组 | 来源 | train 次数 | gold 次数 |\n| --- | --- | --- | --- | --- |")
        for row in leak["rows"]:
            L.append(f"| {row['word']} | {row['group']} | {row['source']} | {row['train']} | {row['gold']} |")
        L.append(f"\n上表 {len(leak['rows'])} 个词在训练集里出现 0 次,5 折交叉验证衡量不到它们,"
                 f"gold 与探针上的增益里混着它们的贡献。"
                 f"主口径已经去掉它们(想看含词口径用 `--keep-gold-only-words`),"
                 f"两种口径在同一张三列对照表里并列。")

    rt = req["routing"]
    L.append(f"\n**路由 Recall@k**(工具目录 {rt['catalog_size']} 个工具,过滤后平均候选 "
             f"{rt['avg_candidates_after_filter']:.1f} 个,缩减 {_pct(rt['candidate_reduction'])})\n")
    L.append("| k | 按能力域过滤 | 不过滤(基线) |\n| --- | --- | --- |")
    for k in (1, 3, 5):
        L.append(f"| {k} | {_pct(rt[f'filtered@{k}'])} | {_pct(rt[f'unfiltered@{k}'])} |")
    L.append(f"\n## 工具侧(按 server 留一,{tools_cv['servers']} 个 server,{tools_cv['n']} 个工具)\n")
    L.append("| 指标 | 数值 |\n| --- | --- |")
    L += [f"| 能力域 micro-F1 / macro-F1 | {_pct(tools_cv['domain']['micro_f1'])} / {_pct(tools_cv['domain']['macro_f1'])} |",
          f"| 动作 micro-F1 / macro-F1 | {_pct(tools_cv['action']['micro_f1'])} / {_pct(tools_cv['action']['macro_f1'])} |",
          f"| 风险准确率(模型+规则) | {_pct(tools_cv['risk_accuracy'])} |",
          f"| 风险准确率(仅规则) | {_pct(tools_cv['rule_only_risk_accuracy'])} |",
          f"| L3 召回率 | {_pct(tools_cv['l3_recall'])} |",
          f"| 风险低报率 | {_pct(tools_cv['risk_under_report_rate'])} |",
          f"| 进入人工审核队列 | {tools_cv['review_queue']} 个 |"]
    if tools_cv["under_reported"]:
        L.append(f"\n低报的工具:{', '.join(tools_cv['under_reported'])}")
    L.append(f"\n## 对抗集({adv['n']} 个工具)\n")
    L.append("| 工具 | 金标风险 | 预测风险 | 预测动作 |\n| --- | --- | --- | --- |")
    for it in adv["items"]:
        L.append(f"| {it['tool_id']} | {it['gold_risk']} | {it['pred_risk']} | {', '.join(it['pred_actions'])} |")
    L.append(f"\nL3 召回率 {_pct(adv['l3_recall'])},低报率 {_pct(adv['risk_under_report_rate'])}。")
    L.append(f"\n## 原有任务回归\n\n第一阶段编码器冻结,原有分类头的输入和参数都不变,原有输出逐条一致。"
             f"接入原方案后用 `Router(original_head=...)` 透传原结果,发布前对比新旧版本输出。")
    if req["errors"]:
        L.append(f"\n## 请求侧错例({len(req['errors'])} 条)\n")
        L.append("| 文本 | 金标 | 预测 | 置信度 |\n| --- | --- | --- | --- |")
        for e in req["errors"]:
            L.append(f"| {e['text']} | {e['gold'][0]} {e['gold'][1]} | {e['pred'][0]} {e['pred'][1]} | {e['conf']} |")
    return "\n".join(L) + "\n"


def render_jev_report(baseline, fallback, full, model=DEFAULT_JEV_MODEL, threshold=0.55,
                      noul_threshold=0.5) -> str:
    """Jev 对照报告:本地基线 / 本地+Jev 兜底 / 全量 Jev,三列并排。"""
    cols = [("本地基线(不接 Jev)", baseline), ("本地 + Jev 兜底", fallback), ("全量交给 Jev", full)]
    L = ["# Jev 对照评估\n",
         f"Jev 模型:`{model}`;兜底阈值 {threshold};noul 阈值 {noul_threshold};"
         f"金标 {fallback['n']} 条。\n",
         "> gold 只用于报告。这三列是为了看 Jev 值不值这份延迟和费用,不是用来挑阈值调参的。\n",
         "\n## 主指标\n",
         "| 指标 | " + " | ".join(name for name, _ in cols) + " |\n| --- | --- | --- | --- |"]

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(r) for _, r in cols) + " |")

    row("need 准确率", lambda r: _pct(r["need"]["accuracy_3class"]))
    row("need 三分类 macro-F1", lambda r: _pct(r["need"]["macro_f1_3class"]))
    row("需要 MCP 召回率", lambda r: _pct(r["need"]["recall_needs_mcp"]))
    row("能力域 micro-F1 / macro-F1",
        lambda r: f"{_pct(r['domain']['micro_f1'])} / {_pct(r['domain']['macro_f1'])}")
    row("动作 micro-F1 / macro-F1",
        lambda r: f"{_pct(r['action']['micro_f1'])} / {_pct(r['action']['macro_f1'])}")
    row("端到端延迟 p50 / p95",
        lambda r: f"{r['latency_ms']['p50']:.2f} ms / {r['latency_ms']['p95']:.2f} ms")

    def jev_row(label, key, fmt=str):
        row(label, lambda r: fmt(r["jev"][key]) if r.get("jev") else "-")

    jev_row("兜底比例(结果来自 Jev)", "fallback_rate", _pct)
    jev_row("实际调用比例(调用次数 / 条数)", "called_rate", _pct)
    jev_row("Jev 调用次数(实际)", "calls")
    jev_row("调用失败次数", "failures")
    jev_row("Jev 调用延迟 p50 / p95", "jev_latency_ms",
            lambda d: f"{d['p50']:.0f} ms / {d['p95']:.0f} ms")
    jev_row("Jev 输入 token 总数", "input_tokens")
    jev_row("Jev 总费用(美元)", "cost_usd", lambda x: f"{x:.6f}")

    for name, r in cols[1:]:
        cm = r["need"]["confusion"]
        L.append(f"\n**{name} 的 need 混淆矩阵**(行=金标,列=预测)\n")
        L.append("| 金标 \\ 预测 | none | single | multi |\n| --- | --- | --- | --- |")
        for g in NEED_IDS:
            L.append(f"| {g} | {cm[g]['none']} | {cm[g]['single']} | {cm[g]['multi']} |")

    L.append(f"\n**两个兜底比例不是一回事**:本地基线里的「LLM 兜底比例(低于阈值)」"
             f"{_pct(baseline['llm_fallback_rate'])} 是按置信度算的估计值,不接 Jev 时没有任何实际调用;"
             f"Jev 两列的调用次数、失败次数、延迟、费用都来自每次 decide 的实际记录(`JevCall`)。"
             f"Jev 超时或报错时该条回退到本地预测并按 `model_low_conf` 标记,"
             f"所以实际调用次数可能多于结果被改写的条数。")
    return "\n".join(L) + "\n"
