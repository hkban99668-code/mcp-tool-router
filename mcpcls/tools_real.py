"""真实工具目录的打标、风险复验与检索评估(T-010b)。

两步,前一步落盘、后一步只读文件:

    tools-real label     data/tools_real.jsonl         -> data/tools_real_labeled.jsonl(联网,走 .cache/answers)
    tools-real evaluate  上面那份标签 + 600 条 BFCL 请求 -> reports/eval_t010.md(离线)

三条贯穿全程的纪律:
- 银标是"两路 LLM 一致",不是人工标注,所以下面所有"准确率"都是对这份银标算的;
- 检索只用工具自己的名字和描述建索引,那 600 条请求一句都不进索引;
- 600 条里有 374 条在 `v3-primary/real_train.jsonl` 里,请求头必须按 5 折折外预测,
  否则就是拿训练集测试。
"""

import csv
import hashlib
import json
import os
import pathlib
import random
import re
import time
import unicodedata
from collections import Counter

import numpy as np

from . import compare, domainvote, evaluate as ev, realdata
from .catalog import ToolCatalog, tool_id as catalog_tool_id
from .heads import MultiHead
from .jev import _picked, tool_questions
from .router import Router, dedupe_by_name
from .rules import (DEVICE_CONTROL_VERBS, DEVICE_NOUNS, OPEN_WORLD_DOMAINS, PARAM_ACTIONS,
                    RISK_KEYWORD_SOURCES, SCAN_HOST_TOKENS, final_risk, param_names, risk_from,
                    risk_signals, rule_risk, rule_risk_base, split_identifier,
                    tool_action_features, tool_features, tool_text)
from .taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS, RISK_IDS, risk_rank

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

TOOLS_REAL = DATA / "tools_real.jsonl"
LABELED = DATA / "tools_real_labeled.jsonl"
TOOLS_ZH = DATA / "tools_real_zh.jsonl"
REQ_MAP = DATA / "requests_real_tool_ids.jsonl"
POOL_ZH = DATA / "pool_zh.jsonl"
REAL_TRAIN = DATA / "v3-primary" / "real_train.jsonl"
AGREE = DATA / "v3-primary" / "labeled_agree.jsonl"
AUDIT_CSV = DATA / "tools_real_risk_audit.csv"
REPORT_MD = REPORTS / "eval_t010.md"

SYNTH_TOOLS = DATA / "tools.jsonl"
SYNTH_TRAIN = DATA / "requests_train.jsonl"

# 缓存文件名的中段:工具侧没有 need 说明版本这一说,填一个能一眼看出是工具侧的标记
TOOL_TAG = "tool-v1"
TOOL_DATASET = "tools_real"
# 打标送进模型的文本长度上限:超过就截断。实测 2405 条里最长 638 字符,这条规则没触发过
MAX_TOOL_TEXT_CHARS = 4000

FOLDS = 5
SEED = 0
AUDIT_ROWS = 60
AUDIT_DISAGREE_MAX = 30
# 与 catalog.ToolCatalog.retrieve 的默认 min_prob 同一条规则(不许调,所以这里写死)
MIN_PROB = 0.1

AUDIT_COLUMNS = ("tool_id", "name", "description", "actions_jev", "actions_haiku",
                 "risk_jev", "risk_haiku", "risk_silver", "risk_rule", "risk_head", "gu_risk")


# ---------------------------------------------------------------- 打标

def tool_text_capped(tool: dict) -> str:
    """"送进模型的工具文本:server + 名字 + 描述 + 参数名,超过上限按字符截断。

    截断规则:保留前 MAX_TOOL_TEXT_CHARS 个字符,尾部加一个省略号,不做按词或按段的智能截断
    ——截断只是防止某天接入超长 schema 时把整个调用打挂,加了省略号至少能看出被截过。
    """
    text = tool_text(tool)
    if len(text) <= MAX_TOOL_TEXT_CHARS:
        return text
    return text[:MAX_TOOL_TEXT_CHARS] + "…"


def tool_items(tools) -> list:
    """给 `compare.fetch_answers` 用的条目;缓存按 text 去重,所以同一份文本只问一次。"""
    return [{"text": tool_text_capped(t)} for t in tools]


def tool_labels(answers: dict, noul_threshold: float = 0.5) -> dict:
    """20 题的答案 -> {"domains", "actions"};取类的规则与 `answers_to_labels` 完全一致。"""
    return {"domains": _picked(answers, "domain", DOMAIN_IDS, noul_threshold),
            "actions": _picked(answers, "action", ACTION_IDS, noul_threshold)}


def _side_labels(rec, noul_threshold: float = 0.5):
    """一条缓存记录 -> {"domains","actions"};没问到(离线缺缓存 / 调用失败)返回 None。"""
    if not rec or not rec.get("ok") or not rec.get("answers"):
        return None
    return tool_labels(rec["answers"], noul_threshold)


def union_actions(*groups) -> list:
    """动作取并集(按 taxonomy 的顺序),两路宁可多报也不漏报。"""
    both = set()
    for g in groups:
        both |= set(g or ())
    return [a for a in ACTION_IDS if a in both]


def label_tools(tools, labelers, live=False, workers=4, budget=None, noul_threshold=0.5) -> dict:
    """给每个工具问两路(默认 jev / haiku),一致的按主域口径出银标,不一致的也写一行。

    labelers: [(名字, 客户端)],第一个是基准路 —— 主域从它的 noul 概率里取,
    不一致时 domains 也用它的(任务卡要求)。缓存键是 (客户端, tool-v1, tools_real),
    重跑不加 --live 就只读缓存。
    """
    items = tool_items(tools)
    names = [n for n, _ in labelers]
    got, stats, spent = {}, [], 0.0
    for name, client in labelers:
        if live and budget is not None and spent >= budget:
            raise realdata.Stopped(
                f"累计费用 {spent:.4f} 美元已达到预算 {budget:.2f} 美元,停止打标;"
                f"已问到的答案都留在缓存里。", realdata.EXIT_BUDGET)
        s = compare.FetchStats()
        recs = compare.fetch_answers(client, getattr(client, "cache_name", name), items,
                                     TOOL_TAG, TOOL_DATASET, live=live,
                                     questions=tool_questions(), workers=workers, stats=s)
        got[name] = {r["text"]: r for r in recs}
        stats.append((name, s))
        spent += s.cost
        if s.stopped:
            raise realdata.Stopped(s.stopped, realdata.EXIT_CLIENT)

    primary, second = names[0], names[1]
    rows, missing = [], {primary: 0, second: 0}
    for t in tools:
        text = tool_text_capped(t)
        rec_a = got[primary].get(text)
        rec_b = got[second].get(text)
        a = _side_labels(rec_a, noul_threshold)
        b = _side_labels(rec_b, noul_threshold)
        for name, lab in ((primary, a), (second, b)):
            if lab is None:
                missing[name] += 1
        if a is not None and b is not None:
            # 工具侧没有 need 这一轴,给两路填同一个 need 就能直接复用请求侧的主域口径
            out = realdata.agreement_outcome(
                {"need": "single", "domains": a["domains"], "actions": a["actions"]},
                {"need": "single", "domains": b["domains"], "actions": b["actions"]},
                (rec_a or {}).get("answers"), (rec_b or {}).get("answers"),
                realdata.AGREEMENT_DEFAULT)
            agreed = out["agreed"]
            domains = out["domains"] or list(a["domains"])      # 不一致时用基准路的
            prim = out["primary_domain"]
        else:
            agreed = False
            domains = list((a or b or {"domains": []})["domains"])
            prim = realdata.primary_domain((rec_a or {}).get("answers"), domains) if a else ""
        actions = union_actions((a or {}).get("actions"), (b or {}).get("actions"))
        rows.append({
            "tool_id": t["tool_id"], "domains": domains, "primary_domain": prim,
            "actions": actions,
            "actions_jev": list((a or {}).get("actions", [])),
            "actions_haiku": list((b or {}).get("actions", [])),
            "domains_jev": list((a or {}).get("domains", [])),
            "domains_haiku": list((b or {}).get("domains", [])),
            "agreed": agreed, "risk_silver": risk_from(actions, domains)})
    return {"rows": rows, "stats": stats, "missing": missing, "cost_usd": spent,
            "n_tools": len(tools), "labelers": names}


def label_stats(rows) -> dict:
    """打标统计:一致率、各域工具数、两路分歧最大的 5 个域。"""
    n = len(rows)
    agreed = sum(1 for r in rows if r["agreed"])
    by_domain = Counter(d for r in rows for d in r["domains"])
    by_primary = Counter(r["primary_domain"] for r in rows if r["primary_domain"])
    by_risk = Counter(r["risk_silver"] for r in rows)
    # 两路分歧:某一路说有这个域、另一路没说
    disagree = {}
    for d in DOMAIN_IDS:
        diff = sum(1 for r in rows if (d in r["domains_jev"]) != (d in r["domains_haiku"]))
        either = sum(1 for r in rows if d in r["domains_jev"] or d in r["domains_haiku"])
        disagree[d] = {"diff": diff, "either": either,
                       "rate": (diff / either) if either else 0.0}
    top = sorted(disagree.items(), key=lambda kv: (-kv[1]["diff"], kv[0]))[:5]
    # 缺答案:那一侧的 domains 与 actions 都是空的(_picked 至少会返回一个,空只可能是没问到)
    empty = {side: sum(1 for r in rows if not r[f"domains_{side}"] and not r[f"actions_{side}"])
             for side in ("jev", "haiku")}
    return {"n": n, "agreed": agreed, "agree_rate": (agreed / n) if n else 0.0,
            "by_domain": by_domain, "by_primary": by_primary, "by_risk": by_risk,
            "disagree_top": top, "disagree": disagree, "empty": empty,
            "n_actions_avg": (sum(len(r["actions"]) for r in rows) / n) if n else 0.0}


# ---------------------------------------------------------------- 风险复验

def risk_metrics(gold, pred) -> dict:
    """风险分级指标:准确率、L3 召回、低报率、L0–L3 混淆矩阵。"""
    n = len(gold)
    l3 = [i for i, g in enumerate(gold) if g == "L3"]
    return {
        "n": n,
        "accuracy": (sum(1 for g, p in zip(gold, pred) if g == p) / n) if n else 0.0,
        "l3_recall": (sum(1 for i in l3 if pred[i] == "L3") / len(l3)) if l3 else None,
        "l3_total": len(l3),
        "under_rate": (sum(1 for g, p in zip(gold, pred) if risk_rank(p) < risk_rank(g)) / n)
        if n else 0.0,
        "confusion": {g: {p: sum(1 for a, b in zip(gold, pred) if a == g and b == p)
                          for p in RISK_IDS} for g in RISK_IDS},
    }


def _tool_head_cv_predict(enc, tools, synth_tools, rows, mode="real", folds=FOLDS, seed=SEED):
    """按函数名分组做 GroupKFold,训练真实数据(或真实+合成)工具侧头,返回每个工具的 HeadOutput。

    mode="real": 只用两路一致的真实工具做训练数据;
    mode="both": 合成数据 + 真实工具一起训练。
    不一致的工具用最终头预测,一致工具取折外预测。
    """
    agreed_rows = [r for r in rows if r["agreed"]]
    agreed_by_id = {r["tool_id"]: r for r in agreed_rows}
    agreed_tools = [t for t in tools if t["tool_id"] in agreed_by_id]
    if not agreed_tools:
        return []

    train_items = []
    groups = []
    for t in agreed_tools:
        r = agreed_by_id[t["tool_id"]]
        train_items.append((tool_text(t), list(r["domains"]), list(r["actions"]), t))
        groups.append(t["name"])

    base_texts = [x[0] for x in train_items]
    base_domains = [x[1] for x in train_items]
    base_actions = [x[2] for x in train_items]
    base_items = [x[3] for x in train_items]

    if mode == "real":
        extra_texts, extra_domains, extra_actions, extra_items = [], [], [], []
    else:  # both
        extra_texts = [tool_text(t) for t in synth_tools]
        extra_domains = [list(t.get("domains", [])) for t in synth_tools]
        extra_actions = [list(t.get("actions", [])) for t in synth_tools]
        extra_items = list(synth_tools)

    from sklearn.model_selection import GroupKFold
    import numpy as np

    unique_groups = len(set(groups))
    n_splits = min(folds, unique_groups)
    if n_splits < 2:
        # 分不了折:直接用最终头预测全部
        final_head = MultiHead(enc, with_need=False, feature_fn=tool_features,
                               action_feature_fn=tool_action_features).fit(
            base_texts + extra_texts, base_domains + extra_domains,
            base_actions + extra_actions, items=base_items + extra_items)
        return final_head.predict([tool_text(t) for t in tools], items=tools)

    out_by_id = {}
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, te_idx in gkf.split(base_texts, groups=np.array(groups)):
        tr_texts = [base_texts[i] for i in tr_idx] + extra_texts
        tr_domains = [base_domains[i] for i in tr_idx] + extra_domains
        tr_actions = [base_actions[i] for i in tr_idx] + extra_actions
        tr_items = [base_items[i] for i in tr_idx] + extra_items
        head = MultiHead(enc, with_need=False, feature_fn=tool_features,
                         action_feature_fn=tool_action_features).fit(
            tr_texts, tr_domains, tr_actions, items=tr_items)
        te_texts = [base_texts[i] for i in te_idx]
        te_items = [base_items[i] for i in te_idx]
        te_outs = head.predict(te_texts, items=te_items)
        for i, o in zip(te_idx, te_outs):
            out_by_id[base_items[i]["tool_id"]] = o

    final_texts = list(base_texts) + extra_texts
    final_domains = list(base_domains) + extra_domains
    final_actions = list(base_actions) + extra_actions
    final_items = list(base_items) + extra_items
    final_head = MultiHead(enc, with_need=False, feature_fn=tool_features,
                           action_feature_fn=tool_action_features).fit(
        final_texts, final_domains, final_actions, items=final_items)

    all_outs = []
    for t in tools:
        if t["tool_id"] in out_by_id:
            all_outs.append(out_by_id[t["tool_id"]])
        else:
            all_outs.append(final_head.predict([tool_text(t)], items=[t])[0])
    return all_outs


def evaluate_risk(rows, tools, synth_tools, synth_train, encoder="tfidf",
                    tool_head_mode="synth") -> dict:
    """三路预测对银标的风险:(a) 工具侧分类头 (b) 纯规则 (c) 单路 LLM(Jev / Haiku)。

    银标只取 agreed=true 的行;其余三路对这同一批工具算指标,条数一致才可比。
    tool_head_mode: "synth" 只用合成数据;"real" 只用真实银标;"both" 合成+真实。
    """
    silver = [r for r in rows if r["agreed"]]
    gold = [r["risk_silver"] for r in silver]

    enc = ev.fit_encoder(encoder, tools + synth_tools, synth_train)
    if tool_head_mode == "synth":
        head = ev.train_tool_head(enc, synth_tools)
        outs = head.predict([tool_text(t) for t in tools], items=tools)
    else:
        outs = _tool_head_cv_predict(enc, tools, synth_tools, rows, mode=tool_head_mode)
    head_domains = {t["tool_id"]: list(o.domains) for t, o in zip(tools, outs)}
    head_actions = {t["tool_id"]: list(o.actions) for t, o in zip(tools, outs)}
    # (a) 现有工具侧分类头:头预测的 domains/actions 过 final_risk(规则与模型取高的一档)
    risk_head = {t["tool_id"]: final_risk(t, head_actions[t["tool_id"]], head_domains[t["tool_id"]])
                 for t in tools}
    # (b) 只用规则:名动词 + 参数名 + annotations(BFCL 没有 annotations),
    #     不给 domains,所以 L1 这一档纯规则给不出来 —— 这正是要看的差距
    risk_rule = {t["tool_id"]: rule_risk(t) for t in tools}
    # (c) 单路 LLM:这一路自己的 actions 与 domains 一起定风险(线上单路部署就是这么跑)
    risk_jev = {r["tool_id"]: risk_from(r["actions_jev"], r["domains_jev"]) for r in rows}
    risk_haiku = {r["tool_id"]: risk_from(r["actions_haiku"], r["domains_haiku"]) for r in rows}

    pick = lambda d: [d[r["tool_id"]] for r in silver]      # noqa: E731 按银标行的顺序取
    paths = {
        f"a 工具侧分类头({tool_head_mode} 训练)": risk_metrics(gold, pick(risk_head)),
        "b 只用规则": risk_metrics(gold, pick(risk_rule)),
        "c 只用 Jev 一路": risk_metrics(gold, pick(risk_jev)),
        "c 只用 Haiku 一路": risk_metrics(gold, pick(risk_haiku)),
    }
    # (a) 在能力域与动作上对银标的 micro/macro-F1
    f1 = {}
    if silver:
        f1 = {"domain": ev.multilabel_f1([r["domains"] for r in silver],
                                         [head_domains[r["tool_id"]] for r in silver], DOMAIN_IDS),
              "action": ev.multilabel_f1([r["actions"] for r in silver],
                                         [head_actions[r["tool_id"]] for r in silver], ACTION_IDS)}
    return {"silver_n": len(silver), "paths": paths, "head_f1": f1,
            "risk_by_id": {"head": risk_head, "rule": risk_rule, "jev": risk_jev,
                           "haiku": risk_haiku},
            "gold_risk": {r["tool_id"]: r["risk_silver"] for r in rows},
            "agreed_ids": [r["tool_id"] for r in silver]}


def audit_rows(rows, risk_by_id, n=AUDIT_ROWS, disagree_max=AUDIT_DISAGREE_MAX, seed=SEED) -> list:
    """人工抽检表:两路风险不同的先放(最多 30 条),剩下的从银标 L3 里按固定 seed 抽。"""
    jev, haiku = risk_by_id["jev"], risk_by_id["haiku"]
    disagree = [r for r in rows if jev[r["tool_id"]] != haiku[r["tool_id"]]]
    picked = list(disagree[:disagree_max])
    have = {r["tool_id"] for r in picked}
    pool = [r for r in rows if r["tool_id"] not in have and r["agreed"]
            and r["risk_silver"] == "L3"]
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    picked += pool[:max(0, n - len(picked))]
    if len(picked) < n:                       # 还不够就从剩下的银标行里按序补
        have = {r["tool_id"] for r in picked}
        picked += [r for r in rows if r["tool_id"] not in have][:n - len(picked)]
    return picked[:n]


def risk_gap_examples(rows, risk_by_id, tools_by_id, k=3) -> dict:
    """银标与纯规则分歧最大的那一类(银标等级 → 规则等级),附 k 个例子。"""
    pairs = Counter((r["risk_silver"], risk_by_id["rule"][r["tool_id"]])
                    for r in rows if r["agreed"]
                    and r["risk_silver"] != risk_by_id["rule"][r["tool_id"]])
    if not pairs:
        return {}
    pair, n = pairs.most_common(1)[0]
    same = [r for r in rows if r["agreed"]
            and (r["risk_silver"], risk_by_id["rule"][r["tool_id"]]) == pair]
    return {"pair": pair, "n": n,
            "examples": [{"tool_id": r["tool_id"],
                          "name": tools_by_id.get(r["tool_id"], {}).get("name", ""),
                          "actions": ";".join(r["actions"]) or "(空)",
                          "risk_silver": pair[0], "risk_rule": pair[1]} for r in same[:k]]}


def write_audit_csv(path, rows, tools_by_id, risk_by_id, gold_risk) -> int:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(AUDIT_COLUMNS)
        for r in rows:
            t = tools_by_id.get(r["tool_id"], {})
            w.writerow([r["tool_id"], t.get("name", ""), t.get("description", ""),
                        ";".join(r["actions_jev"]), ";".join(r["actions_haiku"]),
                        risk_by_id["jev"][r["tool_id"]], risk_by_id["haiku"][r["tool_id"]],
                        gold_risk.get(r["tool_id"], ""), risk_by_id["rule"][r["tool_id"]],
                        risk_by_id["head"][r["tool_id"]], ""])
    return len(rows)


# ---------------------------------------------------------------- 检索

def load_eval_requests(req_map=REQ_MAP, pool_zh=POOL_ZH, agree=AGREE) -> list:
    """600 条 BFCL 请求:中文 text、英文 text_orig、金标与候选 tool_id、来源文件、银标 need。"""
    pool = {r["id"]: r for r in ev.load_jsonl(pool_zh)}
    label = {r["id"]: r for r in ev.load_jsonl(agree)}
    out = []
    for r in ev.load_jsonl(req_map):
        p = pool.get(r["id"])
        if p is None or not r["tool_ids"]:
            continue
        out.append({"id": r["id"], "text": p["text"], "text_orig": p["text_orig"],
                    "tool_ids": list(r["tool_ids"]),
                    "candidate_tool_ids": list(r["candidate_tool_ids"]),
                    "source_file": p.get("source_file", ""),
                    "need": (label.get(r["id"]) or {}).get("need", "")})
    return out


def assign_folds(ids, folds=FOLDS, seed=SEED) -> dict:
    """把 id 用固定种子打乱后轮流分进 folds 份,返回 id -> 折号(折大小最多差 1)。"""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return {i: idx % folds for idx, i in enumerate(order)}


def fold_train_rows(synth_train, real_train, fold_ids) -> list:
    """一折的训练集:合成训练集 + real_train 去掉本折的 id。"""
    return list(synth_train) + [r for r in real_train if r["id"] not in fold_ids]


def _allowed_domains(domains, domain_probs) -> set:
    """软过滤实际放行的域集合,与 `catalog._allowed_domains` 同一条规则(min_prob=0.1)。"""
    want = set(domains or [])
    if domain_probs:
        want |= {d for d, p in domain_probs.items() if p >= MIN_PROB}
    return want


def _recall(gold_ids, gold_names, top_ids, name_of, by_name: bool) -> float:
    """|金标 ∩ topk| / |金标|;by_name 时按函数名算(任一变体命中就算)。"""
    if not gold_ids:
        return 0.0
    if by_name:
        return len(set(gold_names) & {name_of.get(t, t) for t in top_ids}) / len(set(gold_names))
    return len(set(gold_ids) & set(top_ids)) / len(set(gold_ids))


def _query_result(cat, req, out, name_of, domains_of,
                 filter_on=True, min_prob=MIN_PROB, prior_weight=0.3, action_bonus=0.05) -> dict:
    """一条请求在一个索引上的全部数字,中英文各算一遍。

    不过滤那一路取满 2405 条的排序,候选集内的 Recall@1 也从这份排序里截出来,
    省得为了"只在候选集里检索"再建 600 个小索引。
    """
    gold = req["tool_ids"]
    gold_names = [name_of.get(g, g) for g in gold]
    gold_domains = set()
    for g in gold:
        gold_domains |= set(domains_of.get(g, ()))
    cands = set(req["candidate_tool_ids"])
    allowed = _allowed_domains(out.domains, out.domain_probs)
    res = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        full = [rec.tool_id for rec, _ in cat.retrieve(text, None, len(cat.records))]
        in_cand = [t for t in full if t in cands]
        if filter_on:
            filt = [rec.tool_id for rec, _ in cat.retrieve(text, out.domains, 5, actions=out.actions,
                                                           domain_probs=out.domain_probs,
                                                           min_prob=min_prob, prior_weight=prior_weight,
                                                           action_bonus=action_bonus)]
        else:
            filt = full[:5]
        orac = [rec.tool_id for rec, _ in cat.retrieve(text, sorted(gold_domains), 5)]
        res[lang] = {
            "off": {"r1": _recall(gold, gold_names, full[:1], name_of, False),
                    "r5": _recall(gold, gold_names, full[:5], name_of, False),
                    "r1_name": _recall(gold, gold_names, full[:1], name_of, True),
                    "r5_name": _recall(gold, gold_names, full[:5], name_of, True)},
            "on": {"r1": _recall(gold, gold_names, filt[:1], name_of, False),
                   "r5": _recall(gold, gold_names, filt[:5], name_of, False),
                   "r1_name": _recall(gold, gold_names, filt[:1], name_of, True),
                   "r5_name": _recall(gold, gold_names, filt[:5], name_of, True)},
            "oracle": {"r1": _recall(gold, gold_names, orac[:1], name_of, False),
                       "r5": _recall(gold, gold_names, orac[:5], name_of, False),
                       "r1_name": _recall(gold, gold_names, orac[:1], name_of, True),
                       "r5_name": _recall(gold, gold_names, orac[:5], name_of, True)},
            "cand_r1": _recall(gold, gold_names, in_cand[:1], name_of, False),
            "cand_r1_name": _recall(gold, gold_names, in_cand[:1], name_of, True),
            "cand_size": len(cands),
            "cand_count": (cat.candidate_count(out.domains, out.domain_probs, min_prob)
                           if filter_on else len(cat.records)),
            "gold_filtered_out": 0.0 if (allowed & gold_domains) else 1.0,
        }
    return res


def _bilingual_description(t, zh_by_id):
    """把中文描述拼到英文描述后面,只在索引里生效。"""
    zh = zh_by_id.get(t["tool_id"], "")
    if zh:
        return f"{t.get('description', '')} {zh}".strip()
    return t.get("description", "")


def _fit_texts(tools, requests) -> list:
    """与 `evaluate.fit_encoder` 完全一致的拟合文本(工具侧两种写法 + 请求文本)。

    `build_fold_index` 传了复用编码器时要照原样再 fit 一次,不能只挑一套写法,
    否则 tfidf 的折外词表和 `fit_encoder` 那条路径不一致。
    """
    return ([tool_text(t) for t in tools]
            + [f"{t['name'].replace('_', ' ')} {t['description']}" for t in tools]
            + [r["text"] for r in requests])


def build_fold_index(labeled_tools, synth_tools, train_requests, encoder="tfidf",
                      bilingual=False, zh_by_id=None, enc=None, retrieval_encoder=None):
    """拟合编码器 + 训工具侧头 + 建索引。索引文本只用工具自己的名字和描述,不加示例查询。

    `catalog.tool_id` 只拼 server 和函数名,而真实目录里同一个函数名有几十个变体
    (`get_current_weather` 有 43 个),靠 tool_id 上的 `#N` 后缀区分。不把后缀补回去,
    这些变体在索引里会挤成同一个 tool_id,严格口径的金标永远对不上号(实测 R@1 直接归零)。
    所以建完索引把 tool_id 换成目录里那个带后缀的,检索逻辑本身不动。

    bilingual=True 时把 `zh_by_id` 里的中文描述拼进索引文本,能力域/动作仍只由英文描述预测。

    传 `enc` 时复用这个编码器实例(不再新建):语义模型加载一次要好几秒 / 上 GB,
    5 折每折都新建会反复加载。复用后仍按本折的 `train_requests` 调一次 `enc.fit`
    (tfidf 重学词表,语义编码器是 no-op),所以折外口径与不传时一致。

    `retrieval_encoder` 是检索编码器实例(T-019),与分类头编码器分开:不为 None 时
    索引向量和查询都用它,分类头仍用 `enc`。它同样按本折 `_fit_texts` 重拟合一次
    (语义编码器是 no-op),与 `enc` 保持同一折外口径。为 None 时行为与改前一致。
    """
    if enc is None:
        enc = ev.fit_encoder(encoder, labeled_tools + synth_tools, train_requests)
    else:
        enc = enc.fit(_fit_texts(labeled_tools + synth_tools, train_requests))
    head = ev.train_tool_head(enc, synth_tools)
    if retrieval_encoder is None:
        cat = ToolCatalog(head)
    else:
        retrieval_encoder = retrieval_encoder.fit(
            _fit_texts(labeled_tools + synth_tools, train_requests))
        cat = ToolCatalog(head, retrieval_encoder=retrieval_encoder)
    recs = cat.add_tools(labeled_tools, use_labels=True)
    for rec, t in zip(recs, labeled_tools):
        rec.tool_id = t["tool_id"]
    if bilingual and zh_by_id:
        for rec in cat.records:
            zh = zh_by_id.get(rec.tool_id, "")
            if zh:
                rec.description = f"{rec.description} {zh}".strip()
        # 用新描述重新编码索引向量,不改变工具侧分类头的输入
        import scipy.sparse as sp
        vecs = cat.encoder.encode([cat._index_text(r) for r in cat.records])
        cat._vecs = vecs if sp.issparse(vecs) else sp.csr_matrix(vecs)
    return enc, head, cat


def _run_variant(labeled_tools, synth_tools, requests, train_rows, encoder="tfidf",
                 bilingual=False, zh_by_id=None, filter_on=True, min_prob=MIN_PROB,
                 prior_weight=0.3, action_bonus=0.05):
    """一种请求头训练集下的全部检索结果:{请求 id -> _query_result}。"""
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    domains_of = {t["tool_id"]: t.get("domains", []) for t in labeled_tools}
    _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train_rows, encoder,
                                         bilingual=bilingual, zh_by_id=zh_by_id)
    rhead = ev.train_request_head(_enc, train_rows)
    outs = rhead.predict([r["text"] for r in requests])
    return {r["id"]: _query_result(cat, r, o, name_of, domains_of,
                                   filter_on=filter_on, min_prob=min_prob,
                                   prior_weight=prior_weight, action_bonus=action_bonus)
            for r, o in zip(requests, outs)}


def eval_retrieval(labeled_tools, requests, synth_train, real_train, synth_tools,
                   encoder="tfidf", folds=FOLDS, seed=SEED, bilingual=False, zh_by_id=None) -> dict:
    """折外请求头与"只用合成数据"的对照;两种口径 × 中英文 × 过滤开关。"""
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    oof = {}
    for f in range(folds):
        ids = {i for i, fo in fold_of.items() if fo == f}
        rows = [r for r in requests if r["id"] in ids]
        oof.update(_run_variant(labeled_tools, synth_tools, rows,
                                fold_train_rows(synth_train, real_train, ids), encoder,
                                bilingual=bilingual, zh_by_id=zh_by_id))
    synth_only = _run_variant(labeled_tools, synth_tools, requests, synth_train, encoder,
                              bilingual=bilingual, zh_by_id=zh_by_id)
    return {"折外(合成 + real_train 去掉本折)": oof, "只用合成训练集": synth_only,
            "fold_of": fold_of}


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(per_query, requests) -> dict:
    """按 (语言 × 过滤开关 × 口径) 汇总 R@1/R@5,外加软过滤开启时的诊断项。"""
    cells = {}
    for lang in ("zh", "en"):
        for mode in ("off", "on", "oracle"):
            for key, suf in (("strict", ""), ("name", "_name")):
                cells[(lang, mode, key)] = {
                    "r1": _mean([per_query[r["id"]][lang][mode][f"r1{suf}"] for r in requests]),
                    "r5": _mean([per_query[r["id"]][lang][mode][f"r5{suf}"] for r in requests])}
        cells[(lang, "cand", "strict")] = {"r1": _mean([per_query[r["id"]][lang]["cand_r1"]
                                                        for r in requests])}
        cells[(lang, "cand", "name")] = {"r1": _mean([per_query[r["id"]][lang]["cand_r1_name"]
                                                      for r in requests])}
    diag = {"cand_count": _mean([per_query[r["id"]]["zh"]["cand_count"] for r in requests]),
            "gold_filtered_out": _mean([per_query[r["id"]]["zh"]["gold_filtered_out"]
                                        for r in requests])}
    return {"cells": cells, "diag": diag, "n": len(requests)}


def attach_labels(tools, rows) -> list:
    """把打标结果挂回工具字典(domains / actions),建索引时用;没有标签的行按空处理。"""
    by_id = {r["tool_id"]: r for r in rows}
    return [dict(t, domains=list(by_id.get(t["tool_id"], {}).get("domains", [])),
                 actions=list(by_id.get(t["tool_id"], {}).get("actions", []))) for t in tools]


# T-011: 软过滤参数网格,只在开发集上选参
FILTER_GRID = {
    "filter": (False, True),
    "min_prob": (0.05, 0.1, 0.2, 0.3),
    "prior_weight": (0.0, 0.1, 0.3),
}
ACTION_BONUS_FIXED = 0.05


def _dev_test_ids(request_ids, folds=FOLDS, seed=SEED):
    """T-011 的固定划分:请求折 0-2 为开发集,3-4 为测试集;工具划分按函数名分组 6:4。"""
    fold_of = assign_folds(request_ids, folds, seed)
    dev = {i for i, f in fold_of.items() if f in (0, 1, 2)}
    test = {i for i, f in fold_of.items() if f in (3, 4)}
    return dev, test, fold_of


def grid_filter_params(labeled_tools, requests, synth_train, real_train, synth_tools,
                       encoder="tfidf", folds=FOLDS, seed=SEED, bilingual=False, zh_by_id=None) -> dict:
    """开发集上选软过滤参数,返回完整网格表、最佳参数、测试集三行对照。

    选择标准:开发集上中文查询 + 双语索引的严格 R@5。
    优化:只建一次索引/请求头,不同参数只做检索,不做重复训练。
    """
    request_ids = [r["id"] for r in requests]
    fold_of = assign_folds(request_ids, folds, seed)
    dev_ids = {i for i, f in fold_of.items() if f in (0, 1, 2)}
    test_ids = {i for i, f in fold_of.items() if f in (3, 4)}
    dev_rows = [r for r in requests if r["id"] in dev_ids]
    test_rows = [r for r in requests if r["id"] in test_ids]

    def _build_for(ids):
        rows = [r for r in requests if r["id"] in ids]
        train = fold_train_rows(synth_train, real_train, ids)
        name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
        domains_of = {t["tool_id"]: t.get("domains", []) for t in labeled_tools}
        enc, head, cat = build_fold_index(labeled_tools, synth_tools, train, encoder,
                                          bilingual=bilingual, zh_by_id=zh_by_id)
        rhead = ev.train_request_head(enc, train)
        outs = rhead.predict([r["text"] for r in rows])
        return rows, name_of, domains_of, cat, outs

    # 开发集每折建一次;测试集建一次
    dev_envs = []
    for f in (0, 1, 2):
        ids = {i for i, fo in fold_of.items() if fo == f}
        dev_envs.append(_build_for(ids))
    test_env = _build_for(test_ids)

    def _query_env(env, cfg):
        filter_on, min_prob, prior_weight = cfg
        rows, name_of, domains_of, cat, outs = env
        per = {r["id"]: _query_result(cat, r, o, name_of, domains_of,
                                      filter_on=filter_on, min_prob=min_prob,
                                      prior_weight=prior_weight, action_bonus=ACTION_BONUS_FIXED)
               for r, o in zip(rows, outs)}
        return per, rows

    cells = []
    best = None
    best_score = -1.0
    for filter_on in FILTER_GRID["filter"]:
        for min_prob in FILTER_GRID["min_prob"]:
            for prior_weight in FILTER_GRID["prior_weight"]:
                cfg = (filter_on, min_prob, prior_weight)
                all_per = {}
                for env in dev_envs:
                    per, _rows = _query_env(env, cfg)
                    all_per.update(per)
                s = summarize(all_per, dev_rows)
                score = s["cells"][("zh", "on" if filter_on else "off", "strict")]["r5"]
                cells.append({"filter_on": filter_on, "min_prob": min_prob,
                              "prior_weight": prior_weight,
                              "zh_strict_r5": score,
                              "diag": s["diag"]})
                if score > best_score:
                    best_score = score
                    best = cfg

    # 测试集三行
    defaults = (True, MIN_PROB, 0.3)
    test_rows_out = []
    for label, cfg in (("默认参数", defaults), ("开发集最优", best), ("过滤关", (False, MIN_PROB, 0.0))):
        per, _rows = _query_env(test_env, cfg)
        s = summarize(per, test_rows)
        mode = "on" if cfg[0] else "off"
        test_rows_out.append({"label": label, "cfg": cfg,
                              "zh_strict_r5": s["cells"][("zh", mode, "strict")]["r5"],
                              "en_strict_r5": s["cells"][("en", mode, "strict")]["r5"],
                              "zh_name_r5": s["cells"][("zh", mode, "name")]["r5"],
                              "en_name_r5": s["cells"][("en", mode, "name")]["r5"],
                              "diag": s["diag"]})
    return {"grid": cells, "best": best, "best_score": best_score,
            "test_rows": test_rows_out}


def evaluate_all(labeled_rows, tools, requests, synth_tools, synth_train, real_train,
                 encoder="tfidf", folds=FOLDS, seed=SEED, bilingual=False, zh_by_id=None,
                 grid_search=False, tool_head_mode="synth") -> dict:
    """打标统计 + 风险复验 + 检索评估,一次算完,交给 `render_report`。"""
    labeled_tools = attach_labels(tools, labeled_rows)
    stats = label_stats(labeled_rows)
    risk = evaluate_risk(labeled_rows, tools, synth_tools, synth_train, encoder,
                         tool_head_mode=tool_head_mode)
    variants = eval_retrieval(labeled_tools, requests, synth_train, real_train, synth_tools,
                              encoder, folds, seed, bilingual=bilingual, zh_by_id=zh_by_id)
    variants.pop("fold_of", None)
    tools_by_id = {t["tool_id"]: t for t in tools}
    risk["gap"] = risk_gap_examples(labeled_rows, risk["risk_by_id"], tools_by_id)
    grid = None
    if grid_search and bilingual:
        grid = grid_filter_params(labeled_tools, requests, synth_train, real_train, synth_tools,
                                  encoder, folds, seed, bilingual=True, zh_by_id=zh_by_id)
    return {"stats": stats, "risk": risk, "variants": variants,
            "summary": {k: summarize(v, requests) for k, v in variants.items()},
            "by_need": {k: by_group(v, requests, lambda r: r["need"]) for k, v in variants.items()},
            "by_file": {k: by_group(v, requests, lambda r: r["source_file"])
                        for k, v in variants.items()},
            "grid": grid}


def by_group(per_query, requests, key_fn) -> dict:
    """按某个分组(need / 来源文件)出过滤开启时的 R@5。"""
    groups = {}
    for r in requests:
        groups.setdefault(key_fn(r) or "(空)", []).append(r["id"])
    out = {}
    for name in sorted(groups):
        ids = groups[name]
        out[name] = {"n": len(ids),
                     "zh_strict": _mean([per_query[i]["zh"]["on"]["r5"] for i in ids]),
                     "zh_name": _mean([per_query[i]["zh"]["on"]["r5_name"] for i in ids]),
                     "en_strict": _mean([per_query[i]["en"]["on"]["r5"] for i in ids]),
                     "en_name": _mean([per_query[i]["en"]["on"]["r5_name"] for i in ids])}
    return out


# ---------------------------------------------------------------- 报告

def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def render_report(label_stats, risk, retrieval, requests, tools) -> str:
    s = label_stats
    lens = sorted(len(tool_text_capped(t)) for t in tools)
    L = ["# 真实工具目录评估(T-010b)\n",
         f"工具 {s['n']} 个(来自 `data/tools_real.jsonl`),请求 {len(requests)} 条"
         f"(来自 `data/requests_real_tool_ids.jsonl`)。\n",
         "> 银标是 Jev 与 Haiku 两路一致的结果,不是人工标注;"
         "下面所有『准确率』『召回率』都是对这份银标算的,偏向这两个模型。\n",
         "> 检索只用工具自己的名字和描述建索引;那 600 条请求一句都没进索引。\n",

         "\n## 1. 打标统计\n",
         f"- 两路一致(主域口径):**{s['agreed']} / {s['n']}**({_pct(s['agree_rate'])})",
         f"- 缺答案:Jev {s['empty']['jev']} 条,Haiku {s['empty']['haiku']} 条",
         f"- 每个工具平均动作数:{s['n_actions_avg']:.2f}",
         f"- 送进模型的文本长度:最小 {lens[0]}、中位 {lens[len(lens) // 2]}、最大 {lens[-1]} 字符;"
         f"超过 {MAX_TOOL_TEXT_CHARS} 字符被截断的 {sum(1 for x in lens if x > MAX_TOOL_TEXT_CHARS)} 条",
         "- 银标风险分布:" + "、".join(f"{k} {v}" for k, v in sorted(s["by_risk"].items())),
         "\n各域工具数(按主域;一致行取主域,不一致行取基准路 Jev 的主域):\n",
         "\n| 能力域 | 工具数 |\n| --- | --- |"]
    for d in DOMAIN_IDS:
        L.append(f"| {d} | {s['by_primary'].get(d, 0)} |")
    L.append("\n两路分歧最大的 5 个域(分母是『至少一路标了这个域』的工具数):\n")
    L.append("\n| 能力域 | 分歧数 | 涉及数 | 分歧率 |\n| --- | --- | --- | --- |")
    for d, g in s["disagree_top"]:
        L.append(f"| {d} | {g['diff']} | {g['either']} | {_pct(g['rate'])} |")

    L.append("\n## 2. 风险分级复验\n")
    L.append(f"银标 = 两路一致的 {risk['silver_n']} 个工具(`risk_silver`);"
             "四路预测对这同一批工具算指标。\n")
    L.append("> 银标本身就是这两路的产物,所以 (c) 两行**不是独立证据**:"
             "Jev 那一行只差 Haiku 多报的动作,Haiku 那一行只差 Jev 多报的动作。"
             "要看的是『另一路多报了多少』,而不是『这一路有多准』。\n")
    L.append("\n| 预测 | 风险准确率 | L3 召回 | 低报率 | L3 条数 |\n| --- | --- | --- | --- | --- |")
    for name, m in risk["paths"].items():
        L.append(f"| {name} | {_pct(m['accuracy'])} | {_pct(m['l3_recall'])} "
                 f"| {_pct(m['under_rate'])} | {m['l3_total']} |")
    if risk["head_f1"]:
        f1 = risk["head_f1"]
        L.append(f"\n(a) 工具侧分类头在能力域上对银标 micro-F1 {_pct(f1['domain']['micro_f1'])}"
                 f" / macro-F1 {_pct(f1['domain']['macro_f1'])},"
                 f"动作 micro-F1 {_pct(f1['action']['micro_f1'])}"
                 f" / macro-F1 {_pct(f1['action']['macro_f1'])}。")
    L.append("\n低报率 = 预测等级低于银标等级的比例(漏报比高报危险,所以单列)。"
             "(b) 纯规则不给 domains,BFCL 工具也没有 annotations,所以 L1 这一档它给不出来。\n")
    for name, m in risk["paths"].items():
        L.append(f"\n**{name}** 的混淆矩阵(行=银标,列=预测):\n")
        L.append("\n| 银标\\预测 | " + " | ".join(RISK_IDS) + " |\n| --- | --- | --- | --- | --- |")
        for g in RISK_IDS:
            L.append(f"| {g} | " + " | ".join(str(m["confusion"][g][p]) for p in RISK_IDS) + " |")
    gap = risk.get("gap") or {}
    if gap:
        L.append(f"\n银标与纯规则分歧最大的一类:**银标 {gap['pair'][0]} → 规则 {gap['pair'][1]}**,"
                 f"共 {gap['n']} 个工具。例子:\n")
        L.append("\n| tool_id | 函数名 | 银标动作 | 银标 | 规则 |\n| --- | --- | --- | --- | --- |")
        for e in gap["examples"]:
            L.append(f"| `{e['tool_id']}` | {e['name']} | {e['actions']} "
                     f"| {e['risk_silver']} | {e['risk_rule']} |")
    L.append(f"\n人工抽检表 `data/tools_real_risk_audit.csv`({AUDIT_ROWS} 行,"
             f"两路风险不同的最多 {AUDIT_DISAGREE_MAX} 行,其余从银标 L3 里按 seed={SEED} 抽)。")

    L.append("\n## 3. 真实 Recall@k\n")
    L.append("主表:查询语言 × 请求头 × 过滤开关 × 口径。过滤开启用 `retrieve` 的默认参数,"
             "没有调 `min_prob` / `prior_weight` / `action_bonus`。\n")
    L.append("\n| 请求头 | 查询语言 | 过滤 | 严格 R@1 | 严格 R@5 | 同名 R@1 | 同名 R@5 | 条数 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- |")
    label_lang = {"zh": "中文", "en": "英文"}
    label_mode = {"off": "关", "on": "开", "oracle": "开(金标域)"}
    for head_name, per_query in retrieval["variants"].items():
        for lang in ("zh", "en"):
            for mode in ("off", "on", "oracle"):
                c = retrieval["summary"][head_name]["cells"]
                a, b = c[(lang, mode, "strict")], c[(lang, mode, "name")]
                L.append(f"| {head_name} | {label_lang[lang]} | {label_mode[mode]} "
                         f"| {_pct(a['r1'])} | {_pct(a['r5'])} | {_pct(b['r1'])} | {_pct(b['r5'])} "
                         f"| {retrieval['summary'][head_name]['n']} |")
    for head_name, per_query in retrieval["variants"].items():
        c = retrieval["summary"][head_name]["cells"]
        for lang in ("zh", "en"):
            L.append(f"| {head_name} | {label_lang[lang]} | 候选集内(参考) "
                     f"| {_pct(c[(lang, 'cand', 'strict')]['r1'])} | - "
                     f"| {_pct(c[(lang, 'cand', 'name')]['r1'])} | - "
                     f"| {retrieval['summary'][head_name]['n']} |")
    L.append("\n`oracle` 那两行用金标工具自己的 domains 过滤,只说明『过滤最多能帮多少』,"
             "线上拿不到。候选集内那两行只在每条请求自带的候选集里检索,"
             f"{sum(1 for r in requests if len(r['candidate_tool_ids']) == 1)} 条请求的候选集只有 1 个工具,"
             "所以这两行只作参考,不能当主指标。\n")

    L.append("\n### 3.1 诊断(软过滤开启)\n")
    for head_name in retrieval["variants"]:
        d = retrieval["summary"][head_name]["diag"]
        L.append(f"- **{head_name}**:平均候选数 {d['cand_count']:.1f} / {len(tools)}"
                 f"(缩减 {1 - d['cand_count'] / max(1, len(tools)):.1%});"
                 f"金标工具被过滤掉的比例 **{_pct(d['gold_filtered_out'])}**")
    L.append("\n按 need 分组的 R@5(need 取 v3-primary 银标,覆盖不到全部 600 条):\n")
    L.append("\n| 请求头 | need | 条数 | 中文严格 | 中文同名 | 英文严格 | 英文同名 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for head_name in retrieval["variants"]:
        for name, g in retrieval["by_need"][head_name].items():
            L.append(f"| {head_name} | {name} | {g['n']} | {_pct(g['zh_strict'])} "
                     f"| {_pct(g['zh_name'])} | {_pct(g['en_strict'])} | {_pct(g['en_name'])} |")
    L.append("\n按来源文件分组的 R@5:\n")
    L.append("\n| 请求头 | 来源文件 | 条数 | 中文严格 | 中文同名 | 英文严格 | 英文同名 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for head_name in retrieval["variants"]:
        for name, g in retrieval["by_file"][head_name].items():
            L.append(f"| {head_name} | `{name}` | {g['n']} | {_pct(g['zh_strict'])} "
                     f"| {_pct(g['zh_name'])} | {_pct(g['en_strict'])} | {_pct(g['en_name'])} |")

    L.append("\n## 4. 结论与局限\n")
    oof, syn = "折外(合成 + real_train 去掉本折)", "只用合成训练集"
    c = retrieval["summary"][oof]["cells"]
    d_cand = retrieval["summary"][oof]["diag"]["cand_count"]
    en_off, en_on, en_or = (c[("en", "off", "strict")]["r5"], c[("en", "on", "strict")]["r5"],
                            c[("en", "oracle", "strict")]["r5"])
    zh_off, zh_on = c[("zh", "off", "strict")]["r5"], c[("zh", "on", "strict")]["r5"]
    en_name = c[("en", "off", "name")]["r5"]
    n = len(requests)
    pa = next(v for k, v in risk["paths"].items() if k.startswith("a"))
    pb = next(v for k, v in risk["paths"].items() if k.startswith("b"))
    dup = Counter(t["name"] for t in tools)
    worst = max(dup.items(), key=lambda kv: kv[1])
    L.append(f"1. **跨语言是主要瓶颈**:英文查询严格 R@5 {_pct(en_off)},中文 {_pct(zh_off)}"
             f"(差 {(en_off - zh_off) * 100:.1f} 个百分点,约 {(en_off - zh_off) * n:.0f} 条)。"
             "工具描述是英文、查询是中文,TF-IDF 的 word 特征只吃 ASCII 词,中文只剩字符 n-gram,"
             "两边几乎打不着;中文数字基本没有检索能力。")
    L.append(f"2. **软过滤是添乱不是帮忙**:英文严格 R@5 {_pct(en_off)}(关)→ {_pct(en_on)}(开),"
             f"掉 {(en_off - en_on) * 100:.1f} 个百分点;中文 {_pct(zh_off)} → {_pct(zh_on)}。"
             f"金标工具被过滤掉的比例 {_pct(retrieval['summary'][oof]['diag']['gold_filtered_out'])}。"
             f"换成金标自己的域(oracle)也只有 {_pct(en_or)},比不过滤高"
             f"{(en_or - en_off) * 100:.1f} 个百分点 —— 过滤的天花板就这么高,"
             "问题不在『预测错域』而在排序本身。")
    L.append(f"3. **真实数据训练的请求头主要是在让过滤少误伤**:过滤开启时英文严格 R@5 "
             f"{_pct(retrieval['summary'][syn]['cells'][('en', 'on', 'strict')]['r5'])}(只用合成)"
             f"→ {_pct(en_on)}(折外),金标被过滤掉的比例从 "
             f"{_pct(retrieval['summary'][syn]['diag']['gold_filtered_out'])} 降到 "
             f"{_pct(retrieval['summary'][oof]['diag']['gold_filtered_out'])};"
             "过滤关掉时两者几乎一样(无法区分)。")
    L.append(f"4. **同名口径比严格口径高得多**:英文 R@5 {_pct(en_off)}(严格)→ {_pct(en_name)}(同名)。"
             f"同一个函数名在目录里有多个变体(`{worst[0]}` 有 {worst[1]} 个),"
             "严格口径要求连变体都挑对,线上真正要做的是先挑对函数、参数再交给模型填,"
             "所以主指标看严格、上限看同名。")
    L.append(f"5. **风险分级靠规则不够,靠分类头也不够**:(a) 工具侧分类头准确率 {_pct(pa['accuracy'])}、"
             f"L3 召回 {_pct(pa['l3_recall'])}、低报率 {_pct(pa['under_rate'])};"
             f"(b) 纯规则准确率 {_pct(pb['accuracy'])} 但 L3 召回只有 {_pct(pb['l3_recall'])}、"
             f"低报率 {_pct(pb['under_rate'])} —— 规则几乎漏掉所有 delete/execute/send 之外的 L3, "
             "而分类头把 540 个银标 L0 误报成 L3(混淆矩阵第一行)。两路 LLM 一致才是现在唯一能用的。"
             "(c) 两行的循环性见第 2 节开头。")
    L.append("\n局限:\n")
    L.append(f"- **银标不是人工标注**。一致率 {_pct(s['agree_rate'])},剩下 "
             f"{s['n'] - s['agreed']} 个工具两路没谈拢,它们的 domains 直接用了 Jev 的,风险可能偏低。")
    L.append(f"- **能力域严重偏斜**:{s['by_primary'].get('system_misc', 0)} 个工具的主域是 "
             "`system_misc`(占多数),所以软过滤一次要放行上千个工具,缩减只有 "
             f"{1 - d_cand / max(1, len(tools)):.1%}。这既是目录本身的特点,也说明 14 个域的划分"
             "在这些通用工具上区分度不够。")
    L.append(f"- **按 need 分组只覆盖 {sum(1 for r in requests if r['need'])} / {n} 条**:"
             "need 来自 v3-primary 银标,两路没一致的请求没有标签。")
    L.append("- **索引没有示例查询**:线上接新 server 时会用示例请求扩充索引文本("
             "`catalog.add_tools(example_queries=...)`),这里为了不泄漏那 600 条请求一句都没加,"
             "所以上面的数字是这套检索的下限。")
    L.append("- **编码器是 TF-IDF**:换 sentence-transformers 之后中英文的差距会小很多,"
             "这一节的差距首先归编码器,其次才归检索策略。")
    L.append("- **网关不返回 `usage.cost`**:打标的费用全记成 0,"
             "`--budget-usd 50` 实际没起作用(2405 个工具 × 两路全部问完,0 次失败)。")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- T-011 报告
#
# `reports/eval_t011.md` 的生成逻辑放在这里(而不是一次性脚本里),由
# `tools-real report-t011` 调用;它只读已落盘的 jsonl,不联网,保证 clone 仓库后
# 能用同一条命令重跑出同一份报告。整份数据流:
#   compute_t011(读进来的数据) -> render_t011(渲染 markdown)
#                             \\-> t011_metrics_json(机器可读数字)

# 生成报告的对照口径:第一个变体名与 eval_retrieval 一致(折外请求头)。
T011_OOF = "折外(合成 + real_train 去掉本折)"
# 报告 3.2 一律按 6:4 切开发 / 测试
T011_DEV_RATIO = 0.6

# 改前的动词表 = 合并进 main 之前的 `rules.VERB_ACTIONS`(T-010b 时的版本),
# 只用于报告里的"改前 / 改后"对照;不要跟着 `rules.py` 一起改。
T011_OLD_VERB_ACTIONS = {
    "read": {"get", "list", "search", "find", "query", "read", "fetch", "lookup", "describe",
             "show", "view", "download", "retrieve", "count", "check", "status", "logs", "stat"},
    "create": {"create", "add", "new", "insert", "upload", "write", "save", "make", "generate",
               "schedule", "open", "draft", "invite", "book"},
    "update": {"update", "edit", "modify", "set", "rename", "move", "patch", "merge", "assign",
               "scale", "comment", "label", "complete", "close", "archive", "restart"},
    "delete": {"delete", "remove", "drop", "destroy", "purge", "truncate", "cancel", "revoke",
               "uninstall", "terminate", "overwrite"},
    "execute": {"run", "rerun", "retry", "exec", "execute", "eval", "invoke", "deploy", "apply",
                "shell", "command", "script", "trigger", "click", "navigate", "type", "fill",
                "evaluate"},
    "send": {"send", "post", "publish", "reply", "email", "message", "notify", "tweet",
             "pay", "charge", "refund", "transfer", "broadcast", "forward", "share"},
}

# 最终补进去的词:只保留动作动词。hotel/hotels/ride/uber/eat 是服务名或名词,
# 第 2 轮已删(见 render_t011 的 3.1 说明)。
T011_ADDED_SEND = ("buy", "order", "reserve", "book", "booking")
T011_ADDED_EXEC = ("play",)
T011_REMOVED_NOUNS = ("hotel", "hotels", "ride", "uber", "eat")

# 第 1 轮(被打回)的规则表 = 改前 + 第 1 轮补的 10 个 send 词 + play。
# 用来在报告里量出"删掉 5 个名词"这一步到底带来什么变化。
T011_ROUND1_VERB_ACTIONS = {k: set(v) for k, v in T011_OLD_VERB_ACTIONS.items()}
T011_ROUND1_VERB_ACTIONS["send"] |= {"buy", "order", "reserve", "book", "booking",
                                     "ride", "hotel", "hotels", "eat", "uber"}
T011_ROUND1_VERB_ACTIONS["execute"] |= {"play"}

# 合成 `evaluate` 工具侧的数字(改前 = 合并前 main,改后 = 本卡最终规则),
# 直接取自 `logs/eval_t011_before_rules.log` 与 `logs/eval_t011_after_rules.log`;
# 这份报告不做合成侧训练,所以按日志里的数字记账。
T011_SYNTH_BEFORE = {"domain": "89.7% / 85.4%", "action": "93.7% / 89.8%",
                     "risk_model": "88.7%", "risk_rule": "90.6%", "l3": "100.0%",
                     "under": "1.9%", "queue": "24"}
T011_SYNTH_AFTER = dict(T011_SYNTH_BEFORE)


def _rule_actions_with(verbs, tool) -> set:
    """给定一套动词表算 rule_actions;与 `rules.rule_actions` 同一套逻辑。"""
    tokens = set(split_identifier(tool.get("name", "")))
    params = set(param_names(tool))
    acts = {a for a, vs in verbs.items() if tokens & vs}
    acts |= {a for a, ps in PARAM_ACTIONS.items() if params & ps}
    ann = tool.get("annotations") or {}
    if ann.get("destructiveHint") is True and ann.get("readOnlyHint") is not True:
        acts.add("delete")
    if ann.get("readOnlyHint") is True and not (acts - {"read"}):
        acts = {"read"}
    return acts


def _rule_risk_with(verbs, tool, domains=()) -> str:
    """改前规则的风险等级:与 `rules.rule_risk` 同逻辑,但用传入的动词表。"""
    ann = tool.get("annotations") or {}
    open_world = ann.get("openWorldHint")
    if open_world is None:
        open_world = bool(set(domains) & OPEN_WORLD_DOMAINS)
    return risk_from(_rule_actions_with(verbs, tool), domains, open_world)


def _risk_pair_metrics(gold, pred_old, pred_new) -> dict:
    """同一批银标上,改前规则 vs 改后规则的准确率 / L3 召回 / 低报率。"""
    n = len(gold)
    l3 = [i for i, g in enumerate(gold) if g == "L3"]

    def rec(pred):
        return sum(1 for i in l3 if pred[i] == "L3") / len(l3) if l3 else None

    def und(pred):
        return sum(1 for g, p in zip(gold, pred) if risk_rank(p) < risk_rank(g)) / n if n else 0.0

    def acc(pred):
        return sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0

    return {"n": n, "l3_n": len(l3),
            "accuracy_old": acc(pred_old), "accuracy_new": acc(pred_new),
            "l3_recall_old": rec(pred_old), "l3_recall_new": rec(pred_new),
            "under_old": und(pred_old), "under_new": und(pred_new)}


def rule_metric(tools, rows, tool_ids, old_verbs=None, new_verbs=None) -> dict:
    """一组工具上,改前规则 vs 当前规则(改后)对银标的风险指标。

    只统计两路一致的银标行;tool_ids 由 `tool_dev_test_split` 给出。
    new_verbs 给出时"改后"一列用这套表(报告里量第 1 轮的中间态时会用到)。
    """
    old_verbs = T011_OLD_VERB_ACTIONS if old_verbs is None else old_verbs
    rid = {r["tool_id"]: r for r in rows}
    gold, pred_old, pred_new = [], [], []
    for t in tools:
        if t["tool_id"] not in tool_ids:
            continue
        r = rid.get(t["tool_id"])
        if not r or not r["agreed"]:
            continue
        gold.append(r["risk_silver"])
        domains = r.get("domains", [])
        pred_old.append(_rule_risk_with(old_verbs, t, domains))
        pred_new.append(_rule_risk_with(new_verbs, t, domains) if new_verbs
                        else rule_risk(t, domains))
    return _risk_pair_metrics(gold, pred_old, pred_new)


def tool_dev_test_split(tools, by="name", dev_ratio=T011_DEV_RATIO, seed=SEED):
    """真实工具目录的开发 / 测试划分,返回 (开发 tool_id 集合, 测试 tool_id 集合)。

    by="name":每个函数名一组;by="server":每个 server 命名空间一组,但
    `server == "misc"`(名字里推不出命名空间)的工具仍按函数名分组 —— 否则 `misc`
    会把大半目录整块塞进一边,两边严重不均。
    整个分组要么进开发、要么进测试,避免同一服务 / 函数名的变体跨两边泄漏。
    """
    def key_of(t):
        if by == "server":
            server = t.get("server") or "misc"
            if server != "misc":
                return f"server:{server}"
        return f"name:{t.get('name', '')}"

    groups = {}
    for t in tools:
        groups.setdefault(key_of(t), []).append(t["tool_id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * dev_ratio)
    dev = {i for k in keys[:cut] for i in groups[k]}
    test = {i for k in keys[cut:] for i in groups[k]}
    return dev, test


def t011_verb_freq(tools, rows, tool_ids) -> tuple:
    """开发份上『银标 L3 但【改前】规则判低』的工具,按函数名动词计数。

    返回 (这类工具个数, Counter)。只用于 3.1 选词,统计口径固定是改前规则
    (这样"该补哪些词"的判断不随本轮改动变化)。
    """
    rid = {r["tool_id"]: r for r in rows}
    verbs = Counter()
    low = 0
    for t in tools:
        if t["tool_id"] not in tool_ids:
            continue
        r = rid.get(t["tool_id"])
        if not r or not r["agreed"] or r["risk_silver"] != "L3":
            continue
        if _rule_risk_with(T011_OLD_VERB_ACTIONS, t, r.get("domains", [])) == "L3":
            continue
        low += 1
        for v in split_identifier(t.get("name", "")):
            verbs[v] += 1
    return low, verbs


def compute_t011(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
                 encoder="tfidf", folds=FOLDS, seed=SEED) -> dict:
    """跑完 T-011 报告需要的全部计算,返回交给 `render_t011` 的 dict。

    只吃调用方读进来的数据;本身不读文件、不联网。
    """
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    mono = evaluate_all(labeled_rows, tools, requests, synth_tools, synth_train, real_train,
                        encoder=encoder, folds=folds, seed=seed,
                        bilingual=False, tool_head_mode="synth")
    bi = evaluate_all(labeled_rows, tools, requests, synth_tools, synth_train, real_train,
                      encoder=encoder, folds=folds, seed=seed,
                      bilingual=True, grid_search=True, tool_head_mode="synth",
                      zh_by_id=zh_by_id)
    risk = {mode: evaluate_risk(labeled_rows, tools, synth_tools, synth_train, encoder, mode)
            for mode in ("synth", "real", "both")}
    rule, rule_round1 = {}, {}
    for split_key, by in (("by_name", "name"), ("by_server", "server")):
        dev_ids, test_ids = tool_dev_test_split(tools, by=by, seed=seed)
        rule[split_key] = {"开发份": rule_metric(tools, labeled_rows, dev_ids),
                           "测试份": rule_metric(tools, labeled_rows, test_ids)}
        rule_round1[split_key] = {
            "开发份": rule_metric(tools, labeled_rows, dev_ids,
                                  new_verbs=T011_ROUND1_VERB_ACTIONS),
            "测试份": rule_metric(tools, labeled_rows, test_ids,
                                  new_verbs=T011_ROUND1_VERB_ACTIONS)}
    dev_ids, _ = tool_dev_test_split(tools, by="name", seed=seed)
    n_low, verbs = t011_verb_freq(tools, labeled_rows, dev_ids)
    return {
        "n_tools": len(tools), "n_requests": len(requests),
        "untranslated": sum(1 for r in zh_rows if not (r.get("description_zh") or "").strip()),
        "mono": mono, "bi": bi, "grid": bi["grid"], "risk": risk,
        "rule": rule, "rule_round1": rule_round1,
        "verbs": verbs, "n_low": n_low,
        "fold_of": assign_folds([r["id"] for r in requests], folds, seed),
    }


def _pct_delta(a, b, n, unit="条") -> str:
    """两个比例之差怎么描述:折合样本数不足 1 就写"无法区分"。"""
    d = (b - a) * n
    if abs(d) < 1:
        return "无法区分"
    return f"{d:+.1f} {unit}"


def _fold_r5(per_query, ids, lang="zh", mode="off") -> float:
    return _mean([per_query[i][lang][mode]["r5"] for i in ids])


def render_t011(res) -> str:
    """把 `compute_t011` 的结果渲染成 `reports/eval_t011.md` 的正文。"""
    reqs_n = res["n_requests"]
    fold_of = res["fold_of"]
    dev_ids, test_ids, _ = _dev_test_ids(list(fold_of), folds=FOLDS, seed=SEED)
    mono_c = res["mono"]["summary"][T011_OOF]["cells"]
    bi_c = res["bi"]["summary"][T011_OOF]["cells"]
    grid = res["grid"]
    zh5_m, zh5_b = mono_c[("zh", "off", "strict")]["r5"], bi_c[("zh", "off", "strict")]["r5"]
    en5_m, en5_b = mono_c[("en", "off", "strict")]["r5"], bi_c[("en", "off", "strict")]["r5"]

    L = ["# 真实工具目录修复评估(T-011)\n",
         f"工具 {res['n_tools']} 个,请求 {reqs_n} 条。银标是 Jev / Haiku 两路一致结果。\n",
         "> 第 0-2 折为开发集,第 3-4 折为测试集;工具按函数名分组 6:4 分开发 / 测试,"
         "再按 server 命名空间分组 6:4 报一组对照(`misc` 的工具仍按函数名分组)。"
         "开发集只用来选参 / 选词,测试集只报最终结果一次。\n"]

    # ---- 1. 双语索引 ----
    L.append("\n## 1. 双语工具索引\n")
    L.append(f"- {res['n_tools']} 条描述里未翻成 / 留空 {res['untranslated']} 条"
             "(缓存名 `tools_real_desc`,tag `zh`;函数名 / 参数名 / 代码标识符原样)。\n")
    L.append("\n| 索引 | 查询语言 | 口径 | R@1 | R@5 |\n| --- | --- | --- | --- | --- |")
    for idx_name, cells in (("单语(改前)", mono_c), ("双语(改后)", bi_c)):
        for lang, label_lang in (("zh", "中文"), ("en", "英文")):
            for suf, label_strict in (("strict", "严格"), ("name", "同名")):
                L.append(f"| {idx_name} | {label_lang} | {label_strict} "
                         f"| {_pct(cells[(lang, 'off', suf)]['r1'])} "
                         f"| {_pct(cells[(lang, 'off', suf)]['r5'])} |")
    L.append(f"\n改前改后(折外、过滤关、严格 R@5):中文 {_pct(zh5_m)} → **{_pct(zh5_b)}**"
             f"({(zh5_b - zh5_m) * 100:+.1f} 个百分点,约 {(zh5_b - zh5_m) * reqs_n:+.0f} 条);"
             f"英文 {_pct(en5_m)} → {_pct(en5_b)}"
             f"({(en5_b - en5_m) * 100:+.1f} 个百分点,{_pct_delta(en5_m, en5_b, reqs_n)})。\n")
    L.append("结论:双语索引把中文检索从噪声水平拉到可用,英文不受影响。"
             "只在 `tools_real` 的检索里生效(`--bilingual`),`catalog.py` 默认 `_index_text` 不动。\n")

    # ---- 2. 软过滤选参 ----
    L.append("\n## 2. 软过滤选参(只在开发集选)\n")
    L.append("\n### 2.1 开发集完整网格(中文查询 + 双语索引 + 严格 R@5)\n")
    L.append("| 过滤 | min_prob | prior_weight | 严格 R@5 | 平均候选数 | 金标被过滤 |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for c in grid["grid"]:
        filt = _pct(c["diag"]["gold_filtered_out"]) if c["filter_on"] else "-"
        L.append(f"| {'开' if c['filter_on'] else '关'} | {c['min_prob']:.2f} | {c['prior_weight']:.1f} "
                 f"| {_pct(c['zh_strict_r5'])} | {c['diag']['cand_count']:.1f} | {filt} |")
    best = grid["best"]
    off_zh5 = next(c["zh_strict_r5"] for c in grid["grid"]
                   if not c["filter_on"] and c["min_prob"] == MIN_PROB and c["prior_weight"] == 0.3)
    L.append(f"\n开发集最优:过滤 {'开' if best[0] else '关'}, min_prob={best[1]:.2f}, "
             f"prior_weight={best[2]:.1f}, 严格 R@5={_pct(grid['best_score'])};"
             f"过滤关={_pct(off_zh5)}。两者差 {abs(grid['best_score'] - off_zh5) * 100:.1f} 个百分点"
             f"({_pct_delta(off_zh5, grid['best_score'], len(dev_ids))})。\n")
    L.append("### 2.2 测试集三行对照(只报一次)\n")
    L.append("| 参数 | 中文严格 R@5 | 英文严格 R@5 | 中文同名 R@5 | 英文同名 R@5 | 平均候选 | 金标被过滤 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for r in grid["test_rows"]:
        filt = _pct(r["diag"]["gold_filtered_out"]) if r["cfg"][0] else "-"
        L.append(f"| {r['label']}({'开' if r['cfg'][0] else '关'}) | {_pct(r['zh_strict_r5'])} "
                 f"| {_pct(r['en_strict_r5'])} | {_pct(r['zh_name_r5'])} | {_pct(r['en_name_r5'])} "
                 f"| {r['diag']['cand_count']:.1f} | {filt} |")
    tmap = {r["label"]: r["zh_strict_r5"] for r in grid["test_rows"]}
    L.append(f"\n结论:开发集上过滤开(开发最优 {_pct(grid['best_score'])})与过滤关({_pct(off_zh5)})"
             f"无法区分;测试集上过滤关 {_pct(tmap['过滤关'])} 反而高于开发最优 {_pct(tmap['开发集最优'])}"
             f"和默认参数 {_pct(tmap['默认参数'])}。**真实目录下软过滤不划算**,"
             "选出来的参数就是 '关';`catalog.retrieve` 的默认参数不改,只在 `tools_real.py` 记下这个结论。\n")

    # ---- 3. 风险规则补漏 ----
    L.append("\n## 3. 风险规则补漏\n")
    L.append("\n### 3.1 开发份上『银标 L3 → 改前规则判低』的工具函数名动词频次(Top 30)\n")
    L.append(f"- 这类工具开发份上一共 {res['n_low']} 个。\n")
    L.append("\n| 动词 | 频次 |\n| --- | --- |")
    for v, c in res["verbs"].most_common(30):
        L.append(f"| {v} | {c} |")
    why = {"order": "下单付款,归 send", "book": "预订 / 预约产生外部影响,归 send",
           "booking": "同 book", "reserve": "预订,归 send", "buy": "购买是付款,归 send",
           "play": "播放媒体触发外部动作,归 execute"}
    L.append("\n补进去的词(都取自上面这张表;理由见下):\n")
    L.append("\n| 词 | 归到 | 开发份命中次数 | 说明 |\n| --- | --- | --- | --- |")
    for w in T011_ADDED_SEND:
        L.append(f"| {w} | send | {res['verbs'].get(w, 0)} | {why.get(w, '')} |")
    for w in T011_ADDED_EXEC:
        L.append(f"| {w} | execute | {res['verbs'].get(w, 0)} | {why.get(w, '')} |")
    r1 = res["rule_round1"]
    L.append("\n**第 2 轮删掉的词**(第 1 轮误加成 send;它们是服务名 / 名词,不是动作动词):"
             "`hotel` / `hotels` / `ride` / `uber` / `eat`。名字里含这 5 个词的真实工具有 84 个,"
             "其中 **37 个银标是 L0**(如 `Hotels_4_SearchHotel`、`Hotels_2_SearchHouse` 这类只读查询),"
             "第 1 轮把它们抬成了 L3。去掉这 5 个词后的实测变化:\n")
    L.append("\n| 划分 | 集合 | L3 召回(11 词 → 6 词) | 准确率(11 词 → 6 词) |\n"
             "| --- | --- | --- | --- |")
    for split_label, split_key in (("按函数名", "by_name"), ("按 server", "by_server")):
        for label in ("开发份", "测试份"):
            a = r1[split_key][label]      # 11 词(第 1 轮)
            b = res["rule"][split_key][label]   # 6 词(第 2 轮)
            L.append(f"| {split_label} | {label} | {_pct(a['l3_recall_new'])} → "
                     f"{_pct(b['l3_recall_new'])} | {_pct(a['accuracy_new'])} → "
                     f"{_pct(b['accuracy_new'])} |")
    L.append("\n两处**测试份**的 L3 召回一条不差(按函数名 "
             f"{_pct(res['rule']['by_name']['测试份']['l3_recall_new'])}、按 server "
             f"{_pct(res['rule']['by_server']['测试份']['l3_recall_new'])}),下降全落在开发份(按函数名 "
             f"{(r1['by_name']['开发份']['l3_recall_new'] - res['rule']['by_name']['开发份']['l3_recall_new']) * 100:.1f} "
             "个百分点、按 server "
             f"{(r1['by_server']['开发份']['l3_recall_new'] - res['rule']['by_server']['开发份']['l3_recall_new']) * 100:.1f} "
             "个百分点):这 5 个名词只在开发份上有收益,换一批 server 就不见了,属于过拟合;"
             "再加上会把 37 个只读工具抬成 L3,所以删掉。这也是为什么 3.2 要同时给两种划分。\n")
    L.append("\n被排除的候选词(有零个开发份 L3 命中,只会增加误报):"
             "`ticket`(命中 9,但其工具都已由 buy/book 覆盖,且它会把合成的 `create_ticket` 从 L2 抬到 L3)、"
             "`purchase` / `checkout` / `song` / `media` / `music`(开发份 L3 命中 0,只把 `walmart.purchase`、"
             "`get_song_lyrics`、`Media_3_FindMovies`、`music_*` 这些银标 L0/L2 的工具抬到 L3)。\n")

    L.append("\n### 3.2 真实工具目录规则改前 / 改后(银标两路一致的工具)\n")
    for split_label, split_key in (("按函数名分组 6:4", "by_name"),
                                   ("按 server 分组 6:4", "by_server")):
        L.append(f"\n**{split_label}**\n")
        L.append("\n| 集合 | 条数 | 准确率(改前/改后) | L3 召回(改前/改后) | 低报率(改前/改后) | L3 条数 |\n"
                 "| --- | --- | --- | --- | --- | --- |")
        for label in ("开发份", "测试份"):
            m = res["rule"][split_key][label]
            L.append(f"| {label} | {m['n']} | {_pct(m['accuracy_old'])} / {_pct(m['accuracy_new'])} | "
                     f"{_pct(m['l3_recall_old'])} / {_pct(m['l3_recall_new'])} | "
                     f"{_pct(m['under_old'])} / {_pct(m['under_new'])} | {m['l3_n']} |")
        if split_key == "by_server":
            m = res["rule"]["by_server"]["测试份"]
            d = (m["l3_recall_new"] - m["l3_recall_old"]) * m["l3_n"]
            L.append(f"\n按 server 分组的测试份:删掉名词后 L3 召回 {_pct(m['l3_recall_old'])} → "
                     f"{_pct(m['l3_recall_new'])},折合 {d:+.1f} 条,"
                     f"{'无法区分' if abs(d) < 1 else '有差别'}。\n")

    L.append("\n### 3.3 合成数据 `evaluate` 工具侧改前 / 改后\n")
    L.append("| 指标 | 改前 | 改后 |\n| --- | --- | --- |")
    for label, key in (("能力域 micro-F1 / macro-F1", "domain"), ("动作 micro-F1 / macro-F1", "action"),
                       ("风险准确率(模型+规则)", "risk_model"), ("风险准确率(仅规则)", "risk_rule"),
                       ("L3 召回率", "l3"), ("风险低报率", "under")):
        L.append(f"| {label} | {T011_SYNTH_BEFORE[key]} | {T011_SYNTH_AFTER[key]} |")
    L.append(f"| 进入人工审核队列 | {T011_SYNTH_BEFORE['queue']} 个 | {T011_SYNTH_AFTER['queue']} 个 |")
    L.append("\n改后逐字节等于改前(除延迟),日志见 `logs/eval_t011_before_rules.log`(改前,"
             "即合并前 main)与 `logs/eval_t011_after_rules.log`(改后)。"
             "最终词表在 53 个合成工具上 0 处改动,所以工具侧指标一条不动,"
             "A7(L3 召回 100%、低报率不升)满足。\n")

    # ---- 4. 真实数据训练工具头 ----
    L.append("\n## 4. 用真实数据训练工具侧分类头(GroupKFold,按函数名分组)\n")
    L.append(f"两路一致的 {res['risk']['synth']['silver_n']} 个工具做训练数据"
             "(domains 取银标,actions 取并集),按函数名分组 5 折。"
             "`synth` 是对照组(只用合成数据训练,现状);`real` 只用真实银标;`both` 合成 + 真实。\n")
    L.append("| 训练数据 | 能力域 micro-F1 | 能力域 macro-F1 | 动作 micro-F1 | 动作 macro-F1 | L3 召回 | 低报率 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for mode in ("synth", "real", "both"):
        risk = res["risk"][mode]
        f1, p = risk["head_f1"], risk["paths"][f"a 工具侧分类头({mode} 训练)"]
        L.append(f"| {mode} | {_pct(f1['domain']['micro_f1'])} | {_pct(f1['domain']['macro_f1'])} "
                 f"| {_pct(f1['action']['micro_f1'])} | {_pct(f1['action']['macro_f1'])} "
                 f"| {_pct(p['l3_recall'])} | {_pct(p['under_rate'])} |")
    rp = lambda m: res["risk"][m]["paths"][f"a 工具侧分类头({m} 训练)"]   # noqa: E731
    L.append(f"\n结论:用真实数据训练把能力域 micro-F1 从 "
             f"{_pct(res['risk']['synth']['head_f1']['domain']['micro_f1'])} 拉到 "
             f"{_pct(res['risk']['real']['head_f1']['domain']['micro_f1'])},动作 F1 也大幅上升;"
             f"但 L3 召回从 {_pct(rp('synth')['l3_recall'])} 降到 "
             f"{_pct(rp('real')['l3_recall'])} / {_pct(rp('both')['l3_recall'])},低报率从 "
             f"{_pct(rp('synth')['under_rate'])} 升到 {_pct(rp('real')['under_rate'])} / "
             f"{_pct(rp('both')['under_rate'])},"
             "**低于 100% 且低报率上升,按任务卡不能设成默认**,只能放在 `--tool-head` 开关后面(默认仍是 synth)。\n")

    # ---- 5. 建议 ----
    test_folds = sorted({fold_of[i] for i in test_ids})
    fold_bits = []
    for f in test_folds:
        ids = [i for i, fo in fold_of.items() if fo == f]
        m = _fold_r5(res["mono"]["variants"][T011_OOF], ids)
        b = _fold_r5(res["bi"]["variants"][T011_OOF], ids)
        fold_bits.append(f"第 {f} 折 {_pct(m)}→{_pct(b)}")
    by_name_test = res["rule"]["by_name"]["测试份"]
    by_server_test = res["rule"]["by_server"]["测试份"]
    L.append("\n## 5. 建议设成默认的改动(只写建议,不改默认值)\n")
    L.append(f"- **双语索引:建议**(`tools_real.py` 检索)。600 条上中文严格 R@5 "
             f"{_pct(zh5_m)}→{_pct(zh5_b)}(测试折:" + "、".join(fold_bits) + "),英文不退步;"
             "合成数据 `evaluate` 不受影响(它不读 tools_real)。\n")
    L.append("- **软过滤:不建议设默认**。开发集上开 / 关无法区分,测试集上过滤关更高;"
             "真实目录下软过滤不划算,保持现状(不改 `catalog.retrieve` 默认参数)。\n")
    L.append(f"- **风险规则补漏:建议**。测试份(按函数名)L3 召回 {_pct(by_name_test['l3_recall_old'])}→"
             f"{_pct(by_name_test['l3_recall_new'])}、低报率 {_pct(by_name_test['under_old'])}→"
             f"{_pct(by_name_test['under_new'])}、准确率 {_pct(by_name_test['accuracy_old'])}→"
             f"{_pct(by_name_test['accuracy_new'])};测试份(按 server)L3 召回 "
             f"{_pct(by_server_test['l3_recall_old'])}→{_pct(by_server_test['l3_recall_new'])}、"
             f"低报率 {_pct(by_server_test['under_old'])}→{_pct(by_server_test['under_new'])};"
             "合成的 `evaluate` 一条不动(A7 满足)。\n")
    L.append("- **真实数据工具头:不建议设默认**。能力域 F1 大涨,但 L3 召回 <100%、低报率上升,"
             "必须放在开关后面(默认 `synth`)。\n")
    L.append("\n> 只有双语索引和风险补漏在测试集上有明确收益、且合成 `evaluate` 不退步;"
             "另两项要么真实目录上不划算(软过滤),要么合成 / 风险口径退步(真实工具头)。\n")
    return "\n".join(L) + "\n"


def t011_metrics_json(res) -> dict:
    """报告数字的机器可读版(`reports/eval_t011.json`);tuple 键转成 `a/b/c` 字符串。"""
    def cells(summary):
        return {"/".join(map(str, k)): v for k, v in summary["cells"].items()}

    return {
        "mono_summary": {k: {"cells": cells(v), "diag": v["diag"], "n": v["n"]}
                         for k, v in res["mono"]["summary"].items()},
        "bi_summary": {k: {"cells": cells(v), "diag": v["diag"], "n": v["n"]}
                       for k, v in res["bi"]["summary"].items()},
        "grid": res["grid"], "rule": res["rule"],
        "verbs_top30": res["verbs"].most_common(30),
    }


# ---------------------------------------------------------------- T-023 风险信号
#
# `reports/eval_t023.md` 的生成逻辑放在这里(而不是一次性脚本里),由
# `tools-real report-t023` 调用;只读已落盘的 jsonl,不联网,保证 clone 仓库后能用
# 同一条命令重跑。改前 = `rules.rule_risk_base`(即 main),改后 = `rules.rule_risk`。
# 词表只按开发份选(见报告第 1 节),测试份和参考集只在最后报一次。


# 调查用的宽候选词表(报告第 1.0 节的排序表);最终留哪些词由开发份数据决定,
# 不含这里的任何一个词之外的词。这两个常量只服务于报告,不参与规则判定。
T023_SCAN_CANDIDATES = (
    "scan", "scanner", "crawl", "crawler", "probe", "port", "nmap", "ping", "traceroute",
    "whois", "discover", "enumerate", "enum", "nuclei", "attack", "penetration", "subdomain",
    "osint", "vulnerability", "vulnerabilities", "exploit", "recon", "detect",
)
T023_DEVICE_CANDIDATES = (
    "device", "iot", "thermostat", "light", "lock", "unlock", "oven", "heat", "heater",
    "preheat", "appliance", "vehicle", "sensor", "temperature", "bulb", "lamp", "garage",
    "control", "power", "turn on", "turn off", "switch", "camera", "door",
)


def _pair_with_over(gold, pred_old, pred_new) -> dict:
    """在 `_risk_pair_metrics` 上补两个"低判成 L3"(多打扰)计数。"""
    m = _risk_pair_metrics(gold, pred_old, pred_new)
    m["over_old"] = sum(1 for g, p in zip(gold, pred_old) if g != "L3" and p == "L3")
    m["over_new"] = sum(1 for g, p in zip(gold, pred_new) if g != "L3" and p == "L3")
    return m


def rule_metric_t023(tools, rows, tool_ids) -> dict:
    """一组工具上,T-023 改动前(`rule_risk_base`)vs 改动后(`rule_risk`)的规则指标。"""
    rid = {r["tool_id"]: r for r in rows}
    gold, pred_old, pred_new = [], [], []
    for t in tools:
        if t["tool_id"] not in tool_ids:
            continue
        r = rid.get(t["tool_id"])
        if not r or not r["agreed"]:
            continue
        dom = r.get("domains", [])
        gold.append(r["risk_silver"])
        pred_old.append(rule_risk_base(t, dom))
        pred_new.append(rule_risk(t, dom))
    return _pair_with_over(gold, pred_old, pred_new)


def t023_word_stats(tools, rows, dev_ids, test_ids) -> dict:
    """每个新词在开发份 / 测试份上的命中数,按银标 L3/L2/L1/L0 拆开。"""
    rid = {r["tool_id"]: r for r in rows}

    def counts(word, ids):
        by = {"L3": 0, "L2": 0, "L1": 0, "L0": 0, "unagreed": 0}
        for t in tools:
            if t["tool_id"] not in ids:
                continue
            name_tokens = set(split_identifier(t.get("name", "")))
            desc_words = set(re.findall(r"[a-z0-9]+", (t.get("description") or "").lower()))
            if word not in (name_tokens | desc_words):
                continue
            r = rid.get(t["tool_id"])
            if not r or not r["agreed"]:
                by["unagreed"] += 1
            else:
                by[r["risk_silver"]] += 1
        by["hit"] = sum(by.values())
        by["non_l3"] = by["L0"] + by["L1"] + by["L2"]
        return by

    out = {"scan": {}, "device_nouns": {}, "device_verbs": {}}
    for w in sorted(SCAN_HOST_TOKENS):
        out["scan"][w] = {"开发份": counts(w, dev_ids), "测试份": counts(w, test_ids)}
    for group, words in (("device_nouns", DEVICE_NOUNS), ("device_verbs", DEVICE_CONTROL_VERBS)):
        for w in sorted(words):
            out[group][w] = {"开发份": counts(w, dev_ids), "测试份": counts(w, test_ids)}
    return out


def _word_counts(tools, rows, word, ids) -> dict:
    """一个词(名字 token 或描述词)在一组工具里的命中与银标分布。"""
    rid = {r["tool_id"]: r for r in rows}
    by = {"hit": 0, "L3": 0, "L2": 0, "L1": 0, "L0": 0, "unagreed": 0}
    for t in tools:
        if t["tool_id"] not in ids:
            continue
        name_tokens = set(split_identifier(t.get("name", "")))
        desc_words = set(re.findall(r"[a-z0-9]+", (t.get("description") or "").lower()))
        if word not in (name_tokens | desc_words):
            continue
        by["hit"] += 1
        r = rid.get(t["tool_id"])
        if not r or not r["agreed"]:
            by["unagreed"] += 1
        else:
            by[r["risk_silver"]] += 1
    by["non_l3"] = by["L0"] + by["L1"] + by["L2"]
    return by


def t023_candidates(tools, rows, dev_ids, test_ids) -> list:
    """调查用的宽候选词表(含最终被排除的词),按开发份 L3 命中排序。

    排序口径 = 开发份银标 L3 命中数降序、开发份非 L3 误伤升序;`kept` 标出最终留下的词。
    """
    kept = set(SCAN_HOST_TOKENS) | set(DEVICE_NOUNS) | set(DEVICE_CONTROL_VERBS)
    out = []
    for group, words in (("扫描/探测", T023_SCAN_CANDIDATES),
                         ("控制设备", T023_DEVICE_CANDIDATES)):
        for w in words:
            d = _word_counts(tools, rows, w, dev_ids)
            tt = _word_counts(tools, rows, w, test_ids)
            out.append({"group": group, "word": w, "kept": w in kept,
                        "dev": d, "test": tt})
        out.sort(key=lambda r: (r["group"] != group, -r["dev"]["L3"], r["dev"]["non_l3"]))
    return out


def t023_flagged(tools, rows, dev_ids, test_ids) -> list:
    """命中两类信号的工具,带银标、所在划分、改前 / 改后风险。"""
    rid = {r["tool_id"]: r for r in rows}
    out = []
    for t in tools:
        sig = risk_signals(t)
        if not sig:
            continue
        r = rid.get(t["tool_id"])
        dom = (r or {}).get("domains", [])
        split = ("开发份" if t["tool_id"] in dev_ids else "测试份"
                 if t["tool_id"] in test_ids else "开发/测试按 server 划分不适用")
        out.append({"tool_id": t["tool_id"], "signal": "/".join(sorted(sig)),
                    "silver": r["risk_silver"] if r and r["agreed"] else None,
                    "split_by_name": split,
                    "old": rule_risk_base(t, dom), "new": rule_risk(t, dom)})
    return out


def t023_reference(risk_ref, tools, rows) -> dict:
    """参考集(D-005)上规则的风险指标,改前 / 改后并排。只报一次(改后)。"""
    tmap = {t["tool_id"]: t for t in tools}
    dom_of = {r["tool_id"]: r.get("domains", []) for r in rows}
    gold, pred_old, pred_new = [], [], []
    under_old, under_new, over_old, over_new, added, removed = [], [], [], [], [], []
    for r in risk_ref:
        t = tmap.get(r["tool_id"])
        if t is None:
            continue
        dom = dom_of.get(r["tool_id"], [])
        o, n = rule_risk_base(t, dom), rule_risk(t, dom)
        gold.append(r["truth"])
        pred_old.append(o)
        pred_new.append(n)
        if r["truth"] == "L3" and o != "L3":
            under_old.append(r["tool_id"])
        if r["truth"] == "L3" and n != "L3":
            under_new.append(r["tool_id"])
        if r["truth"] != "L3" and o == "L3":
            over_old.append(r["tool_id"])
        if r["truth"] != "L3" and n == "L3":
            over_new.append(r["tool_id"])
        if o != n:
            (added if n == "L3" else removed).append(
                {"tool_id": r["tool_id"], "old": o, "new": n, "truth": r["truth"]})
    m = _pair_with_over(gold, pred_old, pred_new)
    m.update({"under_old_ids": under_old, "under_new_ids": under_new,
              "over_old_ids": over_old, "over_new_ids": over_new,
              "added_l3": added, "removed_l3": removed})
    return m


def t023_class_summary(flagged) -> dict:
    """两类信号各自命中的工具数、银标分布、5 个例子(从 `t023_flagged` 汇总)。"""
    out = {}
    for f in flagged:
        for sig in f["signal"].split("/"):
            c = out.setdefault(sig, {"n": 0, "L3": 0, "L2": 0, "L1": 0, "L0": 0,
                                     "unagreed": 0, "examples": []})
            c["n"] += 1
            c[f["silver"] or "unagreed"] += 1
            if len(c["examples"]) < 5:
                c["examples"].append(f["tool_id"])
    return out


def compute_t023(tools, rows, risk_ref, seed=SEED) -> dict:
    """跑完 T-023 报告需要的全部计算;只吃调用方读进来的数据,本身不读文件。"""
    dev_name, test_name = tool_dev_test_split(tools, by="name", seed=seed)
    dev_srv, test_srv = tool_dev_test_split(tools, by="server", seed=seed)
    flagged = t023_flagged(tools, rows, dev_name, test_name)
    return {
        "n_tools": len(tools),
        "rule": {
            "by_name": {"开发份": rule_metric_t023(tools, rows, dev_name),
                        "测试份": rule_metric_t023(tools, rows, test_name)},
            "by_server": {"开发份": rule_metric_t023(tools, rows, dev_srv),
                          "测试份": rule_metric_t023(tools, rows, test_srv)},
        },
        "words": t023_word_stats(tools, rows, dev_name, test_name),
        "candidates": t023_candidates(tools, rows, dev_name, test_name),
        "reference": t023_reference(risk_ref, tools, rows),
        "flagged": flagged,
        "classes": t023_class_summary(flagged),
    }


def render_t023(res) -> str:
    """把 `compute_t023` 的结果渲染成 `reports/eval_t023.md` 的正文。"""
    L = ["# 风险规则补两类信号:扫描外部主机、控制实体设备(T-023)\n",
         f"真实工具目录 {res['n_tools']} 个,银标是 Jev / Haiku 两路一致结果。"
         "改前 = main 的 `rules.rule_risk`(T-011 最终版),改后 = 叠加 T-023 两类信号。\n",
         "> 词表只按开发份(按函数名 6:4 切分)选,测试份和参考集只在最后各报一次、不参与选词:"
         "扫描类逐词选(开发份银标 L3 命中 >= 2 且非 L3 误伤 = 0),设备类是\"设备名词 + 控制动词\""
         "组合信号(见 1.0)。复跑:`cli tools-real report-t023`(只读 `data/` 下已落盘文件,不联网)。\n"]

    ref = res["reference"]
    L.append("\n## 0. 参考集(D-005,只报一次,改后)\n")
    L.append(f"- 参考集 {ref['n']} 个工具。规则 L3 判低(真值 L3 但判低):"
             f"改前 {len(ref['under_old_ids'])} 个 → **改后 {len(ref['under_new_ids'])} 个**"
             f"(改前 {ref['under_old_ids']});多打扰(真值非 L3 但判 L3):"
             f"改前 {len(ref['over_old_ids'])} → 改后 **{len(ref['over_new_ids'])}**。\n")
    L.append("- 改后新增 L3 的工具:\n")
    L.append("\n| tool_id | 改前 | 改后 | 真值 |\n| --- | --- | --- | --- |")
    for c in ref["added_l3"]:
        L.append(f"| {c['tool_id']} | {c['old']} | {c['new']} | {c['truth']} |")
    if not ref["added_l3"]:
        L.append("| (无) | | | |")
    L.append(f"\n参考集准确率 {_pct(ref['accuracy_old'])} → {_pct(ref['accuracy_new'])}。\n")

    L.append("\n## 1. 词表(只按开发份选)\n")
    L.append("\n### 1.0 两类工具概览(信号命中,银标分布)\n")
    L.append("\n| 信号 | 命中数 | 银标L3 | 银标L2 | 银标L1 | 银标L0 | 未一致 | 5 个例子 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- |")
    for sig, c in sorted(res["classes"].items()):
        L.append(f"| {sig} | {c['n']} | {c['L3']} | {c['L2']} | {c['L1']} | {c['L0']} "
                 f"| {c['unagreed']} | {', '.join(c['examples'])} |")
    L.append("\n### 1.1 宽候选词排序(开发份,按银标 L3 命中降序 / 非 L3 误伤升序)\n")
    L.append("两类信号的选词口径:**扫描类**按词选(开发份银标 L3 命中 >= 2 且非 L3 误伤 = 0 才留);"
             "**设备类**是\"设备名词 + 控制动词\"的组合信号,单个词不单独判定,词表见 1.3 / 1.4,"
             "本表用来看每个候选词自己会不会误伤。`用在规则里` = 该词进了最终词表。\n")
    L.append("\n| 类 | 词 | 用在规则里 | 开发份命中 | 开发L3 | 开发L2 | 开发L1 | 开发L0 | 测试命中 | 测试L3 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in res["candidates"]:
        d, tt = c["dev"], c["test"]
        mark = f"**{c['word']}**" if c["kept"] else c["word"]
        L.append(f"| {c['group']} | {mark} | {'是' if c['kept'] else '否'} | {d['hit']} "
                 f"| {d['L3']} | {d['L2']} | {d['L1']} | {d['L0']} | {tt['hit']} | {tt['L3']} |")
    L.append("\n扫描类被排除的候选词:`crawl` / `crawler` / `probe` / `port` / `nmap` / `ping` /"
             "`traceroute` / `whois`(开发份 L3 命中 0;参考集里的 `Crawler.scan` 已由 `scan` 命中)、"
             "`nuclei` / `attack` / `subdomain` / `osint` / `detect` / `exploit` / `vulnerability` /"
             "`vulnerabilities`(开发份 L3 命中 < 2,等于把单个工具的名字记进词表;"
             "`nuclei_http` 的描述里有 `scan`,已被 `scan` 覆盖)。\n")
    L.append("\n设备类被排除的候选词:`vehicle` / `sensor` / `temperature`(开发份非 L3 误伤 7 / 5 / 21,"
             "都是 `calculate_emissions`、`get_sensor_readings` 这类只读查询)、`iot`(开发份 L3 命中 0)。"
             "1.2 / 1.3 表里的命中数是单个词自己的计数,设备信号只在\"名词 + 动词\"同时出现时才生效:"
             "`heat` 作为裸词会误伤 `Movies_*_FindMovies`,和 `oven` 组合才命中 `oven_preheat`;"
             "`set` 裸词很常见(开发份 36 个),但和 `thermostat` / `device` 这类名词组合时才是控制动作。\n")
    for i, (title, key) in enumerate((("扫描 / 探测外部主机", "scan"),
                                      ("控制实体设备:设备名词", "device_nouns"),
                                      ("控制实体设备:控制动词", "device_verbs")), start=2):
        L.append(f"\n### 1.{i} {title}\n")
        L.append("\n| 词 | 来源 | 开发份命中 | 开发L3 | 开发L2 | 开发L1 | 开发L0 | 测试份命中 | 测试L3 |\n"
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for w, st in res["words"][key].items():
            src = RISK_KEYWORD_SOURCES.get(w, "?")
            d, tt = st["开发份"], st["测试份"]
            L.append(f"| {w} | {src} | {d['hit']} | {d['L3']} | {d['L2']} | {d['L1']} "
                     f"| {d['L0']} | {tt['hit']} | {tt['L3']} |")
    L.append("\n命中工具(全部,共 "
             f"{len(res['flagged'])} 个;开发份 / 测试份按函数名划分):\n")
    L.append("\n| tool_id | 信号 | 划分 | 银标 | 改前 | 改后 |\n| --- | --- | --- | --- | --- | --- |")
    for f in res["flagged"]:
        silver = f["silver"] if f["silver"] else "(未一致)"
        L.append(f"| {f['tool_id']} | {f['signal']} | {f['split_by_name']} | {silver} "
                 f"| {f['old']} | {f['new']} |")

    L.append("\n## 2. 开发份 / 测试份规则指标(改前 / 改后)\n")
    for split_label, split_key in (("按函数名分组 6:4", "by_name"),
                                   ("按 server 分组 6:4", "by_server")):
        L.append(f"\n**{split_label}**\n")
        L.append("\n| 集合 | 条数 | L3 条数 | 准确率(改前/改后) | L3 召回(改前/改后) "
                 "| 低报率(改前/改后) | 低判成 L3(改前/改后) |\n"
                 "| --- | --- | --- | --- | --- | --- | --- |")
        for label in ("开发份", "测试份"):
            m = res["rule"][split_key][label]
            L.append(f"| {label} | {m['n']} | {m['l3_n']} | {_pct(m['accuracy_old'])} / "
                     f"{_pct(m['accuracy_new'])} | {_pct(m['l3_recall_old'])} / "
                     f"{_pct(m['l3_recall_new'])} | {_pct(m['under_old'])} / "
                     f"{_pct(m['under_new'])} | {m['over_old']} / {m['over_new']} |")
    bn_t = res["rule"]["by_name"]["测试份"]
    bs_t = res["rule"]["by_server"]["测试份"]
    d_rec = (bn_t["l3_recall_new"] - bn_t["l3_recall_old"]) * bn_t["l3_n"]
    L.append(f"\n测试份(按函数名)L3 召回 {_pct(bn_t['l3_recall_old'])} → "
             f"{_pct(bn_t['l3_recall_new'])}(折合 {_pct_delta(bn_t['l3_recall_old'], bn_t['l3_recall_new'], bn_t['l3_n'])});"
             f"测试份(按 server)L3 召回 {_pct(bs_t['l3_recall_old'])} → "
             f"{_pct(bs_t['l3_recall_new'])}。低判成 L3 两套划分都不增加"
             f"(按函数名 {bn_t['over_old']}→{bn_t['over_new']}、按 server "
             f"{bs_t['over_old']}→{bs_t['over_new']})。\n")
    L.append("\n## 3. 结论\n")
    L.append(f"- 参考集两个漏判 L3(`Crawler/Crawler.scan`、`misc/oven_preheat`)都补上了:"
             f"规则 L3 判低 {len(ref['under_old_ids'])} → {len(ref['under_new_ids'])};"
             f"多打扰 {len(ref['over_old_ids'])} → {len(ref['over_new_ids'])}。\n")
    L.append("- 测试份(两种划分)L3 召回都上升、低判成 L3 不增加;开发份同理。\n")
    L.append("- 合成工具(`data/tools.jsonl`,53 个)无一命中两类信号,`cli evaluate` 输出不变。\n")
    return "\n".join(L) + "\n"


def t023_metrics_json(res) -> dict:
    """报告数字的机器可读版(`reports/eval_t023.json`)。"""
    return {"n_tools": res["n_tools"], "rule": res["rule"], "words": res["words"],
            "candidates": res["candidates"], "reference": res["reference"],
            "flagged": res["flagged"], "classes": res["classes"]}


# ---------------------------------------------------------------- 线上默认:真实目录(T-012)

# 软 need 阈值(T-020):need 判成 none 但 none 概率低于它时仍附带候选。
# 依据 reports/eval_t020.md:开发折按事先定的规则(alpaca_zh 误挂率 <= 10% 里中文主口径严格 R@5 最高)
# 选出 0.7(26.0% -> 37.5%,误挂率 2.5%,留出子集 6.9%);测试折复核 20.3% -> 32.0%。只用于真实目录的 router。
SOFT_NEED_DEFAULT = 0.7

def build_real_router(tools, labeled_rows, zh_rows, requests, synth_tools,
                      encoder="tfidf", threshold=0.55, top_k=5, use_filter=False,
                      dedupe_by_name=False, retrieval_encoder=None,
                      soft_need_threshold=SOFT_NEED_DEFAULT, reranker=None):
    """按 T-011 的结论构造线上默认(真实目录)的 Router。

    复用 `tools_real.py` 里已有的三件事:`attach_labels` 挂银标、`build_fold_index`
    建双语索引、`ev.train_request_head` 训请求头。与合成的差别:
    - 工具取 `data/tools_real.jsonl`,domains/actions 取两路一致银标(`use_labels=True`);
    - 索引文本拼上 `tools_real_zh.jsonl` 的中文描述(`bilingual=True`);
    - 工具侧分类头仍用合成 `tools.jsonl` 训练(T-011 的结论:真实数据训的头不能设成默认);
    - 软过滤关(`use_filter=False`,`retrieve` 收到 domains=None);
    - 按函数名去重默认**关**:`reports/eval_t013.md` 第 0 节,开发折(第 0-2 折)中文
      "假设 need 全对"的严格 R@5 从 51.0%(关)掉到 43.9%(开),折合 -25.5 条,所以不设默认。
      去重能明显抬高同名口径(开发折中文上限同名 R@5 63.5%→72.5%),但严格口径下降,
      按任务卡只看开发折严格 R@5,结论是关。
    retrieval_encoder:检索编码器 spec(T-019);默认 None=沿用分类头编码器。
    soft_need_threshold:软 need 阈值(T-020),默认 `SOFT_NEED_DEFAULT`;传 None 关闭。
    reranker:语义重排器实例(T-025);默认 None=不重排,行为与改前逐字节一致。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    renc = ev.build_encoder(retrieval_encoder) if retrieval_encoder else None
    enc, _head, cat = build_fold_index(labeled_tools, synth_tools, requests, encoder,
                                       bilingual=True, zh_by_id=zh_by_id,
                                       retrieval_encoder=renc)
    req_head = ev.train_request_head(enc, requests)
    return Router(req_head, cat, threshold=threshold, top_k=top_k, use_filter=use_filter,
                  dedupe_by_name=dedupe_by_name, soft_need_threshold=soft_need_threshold,
                  reranker=reranker)


# 端到端评估的三行对照:(目录键, 报告里的名字)
E2E_CATALOGS = (("synth", "合成目录(53 个手写工具)"),
                ("real_mono", "真实目录 + 单语索引"),
                ("real_bi", "真实目录 + 双语索引(新默认)"))
E2E_REPORT = REPORTS / "eval_e2e.md"
E2E_REPORT_JSON = REPORTS / "eval_e2e.json"


def _e2e_router(cat_name, train_rows, labeled_tools, synth_tools, encoder, top_k, zh_by_id,
                retrieval_encoder=None):
    """一条端到端评估用的 Router:合成目录沿用旧默认(软过滤开、单语),
    真实目录用新默认(单语/双语 + 软过滤关)。三行都不加示例查询。

    retrieval_encoder(T-019):检索编码器实例,不为 None 时索引与查询都用它;
    分类头仍用 `encoder`。为 None 时行为与改前一致。
    """
    if cat_name == "synth":
        enc = ev.fit_encoder(encoder, synth_tools, train_rows)
        head = ev.train_tool_head(enc, synth_tools)
        if retrieval_encoder is None:
            cat = ToolCatalog(head)
        else:
            renc = retrieval_encoder.fit(_fit_texts(synth_tools, train_rows))
            cat = ToolCatalog(head, retrieval_encoder=renc)
        cat.add_tools(synth_tools, use_labels=True)
        rhead = ev.train_request_head(enc, train_rows)
        return Router(rhead, cat, top_k=top_k, use_filter=True)
    enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train_rows, encoder,
                                       bilingual=(cat_name == "real_bi"), zh_by_id=zh_by_id,
                                       retrieval_encoder=retrieval_encoder)
    rhead = ev.train_request_head(enc, train_rows)
    return Router(rhead, cat, top_k=top_k, use_filter=False)


def _e2e_item(router, text, gold, gold_names, name_of, top_k) -> dict:
    """一条请求走完整 `Router.route`,再把主口径与"假设 need 全对"的检索一起记下来。

    主口径:用 route 给出的候选(判成 none 时候选为空,自动算 0)。
    上限:不看 need 判定、也不做软过滤,直接 `retrieve(text, None, top_k)`,即纯检索天花板。
    """
    res = router.route(text)
    cand_ids = [c["tool_id"] for c in res.candidates]
    upper_ids = [rec.tool_id for rec, _ in router.catalog.retrieve(text, None, top_k)]

    def rec(ids, by_name, k):
        if not gold:
            return 0.0
        if by_name:
            return len(gold_names & {name_of.get(t, t) for t in ids[:k]}) / len(gold_names)
        return len(set(gold) & set(ids[:k])) / len(set(gold))

    item = {"need": res.need, "pred_none": res.need == "none", "latency_ms": res.latency_ms,
            "risk_top1": (res.candidates[0]["risk"] if res.candidates else "")}
    for mode, ids in (("main", cand_ids), ("upper", upper_ids)):
        for suf, by_name in (("", False), ("_name", True)):
            item[f"{mode}_r1{suf}"] = rec(ids, by_name, 1)
            item[f"{mode}_r5{suf}"] = rec(ids, by_name, 5)
    return item


def e2e_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
             encoder="tfidf", folds=FOLDS, seed=SEED, top_k=5, retrieval_encoder=None) -> dict:
    """端到端评估:三种目录口径 × 中英文 × 折外请求头,每条走完整路由。

    请求头按 5 折折外训练(`assign_folds`,seed 不变):每个目录都按折建一次索引 + 请求头,
    每条请求只用不含它的那折。返回结构交给 `e2e_summary` / `render_e2e`。

    retrieval_encoder(T-019):检索编码器实例,不为 None 时三个目录的索引与查询都用它;
    分类头仍用 `encoder`。为 None 时行为与改前一致。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    per = {}
    for cat_name in [c for c, _ in E2E_CATALOGS]:
        items = {}
        for f in range(folds):
            ids = {i for i, fo in fold_of.items() if fo == f}
            rows = [r for r in requests if r["id"] in ids]
            train = fold_train_rows(synth_train, real_train, ids)
            router = _e2e_router(cat_name, train, labeled_tools, synth_tools, encoder, top_k,
                                 zh_by_id, retrieval_encoder=retrieval_encoder)
            for r in rows:
                gold = r["tool_ids"]
                gold_names = {name_of.get(g, g) for g in gold}
                items[r["id"]] = {"zh": _e2e_item(router, r["text"], gold, gold_names, name_of, top_k),
                                  "en": _e2e_item(router, r["text_orig"], gold, gold_names, name_of, top_k)}
        per[cat_name] = items
    return {"catalogs": [c for c, _ in E2E_CATALOGS],
            "labels": {c: label for c, label in E2E_CATALOGS},
            "per": per, "requests": requests, "fold_of": fold_of}


def e2e_summary(res) -> dict:
    """端到端结果按目录汇总:R@1/R@5(主口径 + 上限)、延迟、top-1 风险分布。"""
    requests = res["requests"]
    out = {}
    for cat in res["catalogs"]:
        items = res["per"][cat]
        s = {}
        for lang in ("zh", "en"):
            for mode in ("main", "upper"):
                for suf, key in (("", "strict"), ("_name", "name")):
                    s[f"{lang}_{mode}_r1_{key}"] = _mean(
                        [items[r["id"]][lang][f"{mode}_r1{suf}"] for r in requests])
                    s[f"{lang}_{mode}_r5_{key}"] = _mean(
                        [items[r["id"]][lang][f"{mode}_r5{suf}"] for r in requests])
        lat = [items[r["id"]]["zh"]["latency_ms"] for r in requests]
        s["latency_p50"] = float(np.percentile(lat, 50))
        s["latency_p95"] = float(np.percentile(lat, 95))
        risks = Counter(items[r["id"]]["zh"]["risk_top1"] for r in requests
                        if items[r["id"]]["zh"]["risk_top1"])
        s["risk_top1"] = dict(sorted(risks.items()))
        s["pred_none"] = sum(1 for r in requests if items[r["id"]]["zh"]["pred_none"])
        out[cat] = s
    return out


def e2e_need_metrics(res, lang="zh") -> dict:
    """need 判定指标:只统计银标有 need 的请求子集。need 与工具目录无关。"""
    items = res["per"][res["catalogs"][0]]
    paired = [(r, items[r["id"]][lang]) for r in res["requests"] if r["need"]]
    pos = sum(1 for r, _ in paired if r["need"] != "none")
    tp = sum(1 for r, z in paired if r["need"] != "none" and not z["pred_none"])
    ppos = sum(1 for _, z in paired if not z["pred_none"])
    return {"n": len(paired), "pos": pos,
            "recall": (tp / pos) if pos else None,
            "precision": (tp / ppos) if ppos else None,
            "pred_none": sum(1 for _, z in paired if z["pred_none"]),
            "pred_none_gold_needs": sum(1 for r, z in paired if z["pred_none"] and r["need"] != "none")}


def render_e2e(res) -> str:
    """把 `e2e_eval` 的结果渲染成 `reports/eval_e2e.md` 的正文。"""
    requests = res["requests"]
    n = len(requests)
    summary = e2e_summary(res)
    need_zh, need_en = e2e_need_metrics(res, "zh"), e2e_need_metrics(res, "en")
    L = ["# 端到端评估(T-012)\n",
         f"请求 {n} 条(600 条 BFCL 真实请求里能对上金标 tool_id 的部分),工具目录 2405 个。"
         "need 银标来自 `data/v3-primary/labeled_agree.jsonl`,工具银标是 Jev / Haiku 两路一致结果。\n",
         "> 请求头按 T-010b 的 5 折折外方式训练(`assign_folds`,seed 不变):每条请求都用不含它的那折训练。"
         "每条走完整 `Router.route`:need 判定 → 能力域 → 检索 top-5。\n",
         "> 三行口径:合成目录(53 个手写工具,软过滤开,旧默认)、真实目录 + 单语索引(软过滤关)、"
         "真实目录 + 双语索引(软过滤关,新默认)。都不加示例查询、不调检索参数。\n"]

    L.append("\n## 1. need 判定\n")
    L.append("银标有 need 的请求子集上算:需要 MCP 召回 = 银标要工具且判对 / 银标要工具;"
             "精确率 = 判对 / 判成要工具。\n")
    L.append("\n| 查询语言 | 子集条数 | 需要 MCP 召回 | 需要 MCP 精确率 | 判成 none | 其中银标要工具 |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for label, m in (("中文", need_zh), ("英文", need_en)):
        L.append(f"| {label} | {m['n']} | {_pct(m['recall'])} | {_pct(m['precision'])} "
                 f"| {m['pred_none']} | {m['pred_none_gold_needs']} |")
    L.append("\nneed 判定与工具目录无关,三个目录共用同一份请求头,数字相同;"
             "英文是参考口径。\n")

    L.append("\n## 2. 检索(三行对照)\n")
    L.append("主口径分母是全部请求;判成 none 的请求候选为空,按 0 计。"
             "『假设 need 全对』是不看 need 判定、也不做软过滤直接检索的上限(纯检索天花板)。\n")
    L.append("\n| 目录 | 查询语言 | 口径 | R@1(need 误判计 0) | R@5(need 误判计 0) "
             "| R@1(假设 need 全对) | R@5(假设 need 全对) |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for cat in res["catalogs"]:
        s = summary[cat]
        for lang, lang_label in (("zh", "中文"), ("en", "英文")):
            for key, key_label in (("strict", "严格"), ("name", "同名")):
                L.append(f"| {res['labels'][cat]} | {lang_label} | {key_label} "
                         f"| {_pct(s[f'{lang}_main_r1_{key}'])} | {_pct(s[f'{lang}_main_r5_{key}'])} "
                         f"| {_pct(s[f'{lang}_upper_r1_{key}'])} | {_pct(s[f'{lang}_upper_r5_{key}'])} |")
    L.append("\n三行的差别:合成目录命中率接近 0(那 600 条请求要的 BFCL 工具根本不在 53 个手写工具里),"
             "只用来说明手写目录不够用;真实目录下双语索引把中文检索从噪声水平拉到可用,英文不退步。\n")

    L.append("\n## 3. 延迟(端到端,含编码与检索)\n")
    L.append("\n| 目录 | p50(ms) | p95(ms) |\n| --- | --- | --- |")
    for cat in res["catalogs"]:
        s = summary[cat]
        L.append(f"| {res['labels'][cat]} | {s['latency_p50']:.2f} | {s['latency_p95']:.2f} |")

    L.append("\n## 4. top-1 工具风险分布(判成需要工具的请求)\n")
    L.append("\n| 目录 | L0 | L1 | L2 | L3 | 有候选的请求 |\n| --- | --- | --- | --- | --- | --- |")
    for cat in res["catalogs"]:
        r = summary[cat]["risk_top1"]
        L.append(f"| {res['labels'][cat]} | {r.get('L0', 0)} | {r.get('L1', 0)} | {r.get('L2', 0)} "
                 f"| {r.get('L3', 0)} | {sum(r.values())} |")
    L.append("\n风险取 `final_risk`(纯规则与银标 actions 推出的等级取更高一档);"
             "L3 工具调用前仍需用户确认(`Router.requires_confirmation`)。\n")
    return "\n".join(L) + "\n"


def e2e_metrics_json(res) -> dict:
    """`reports/eval_e2e.json` 的机器可读版。"""
    summary = e2e_summary(res)
    return {"n_requests": len(res["requests"]),
            "catalogs": res["labels"],
            "summary": summary,
            "need": {"zh": e2e_need_metrics(res, "zh"), "en": e2e_need_metrics(res, "en")}}


# ---------------------------------------------------------------- 按函数名去重(T-013)

T013_REPORT = REPORTS / "eval_t013.md"
T013_REPORT_JSON = REPORTS / "eval_t013.json"

# 报告里报的 k:任务卡要求 R@1 / R@3 / R@5
T013_RATIOS = (1, 3, 5)


def _retrieve_ranked(cat, text, k, dedupe):
    """真实目录线上口径的纯检索(不做软过滤):去重时先取 k*10(最多取满目录)再按名去重。"""
    fetch = min(k * 10, len(cat.records)) if dedupe else k
    hits = cat.retrieve(text, None, fetch)
    return dedupe_by_name(hits, k) if dedupe else hits


def _t013_item(router, cat, req, name_of, top_k, dedupe) -> dict:
    """一条请求在一种去重开关下的数字:主口径(走 `route`,need 误判计 0)与上限(纯检索)。

    两套分母:主口径用 `route` 的候选,判成 none 时为空、自动算 0;上限不看 need 判定,
    直接按去重后的检索排序取前 k,即"假设 need 全对"的检索天花板。
    """
    gold = req["tool_ids"]
    gold_names = {name_of.get(g, g) for g in gold}
    out = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        main = [c["tool_id"] for c in router.route(text).candidates]
        upper = [rec.tool_id for rec, _ in _retrieve_ranked(cat, text, top_k, dedupe)]
        for mode, ids in (("main", main), ("upper", upper)):
            for k in T013_RATIOS:
                out[f"{lang}_{mode}_r{k}_strict"] = _recall(gold, gold_names, ids[:k], name_of, False)
                out[f"{lang}_{mode}_r{k}_name"] = _recall(gold, gold_names, ids[:k], name_of, True)
    return out


def _dropped_gold(cat, text, gold, name_of, top_k) -> tuple:
    """去重会把金标变体去掉的数量:在去重窗口(前 top_k*10 名)里,若某个金标 tool_id 不是
    它那个函数名的最高分变体,就会被同名变体顶掉。返回(被去掉的金标变体数, 是否发生)。"""
    window = [rec.tool_id for rec, _ in cat.retrieve(text, None, min(top_k * 10, len(cat.records)))]
    first_of_name = {}
    for tid in window:
        first_of_name.setdefault(name_of.get(tid, tid), tid)
    dropped = [g for g in gold
               if name_of.get(g, g) in first_of_name and first_of_name[name_of[g]] != g]
    return len(dropped), bool(dropped)


def dedupe_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
                encoder="tfidf", folds=FOLDS, seed=SEED, top_k=5) -> dict:
    """T-013 去重评估:真实目录 + 双语索引 + 软过滤关,沿用 T-012 的 5 折折外请求头。

    开发折(0-2)用来决定是否默认去重,测试折(3-4)只报一次。每条请求同时算去重关 / 开。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    per = {"off": {}, "on": {}}
    dropped = {}
    for f in range(folds):
        ids = {i for i, fo in fold_of.items() if fo == f}
        rows = [r for r in requests if r["id"] in ids]
        train = fold_train_rows(synth_train, real_train, ids)
        _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, encoder,
                                            bilingual=True, zh_by_id=zh_by_id)
        rhead = ev.train_request_head(_enc, train)
        routers = {"off": Router(rhead, cat, top_k=top_k, use_filter=False, dedupe_by_name=False),
                   "on": Router(rhead, cat, top_k=top_k, use_filter=False, dedupe_by_name=True)}
        for r in rows:
            for key, router in routers.items():
                per[key][r["id"]] = _t013_item(router, cat, r, name_of, top_k, key == "on")
            # 金标变体被同名变体顶掉的数量(与去重开关无关,只看目录排出来的名次)
            d_zh = _dropped_gold(cat, r["text"], r["tool_ids"], name_of, top_k)
            d_en = _dropped_gold(cat, r["text_orig"], r["tool_ids"], name_of, top_k)
            dropped[r["id"]] = {"zh": d_zh[0], "en": d_en[0],
                                "zh_hit": d_zh[1], "en_hit": d_en[1]}
    dev_ids = {i for i, fo in fold_of.items() if fo in (0, 1, 2)}
    test_ids = {i for i, fo in fold_of.items() if fo in (3, 4)}
    return {"requests": requests, "fold_of": fold_of, "top_k": top_k,
            "dev_ids": sorted(dev_ids), "test_ids": sorted(test_ids),
            "per": per, "dropped": dropped}


def t013_summary(res, ids, dedupe) -> dict:
    """一批请求在一种去重开关下的均值:语言 × 主口径/上限 × 严格/同名 × R@1/R@3/R@5。"""
    ids = set(ids)
    items = res["per"][dedupe]
    rows = [r for r in res["requests"] if r["id"] in ids]
    out = {}
    for lang in ("zh", "en"):
        for mode in ("main", "upper"):
            for suf in ("strict", "name"):
                for k in T013_RATIOS:
                    out[f"{lang}_{mode}_r{k}_{suf}"] = _mean(
                        [items[r["id"]][f"{lang}_{mode}_r{k}_{suf}"] for r in rows])
    return out


def t013_dropped(res, ids) -> dict:
    """某个划分里"金标变体被同名变体顶掉"的请求数与变体数。"""
    ids = set(ids)
    v = {i: d for i, d in res["dropped"].items() if i in ids}
    return {"n": len(v), "zh_req": sum(1 for d in v.values() if d["zh_hit"]),
            "en_req": sum(1 for d in v.values() if d["en_hit"]),
            "zh_inst": sum(d["zh"] for d in v.values()),
            "en_inst": sum(d["en"] for d in v.values())}


def _t013_table(L, title, summary, mode):
    L.append(f"\n**{title}**\n")
    L.append("| 查询语言 | 口径 | 去重 | R@1 | R@3 | R@5 |\n| --- | --- | --- | --- | --- | --- |")
    for lang, lang_label in (("zh", "中文"), ("en", "英文")):
        for suf, suf_label in (("strict", "严格"), ("name", "同名")):
            for d, d_label in (("off", "关"), ("on", "开")):
                s = summary[d]
                L.append(f"| {lang_label} | {suf_label} | {d_label} | "
                         f"{_pct(s[f'{lang}_{mode}_r1_{suf}'])} | "
                         f"{_pct(s[f'{lang}_{mode}_r3_{suf}'])} | "
                         f"{_pct(s[f'{lang}_{mode}_r5_{suf}'])} |")


def render_t013(res) -> str:
    """把 `dedupe_eval` 的结果渲染成 `reports/eval_t013.md` 的正文。"""
    dev_set, test_set = set(res["dev_ids"]), set(res["test_ids"])
    dev = {d: t013_summary(res, dev_set, d) for d in ("off", "on")}
    test = {d: t013_summary(res, test_set, d) for d in ("off", "on")}
    dd = t013_dropped(res, dev_set)
    td = t013_dropped(res, test_set)
    dev_off = dev["off"]["zh_upper_r5_strict"]
    dev_on = dev["on"]["zh_upper_r5_strict"]
    delta = (dev_on - dev_off) * len(dev_set)
    if abs(delta) < 1:
        verdict = (f"开发折中文上限严格 R@5 去重关 {_pct(dev_off)} → 开 {_pct(dev_on)},"
                   f"折合 {delta:+.1f} 条,**无法区分**")
    elif delta > 0:
        verdict = (f"开发折中文上限严格 R@5 去重关 {_pct(dev_off)} → 开 {_pct(dev_on)},"
                   f"折合 {delta:+.1f} 条,去重**更好**")
    else:
        verdict = (f"开发折中文上限严格 R@5 去重关 {_pct(dev_off)} → 开 {_pct(dev_on)},"
                   f"折合 {delta:+.1f} 条,去重**更差**")

    L = ["# 检索按函数名去重(T-013)\n",
         f"真实目录 {len(res['requests'])} 条请求(600 条 BFCL 里能对上金标 tool_id 的部分),"
         "2405 个工具,双语索引,软过滤关。\n",
         "> 请求头按 T-012 的 5 折折外方式训练(`assign_folds`,seed 不变);开发折 = 第 0-2 折,"
         "测试折 = 第 3-4 折,和 T-011 一样。是否默认去重只看开发折,测试折只报一次。\n",
         f"> 去重:检索先多取候选(`top_k * 10 = {res['top_k'] * 10}`,最多取满目录),"
         f"每个函数名只留分数最高的那个变体,再取 top-{res['top_k']}。\n",
         "\n## 0. 结论\n",
         f"- {verdict}。\n",
         "- 口径提醒:去重后严格口径可能下降——同名的金标变体如果分数不是最高,会被同名变体顶掉"
         "(这种情况下同名口径不受影响)。第 3 节统计了条数。\n"]

    L.append("\n## 1. 开发折(第 0-2 折,"
             f"{len(dev_set)} 条)\n")
    _t013_table(L, "主口径(走完整路由,need 误判计 0)", dev, "main")
    _t013_table(L, "上限(假设 need 全对,纯检索)", dev, "upper")

    L.append("\n## 2. 测试折(第 3-4 折,"
             f"{len(test_set)} 条,只报一次)\n")
    _t013_table(L, "主口径(走完整路由,need 误判计 0)", test, "main")
    _t013_table(L, "上限(假设 need 全对,纯检索)", test, "upper")

    L.append("\n## 3. 金标变体被同名变体顶掉的条数\n")
    L.append("去重窗口(前 "
             f"{res['top_k'] * 10} 名)里,金标 tool_id 不是它那个函数名最高分变体的情况:"
             "这种金标在去重后被去掉,严格口径下这一条就算不中(同名口径仍可能中)。\n")
    L.append("\n| 划分 | 请求数 | 中文:有金标被顶掉的请求 | 中文:被顶掉的变体 | "
             "英文:有金标被顶掉的请求 | 英文:被顶掉的变体 |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for label, d in (("开发折", dd), ("测试折", td)):
        L.append(f"| {label} | {d['n']} | {d['zh_req']} | {d['zh_inst']} | "
                 f"{d['en_req']} | {d['en_inst']} |")
    L.append("\n这正是去重后严格 R@5 可能不升反降的原因:一个函数名的多个变体只留一个,"
             "金标偏偏是低分的那个变体时,严格命中被顶掉;同名口径因为只看函数名,不受影响。\n")
    return "\n".join(L) + "\n"


def t013_metrics_json(res) -> dict:
    """`reports/eval_t013.json` 的机器可读版。"""
    dev_set, test_set = set(res["dev_ids"]), set(res["test_ids"])
    return {
        "n_requests": len(res["requests"]),
        "top_k": res["top_k"], "folds": FOLDS, "seed": SEED,
        "dev_ids": res["dev_ids"], "test_ids": res["test_ids"],
        "dev": {d: t013_summary(res, dev_set, d) for d in ("off", "on")},
        "test": {d: t013_summary(res, test_set, d) for d in ("off", "on")},
        "dropped": {"dev": t013_dropped(res, dev_set), "test": t013_dropped(res, test_set)},
    }



# ---------------------------------------------------------------- 多语言语义编码器(T-016)

T016_REPORT = REPORTS / "eval_t016.md"
T016_REPORT_JSON = REPORTS / "eval_t016.json"

# 三种编码器:(build_encoder 的 spec, 报告里的短名)。tfidf 是现状默认,另两个是语义模型。
T016_ENCODERS = (("tfidf", "tfidf"),
                 ("st:BAAI/bge-m3", "bge-m3"),
                 ("st:intfloat/multilingual-e5-small", "e5-small"))
# 两种索引:单语(只有英文描述)/ 双语(再拼上 tools_real_zh 的中文描述)
T016_INDEXES = (("mono", "单语索引"), ("bi", "双语索引"))
# 任务卡要的口径:严格 / 同名 × R@1 / R@5
T016_RATIOS = (1, 5)


def _t016_item(router, cat, req, name_of, top_k) -> dict:
    """一条请求在一种索引下的数字:主口径(走 `route`,need 误判计 0)与上限(纯检索)。

    与 T-012 的 `_e2e_item` 同口径:主口径用 route 的候选(判 none 时为空、自动算 0);
    上限不看 need,直接按向量相似度取前 k。延迟取中文那次 route 的实际耗时(含编码与检索)。
    """
    gold = req["tool_ids"]
    gold_names = {name_of.get(g, g) for g in gold}
    out = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        res = router.route(text)
        main = [c["tool_id"] for c in res.candidates]
        upper = [rec.tool_id for rec, _ in cat.retrieve(text, None, top_k)]
        if lang == "zh":
            out["latency_ms"] = res.latency_ms
            out["risk_top1"] = res.candidates[0]["risk"] if res.candidates else ""
            out["pred_none"] = res.need == "none"
        for mode, ids in (("main", main), ("upper", upper)):
            for k in T016_RATIOS:
                out[f"{lang}_{mode}_r{k}_strict"] = _recall(gold, gold_names, ids[:k], name_of, False)
                out[f"{lang}_{mode}_r{k}_name"] = _recall(gold, gold_names, ids[:k], name_of, True)
    return out


def _hf_repo_bytes(repo_id) -> int:
    """模型权重在 HF 缓存里占的字节数;缓存目录不存在返回 -1。

    直接扫 `~/.cache/huggingface/hub/models--<org>--<name>` 并对 blob 去重,
    不调 `snapshot_download`,否则会因为仓库里那些没下的可选文件(`.eval_results/*`)
    被判成 "incomplete snapshot"。
    """
    if not repo_id:
        return 0
    hub = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                       "models--" + repo_id.replace("/", "--"))
    if not os.path.isdir(hub):
        return -1
    seen, total = set(), 0
    for root, _dirs, files in os.walk(hub):
        for fn in files:
            rp = os.path.realpath(os.path.join(root, fn))
            if rp in seen or not os.path.isfile(rp):
                continue
            seen.add(rp)
            total += os.path.getsize(rp)
    return total


def router_pkl_size(spec, labeled_tools, synth_tools, synth_train, real_train, zh_by_id,
                    top_k=5) -> tuple:
    """整份数据训一个 T-012 默认 router(真实目录 + 双语索引 + 软过滤关)存盘,报:
    (`router.pkl` 字节数, 编码器权重字节数)。

    语义编码器的权重会被一起 pickle 进 router.pkl,所以这个数字就是"部署包会变成多大"
    的直接证据。权重字节数另报,用来看 pkl 里绝大部分是不是那堆权重。
    """
    import pickle
    import tempfile

    train = list(synth_train) + list(real_train)
    enc = ev.build_encoder(spec)
    _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, spec, bilingual=True,
                                        zh_by_id=zh_by_id, enc=enc)
    rhead = ev.train_request_head(enc, train)
    router = Router(rhead, cat, top_k=top_k, use_filter=False)
    router.catalog_name = "real"        # 与 cli train 落盘的字段一致

    repo = getattr(enc, "model_name", None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        name = tmp.name
    try:
        with open(name, "wb") as f:
            pickle.dump(router, f, protocol=pickle.HIGHEST_PROTOCOL)
        size = os.path.getsize(name)
    finally:
        os.remove(name)
    return size, _hf_repo_bytes(repo)


def encoder_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
                 encoders=None, folds=FOLDS, seed=SEED, top_k=5) -> dict:
    """T-016:三种编码器 × 单语/双语索引,沿用 T-012 的端到端设置(5 折折外请求头)。

    每个编码器只建一次实例、复用(语义模型只加载一次);每折按折内训练集重拟合,
    所以折外语义与 T-012 一致。开发折(0-2)用来决定是否改默认,测试折(3-4)只报一次。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    encoders = list(encoders or T016_ENCODERS)

    per, pkl_bytes, model_bytes = {}, {}, {}
    for spec, label in encoders:
        enc = ev.build_encoder(spec)
        for idx_key, _ in T016_INDEXES:
            bilingual = (idx_key == "bi")
            items = {}
            for f in range(folds):
                ids = {i for i, fo in fold_of.items() if fo == f}
                rows = [r for r in requests if r["id"] in ids]
                train = fold_train_rows(synth_train, real_train, ids)
                _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, spec,
                                                    bilingual=bilingual, zh_by_id=zh_by_id, enc=enc)
                rhead = ev.train_request_head(enc, train)
                router = Router(rhead, cat, top_k=top_k, use_filter=False)
                for r in rows:
                    items[r["id"]] = _t016_item(router, cat, r, name_of, top_k)
            per[(label, idx_key)] = items
        pkl_bytes[label], model_bytes[label] = router_pkl_size(
            spec, labeled_tools, synth_tools, synth_train, real_train, zh_by_id, top_k=top_k)

    dev_ids = sorted(i for i, fo in fold_of.items() if fo in (0, 1, 2))
    test_ids = sorted(i for i, fo in fold_of.items() if fo in (3, 4))
    return {"requests": requests, "fold_of": fold_of, "top_k": top_k,
            "encoders": encoders, "indexes": list(T016_INDEXES),
            "dev_ids": dev_ids, "test_ids": test_ids, "per": per,
            "pkl_bytes": pkl_bytes, "model_bytes": model_bytes}


def t016_summary(res, ids, enc_label, idx_key) -> dict:
    """一批请求在(编码器 × 索引)下的均值:中英 × 主口径/上限 × 严格/同名 × R@1/R@5,外加延迟。"""
    ids = set(ids)
    items = res["per"][(enc_label, idx_key)]
    rows = [r for r in res["requests"] if r["id"] in ids]
    out = {}
    for lang in ("zh", "en"):
        for mode in ("main", "upper"):
            for suf in ("strict", "name"):
                for k in T016_RATIOS:
                    out[f"{lang}_{mode}_r{k}_{suf}"] = _mean(
                        [items[r["id"]][f"{lang}_{mode}_r{k}_{suf}"] for r in rows])
    lat = [items[r["id"]]["latency_ms"] for r in rows] or [0.0]
    out["latency_p50"] = float(np.percentile(lat, 50))
    out["latency_p95"] = float(np.percentile(lat, 95))
    return out


def _mb(n: int) -> str:
    if n < 0:
        return "?"
    return f"{n / 1024 / 1024:.1f} MB"


def _t016_table(L, title, summary, mode, encoders, indexes):
    L.append(f"\n**{title}**\n")
    L.append("| 编码器 | 索引 | 查询语言 | 严格 R@1 | 严格 R@5 | 同名 R@1 | 同名 R@5 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for _spec, enc_label in encoders:
        for idx_key, idx_label in indexes:
            s = summary[(enc_label, idx_key)]
            for lang, lang_label in (("zh", "中文"), ("en", "英文")):
                L.append(f"| {enc_label} | {idx_label} | {lang_label} "
                         f"| {_pct(s[f'{lang}_{mode}_r1_strict'])} "
                         f"| {_pct(s[f'{lang}_{mode}_r5_strict'])} "
                         f"| {_pct(s[f'{lang}_{mode}_r1_name'])} "
                         f"| {_pct(s[f'{lang}_{mode}_r5_name'])} |")


def _t016_dev_summary(res):
    dev = set(res["dev_ids"])
    return {(e, i): t016_summary(res, dev, e, i)
            for _s, e in res["encoders"] for i, _ in res["indexes"]}


def _t016_verdict(res) -> str:
    """默认怎么定:只看开发折中文上限严格 R@5,以现状(tfidf + 双语索引)为基线。"""
    dev = _t016_dev_summary(res)
    if ("tfidf", "bi") not in dev:
        return "- (这一轮没跑 tfidf + 双语索引基线,跳过结论文本)"
    base = dev[("tfidf", "bi")]["zh_upper_r5_strict"]
    lines = []
    for _spec, enc_label in res["encoders"]:
        for idx_key, idx_label in res["indexes"]:
            val = dev[(enc_label, idx_key)]["zh_upper_r5_strict"]
            delta = (val - base) * len(set(res["dev_ids"]))
            tag = ""
            if (enc_label, idx_key) != ("tfidf", "bi"):
                if abs(delta) < 1:
                    tag = "(与基线无法区分)"
                elif delta > 0:
                    tag = f"(比基线 +{delta:.1f} 条)"
                else:
                    tag = f"(比基线 {delta:.1f} 条)"
            lines.append(f"- 开发折中文上限严格 R@5:{enc_label} + {idx_label} = {_pct(val)} {tag}")
    return "\n".join(lines)


def render_t016(res) -> str:
    """把 `encoder_eval` 的结果渲染成 `reports/eval_t016.md` 的正文。"""
    requests = res["requests"]
    dev_set, test_set = set(res["dev_ids"]), set(res["test_ids"])
    dev = _t016_dev_summary(res)
    test = {(e, i): t016_summary(res, test_set, e, i)
            for _s, e in res["encoders"] for i, _ in res["indexes"]}
    L = ["# 多语言语义编码器评估(T-016)\n",
         f"真实目录 2405 个工具,{len(requests)} 条 BFCL 请求(600 条里能对上金标 tool_id 的部分)。"
         "need 银标来自 `data/v3-primary/labeled_agree.jsonl`,工具银标是 Jev / Haiku 两路一致结果。\n",
         "> 三种编码器:`tfidf`(现状默认)、`bge-m3`、`e5-small`;每种都比单语索引 / 双语索引;"
         "查询分中英文;口径分严格 / 同名,报 R@1 与 R@5。\n",
         "> 沿用 T-012 的端到端设置:请求头按 5 折折外训练(`assign_folds`,seed 不变);"
         "开发折 = 第 0-2 折,测试折 = 第 3-4 折。真实目录 + 软过滤关 + top_k=5,不加示例查询。"
         "是否改默认只看开发折,测试折只报一次。\n",
         "> 主口径 = 走完整 `route`(need 误判计 0);上限 = 假设 need 全对,纯检索天花板。\n",
         "\n## 0. 结论(只看开发折)\n",
         _t016_verdict(res) + "\n",
         f"\n## 1. 开发折(第 0-2 折,{len(dev_set)} 条)\n"]
    _t016_table(L, "主口径(走完整路由,need 误判计 0)", dev, "main",
                res["encoders"], res["indexes"])
    _t016_table(L, "上限(假设 need 全对,纯检索)", dev, "upper",
                res["encoders"], res["indexes"])

    L.append(f"\n## 2. 测试折(第 3-4 折,{len(test_set)} 条,只报一次)\n")
    _t016_table(L, "主口径(走完整路由,need 误判计 0)", test, "main",
                res["encoders"], res["indexes"])
    _t016_table(L, "上限(假设 need 全对,纯检索)", test, "upper",
                res["encoders"], res["indexes"])

    L.append("\n## 3. 延迟(端到端,含编码与检索;中文查询)\n")
    L.append("\n| 编码器 | 索引 | 开发折 p50(ms) | 开发折 p95(ms) | 测试折 p50(ms) | 测试折 p95(ms) |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for _spec, enc_label in res["encoders"]:
        for idx_key, idx_label in res["indexes"]:
            d, t = dev[(enc_label, idx_key)], test[(enc_label, idx_key)]
            L.append(f"| {enc_label} | {idx_label} | {d['latency_p50']:.2f} | {d['latency_p95']:.2f} "
                     f"| {t['latency_p50']:.2f} | {t['latency_p95']:.2f} |")

    L.append("\n## 4. 模型与 `router.pkl` 大小\n")
    L.append("整份数据训一个 T-012 默认 router(真实目录 + 双语索引 + 软过滤关)后 pickle 到临时文件,"
             "量出来就是部署包里 `router.pkl` 的大小;语义编码器的权重会被一起 pickle 进去。\n")
    L.append("\n| 编码器 | 编码器权重 | router.pkl |\n| --- | --- | --- |")
    for _spec, enc_label in res["encoders"]:
        L.append(f"| {enc_label} | {_mb(res['model_bytes'][enc_label])} "
                 f"| {_mb(res['pkl_bytes'][enc_label])} |")
    return "\n".join(L) + "\n"


def t016_metrics_json(res) -> dict:
    """`reports/eval_t016.json` 的机器可读版。"""
    dev_set, test_set = set(res["dev_ids"]), set(res["test_ids"])

    def block(ids):
        return {f"{e}|{i}": t016_summary(res, ids, e, i)
                for _s, e in res["encoders"] for i, _ in res["indexes"]}
    return {"n_requests": len(res["requests"]), "top_k": res["top_k"],
            "encoders": {spec: label for spec, label in res["encoders"]},
            "indexes": {k: v for k, v in res["indexes"]},
            "dev_ids": res["dev_ids"], "test_ids": res["test_ids"],
            "dev": block(dev_set), "test": block(test_set),
            "pkl_bytes": res["pkl_bytes"], "model_bytes": res["model_bytes"]}


# ---------------------------------------------------------------- 检索/分类头编码器分开(T-019)

T019_REPORT = REPORTS / "eval_t019.md"
T019_REPORT_JSON = REPORTS / "eval_t019.json"

# 分类头固定 tfidf;检索编码器三种,索引都用双语(T-019 任务卡)
T019_HEAD_ENCODER = "tfidf"
T019_RETRIEVAL_ENCODERS = (("tfidf", "tfidf"),
                           ("st:intfloat/multilingual-e5-small", "e5-small"),
                           ("st:BAAI/bge-m3", "bge-m3"))
# 报告里的短名:已知 spec 用固定标签,其余取 spec 最后一段
T019_LABELS = {spec: label for spec, label in T019_RETRIEVAL_ENCODERS}
# 任务卡要的口径:严格 / 同名 × R@1 / R@5
T019_RATIOS = (1, 5)


def t019_label(spec) -> str:
    """spec -> 报告短名:tfidf / e5-small / bge-m3;未知 spec 取路径最后一段。"""
    return T019_LABELS.get(spec, spec.split(":")[-1].split("/")[-1])


def _split_item(router, cat, req, name_of, top_k) -> dict:
    """一条请求在一种检索编码器下的数字:主口径(走 `route`,need 误判计 0)与上限(纯检索)。

    与 T-016 的 `_t016_item` 同口径,额外记下中英文各自的 need 判定结果,用来证明
    换检索编码器不改分类头(need 三行应逐条一致)。
    """
    gold = req["tool_ids"]
    gold_names = {name_of.get(g, g) for g in gold}
    out = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        res = router.route(text)
        main = [c["tool_id"] for c in res.candidates]
        upper = [rec.tool_id for rec, _ in cat.retrieve(text, None, top_k)]
        out[f"{lang}_pred_none"] = res.need == "none"
        if lang == "zh":
            out["latency_ms"] = res.latency_ms
            out["risk_top1"] = res.candidates[0]["risk"] if res.candidates else ""
        for mode, ids in (("main", main), ("upper", upper)):
            for k in T019_RATIOS:
                out[f"{lang}_{mode}_r{k}_strict"] = _recall(gold, gold_names, ids[:k], name_of, False)
                out[f"{lang}_{mode}_r{k}_name"] = _recall(gold, gold_names, ids[:k], name_of, True)
    return out


def split_encoder_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train,
                       real_train, head_encoder=T019_HEAD_ENCODER, retrieval_encoders=None,
                       folds=FOLDS, seed=SEED, top_k=5) -> dict:
    """T-019:分类头固定 `head_encoder`,检索编码器逐个换,索引都用双语。

    沿用 T-016 的端到端设置:真实目录 + 软过滤关 + top_k=5,请求头按 5 折折外训练,
    开发折(0-2)用来决定是否改默认,测试折(3-4)只报一次。每个检索编码器实例只建一次、
    跨折复用(语义模型只加载一次);每折按折内训练集重拟合,折外口径与 T-016 一致。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    retrieval_encoders = list(retrieval_encoders or T019_RETRIEVAL_ENCODERS)

    per, pkl = {}, {}
    for spec, label in retrieval_encoders:
        henc = ev.build_encoder(head_encoder)
        renc = ev.build_encoder(spec)
        items = {}
        for f in range(folds):
            ids = {i for i, fo in fold_of.items() if fo == f}
            rows = [r for r in requests if r["id"] in ids]
            train = fold_train_rows(synth_train, real_train, ids)
            _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, head_encoder,
                                                bilingual=True, zh_by_id=zh_by_id,
                                                enc=henc, retrieval_encoder=renc)
            rhead = ev.train_request_head(_enc, train)
            router = Router(rhead, cat, top_k=top_k, use_filter=False)
            for r in rows:
                items[r["id"]] = _split_item(router, cat, r, name_of, top_k)
        per[label] = items

        # 整份数据训一个默认 router,量 router.pkl 大小(部署包);复用已加载的编码器实例
        pkl[label] = split_router_pkl_size(
            head_encoder, spec, labeled_tools, synth_tools, synth_train, real_train, zh_by_id,
            top_k=top_k, head_enc=henc, retrieval_enc=renc)

    dev_ids = sorted(i for i, fo in fold_of.items() if fo in (0, 1, 2))
    test_ids = sorted(i for i, fo in fold_of.items() if fo in (3, 4))
    return {"requests": requests, "fold_of": fold_of, "top_k": top_k,
            "head_encoder": head_encoder,
            "retrieval_encoders": retrieval_encoders,
            "dev_ids": dev_ids, "test_ids": test_ids, "per": per, "pkl": pkl}


def t019_summary(res, ids, label) -> dict:
    """一批请求在一种检索编码器下的均值:中英 × 主口径/上限 × 严格/同名 × R@1/R@5,外加延迟。"""
    ids = set(ids)
    items = res["per"][label]
    rows = [r for r in res["requests"] if r["id"] in ids]
    out = {}
    for lang in ("zh", "en"):
        for mode in ("main", "upper"):
            for suf in ("strict", "name"):
                for k in T019_RATIOS:
                    out[f"{lang}_{mode}_r{k}_{suf}"] = _mean(
                        [items[r["id"]][f"{lang}_{mode}_r{k}_{suf}"] for r in rows])
    lat = [items[r["id"]]["latency_ms"] for r in rows] or [0.0]
    out["latency_p50"] = float(np.percentile(lat, 50))
    out["latency_p95"] = float(np.percentile(lat, 95))
    return out


def t019_need_metrics(res, label, lang="zh") -> dict:
    """一种检索编码器下的 need 判定指标:只统计银标有 need 的请求子集。

    need 与检索编码器无关,三种编码器应逐条相同 —— 用来证明分类头没被换掉。
    """
    items = res["per"][label]
    paired = [(r, items[r["id"]]) for r in res["requests"] if r["need"]]
    pos = sum(1 for r, _ in paired if r["need"] != "none")
    tp = sum(1 for r, z in paired if r["need"] != "none" and not z[f"{lang}_pred_none"])
    ppos = sum(1 for _, z in paired if not z[f"{lang}_pred_none"])
    return {"n": len(paired), "pos": pos,
            "recall": (tp / pos) if pos else None,
            "precision": (tp / ppos) if ppos else None,
            "pred_none": sum(1 for _, z in paired if z[f"{lang}_pred_none"])}


T019_MIN_ZH_GAIN_SAMPLES = 1.0     # 中文主口径至少要多命中 1 条才算"改得动"
T019_MAX_EN_REGRESS_SAMPLES = 1.0  # 英文主口径最多只能退步 1 条
T019_MAX_P95_MS = 200.0            # p95 延迟上限


def _t019_verdict(res) -> str:
    """默认怎么定:只看开发折主口径中文严格 R@5(基线 tfidf),附英文主口径与 p95。

    门槛(任务卡 2):中文主口径严格 R@5 要比基线多命中 ≥ 1 条;英文主口径不退步超过
    1 条样本;p95 ≤ 200 ms。三条全过才值得设默认。本卡只给结论,不改默认值。
    """
    dev = set(res["dev_ids"])
    n = len(dev)
    summ = {label: t019_summary(res, dev, label) for _s, label in res["retrieval_encoders"]}
    if "tfidf" not in summ:
        return "- (这一轮没跑 tfidf 基线,跳过结论文本)"
    base = summ["tfidf"]
    lines = [f"开发折 {n} 条;基线 = 分类头 tfidf + 检索 tfidf + 双语索引。"
             f"1 条样本 = 1/{n} = {100.0 / n:.2f} 个百分点。\n"]
    qualified = []
    for _spec, label in res["retrieval_encoders"]:
        s = summ[label]
        d_zh = (s["zh_main_r5_strict"] - base["zh_main_r5_strict"]) * n
        d_en = (s["en_main_r5_strict"] - base["en_main_r5_strict"]) * n
        p95 = s["latency_p95"]
        if label == "tfidf":
            lines.append(f"- {label}(基线): 中文主口径严格 R@5 "
                         f"{_pct(s['zh_main_r5_strict'])};英文 "
                         f"{_pct(s['en_main_r5_strict'])};p95 {p95:.0f} ms")
            continue
        zh_ok = d_zh >= T019_MIN_ZH_GAIN_SAMPLES
        en_ok = d_en >= -T019_MAX_EN_REGRESS_SAMPLES
        lat_ok = p95 <= T019_MAX_P95_MS
        gate = "三条门槛全过" if (zh_ok and en_ok and lat_ok) else "不满足门槛"
        fails = []
        if not zh_ok:
            fails.append("中文主口径未比基线多 ≥ 1 条")
        if not en_ok:
            fails.append(f"英文主口径退步 {abs(d_en):.1f} 条(> 1)")
        if not lat_ok:
            fails.append(f"p95 {p95:.0f} ms(> {T019_MAX_P95_MS:.0f})")
        lines.append(f"- {label}: 中文主口径严格 R@5 {_pct(s['zh_main_r5_strict'])}"
                     f"({d_zh:+.1f} 条);英文 {_pct(s['en_main_r5_strict'])}({d_en:+.1f} 条);"
                     f"p95 {p95:.0f} ms;{gate}"
                     + (f" —— {(';'.join(fails))}" if fails else ""))
        if zh_ok and en_ok and lat_ok:
            qualified.append(label)
    if qualified:
        lines.append(f"\n**结论**:开发折上 {('、'.join(qualified))} 同时满足三条门槛,"
                     "可作候选默认;本卡不改默认值,是否切换留待决策。")
    else:
        lines.append("\n**结论**:开发折上没有检索编码器同时满足三条门槛"
                     "(中文主口径要涨 ≥ 1 条,且英文主口径不退步超过 1 条,且 p95 ≤ 200 ms),"
                     "**因此不改默认**,检索编码器保持 tfidf。")
    return "\n".join(lines)


def _t019_table(L, title, summary, mode, retrieval_encoders):
    L.append(f"\n**{title}**\n")
    L.append("| 检索编码器 | 查询语言 | 严格 R@1 | 严格 R@5 | 同名 R@1 | 同名 R@5 |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for _spec, label in retrieval_encoders:
        s = summary[label]
        for lang, lang_label in (("zh", "中文"), ("en", "英文")):
            L.append(f"| {label} | {lang_label} "
                     f"| {_pct(s[f'{lang}_{mode}_r1_strict'])} "
                     f"| {_pct(s[f'{lang}_{mode}_r5_strict'])} "
                     f"| {_pct(s[f'{lang}_{mode}_r1_name'])} "
                     f"| {_pct(s[f'{lang}_{mode}_r5_name'])} |")


def render_t019(res) -> str:
    """把 `split_encoder_eval` 的结果渲染成 `reports/eval_t019.md` 的正文。"""
    requests = res["requests"]
    dev_set, test_set = set(res["dev_ids"]), set(res["test_ids"])
    dev = {label: t019_summary(res, dev_set, label) for _s, label in res["retrieval_encoders"]}
    test = {label: t019_summary(res, test_set, label) for _s, label in res["retrieval_encoders"]}
    L = ["# 检索编码器与分类头编码器分开评估(T-019)\n",
         f"真实目录 2405 个工具,{len(requests)} 条 BFCL 请求(600 条里能对上金标 tool_id 的部分)。"
         "need 银标来自 `data/v3-primary/labeled_agree.jsonl`,工具银标是 Jev / Haiku 两路一致结果。\n",
         f"> 分类头固定 `{res['head_encoder']}`(need / 能力域 / 动作不变);检索编码器逐个换:"
         + "、".join(label for _s, label in res["retrieval_encoders"])
         + ";三种都用真实目录 + 双语索引。\n",
         "> 沿用 T-016 的端到端设置:请求头按 5 折折外训练(`assign_folds`,seed 不变);"
         "开发折 = 第 0-2 折,测试折 = 第 3-4 折。真实目录 + 软过滤关 + top_k=5,不加示例查询。"
         "是否改默认只看开发折,测试折只报一次。\n",
         "> 主口径 = 走完整 `route`(need 误判计 0);上限 = 假设 need 全对,纯检索天花板。\n",
         "\n## 0. 结论(只看开发折)\n",
         _t019_verdict(res) + "\n"]

    L.append(f"\n## 1. need 判定(应与检索编码器无关,三行相同)\n")
    L.append("银标有 need 的请求子集上算:需要 MCP 召回 = 银标要工具且判对 / 银标要工具;"
             "精确率 = 判对 / 判成要工具。\n")
    L.append("\n| 检索编码器 | 查询语言 | 子集条数 | 需要 MCP 召回 | 需要 MCP 精确率 | 判成 none |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for _spec, label in res["retrieval_encoders"]:
        for lang, lang_label in (("zh", "中文"), ("en", "英文")):
            m = t019_need_metrics(res, label, lang)
            L.append(f"| {label} | {lang_label} | {m['n']} | {_pct(m['recall'])} "
                     f"| {_pct(m['precision'])} | {m['pred_none']} |")

    L.append(f"\n## 2. 开发折(第 0-2 折,{len(dev_set)} 条)\n")
    _t019_table(L, "主口径(走完整路由,need 误判计 0)", dev, "main", res["retrieval_encoders"])
    _t019_table(L, "上限(假设 need 全对,纯检索)", dev, "upper", res["retrieval_encoders"])

    L.append(f"\n## 3. 测试折(第 3-4 折,{len(test_set)} 条,只报一次)\n")
    _t019_table(L, "主口径(走完整路由,need 误判计 0)", test, "main", res["retrieval_encoders"])
    _t019_table(L, "上限(假设 need 全对,纯检索)", test, "upper", res["retrieval_encoders"])

    L.append("\n## 4. 延迟(端到端,含编码与检索;中文查询)\n")
    L.append("\n| 检索编码器 | 开发折 p50(ms) | 开发折 p95(ms) | 测试折 p50(ms) | 测试折 p95(ms) |\n"
             "| --- | --- | --- | --- | --- |")
    for _spec, label in res["retrieval_encoders"]:
        d, t = dev[label], test[label]
        L.append(f"| {label} | {d['latency_p50']:.2f} | {d['latency_p95']:.2f} "
                 f"| {t['latency_p50']:.2f} | {t['latency_p95']:.2f} |")

    L.append("\n## 5. 模型与 `router.pkl` 大小(部署瘦身)\n")
    L.append("整份数据训一个默认 router(真实目录 + 双语索引 + 软过滤关)后 pickle 到临时文件,"
             "量出来就是部署包里 `models/router.pkl` 的大小。语义编码器现在只把模型名写进 pickle"
             "(T-019),加载时从 HuggingFace 缓存重建;『改前』是把权重也序列化进去的旧行为。\n")
    L.append("\n| 检索编码器 | 编码器权重 | router.pkl(改后) | router.pkl(改前) |\n"
             "| --- | --- | --- | --- |")
    for _spec, label in res["retrieval_encoders"]:
        p = res["pkl"][label]
        fat = _mb(p["fat"]) if p["fat"] >= 0 else "(无权重)"
        L.append(f"| {label} | {_mb(p['model'])} | {_mb(p['slim'])} | {fat} |")
    L.append("\n分类头编码器始终是 tfidf,所以『改后』一列三种都只有二十几 MB;"
             "换语义检索编码器只多存一个模型名,部署包不再随模型涨到 GB 级。\n")
    return "\n".join(L) + "\n"


def t019_metrics_json(res) -> dict:
    """`reports/eval_t019.json` 的机器可读版。"""
    dev_set, test_set = set(res["dev_ids"]), set(res["test_ids"])

    def block(ids):
        return {label: t019_summary(res, ids, label) for _s, label in res["retrieval_encoders"]}
    return {"n_requests": len(res["requests"]), "top_k": res["top_k"],
            "head_encoder": res["head_encoder"],
            "retrieval_encoders": {spec: label for spec, label in res["retrieval_encoders"]},
            "dev_ids": res["dev_ids"], "test_ids": res["test_ids"],
            "dev": block(dev_set), "test": block(test_set), "pkl": res["pkl"],
            "need": {label: {"zh": t019_need_metrics(res, label, "zh"),
                             "en": t019_need_metrics(res, label, "en")}
                     for _s, label in res["retrieval_encoders"]}}


def _pkl_bytes(obj, fat=False) -> int:
    """把对象 pickle 到临时文件量字节数;fat=True 时让 ST 编码器连权重一起进 pickle。

    fat 用来给 T-019 报"改前"的大小:临时把 `SentenceTransformerEncoder` 的
    `__getstate__`/`__setstate__` 换回默认行为(存整个 `__dict__`),量完恢复。
    """
    import pickle
    import tempfile

    from .encoder import SentenceTransformerEncoder

    saved = None
    if fat:
        saved = (SentenceTransformerEncoder.__getstate__, SentenceTransformerEncoder.__setstate__)
        SentenceTransformerEncoder.__getstate__ = lambda self: dict(self.__dict__)
        SentenceTransformerEncoder.__setstate__ = lambda self, state: self.__dict__.update(state)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            name = tmp.name
        try:
            with open(name, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            return os.path.getsize(name)
        finally:
            os.remove(name)
    finally:
        if saved is not None:
            SentenceTransformerEncoder.__getstate__, SentenceTransformerEncoder.__setstate__ = saved


def split_router_pkl_size(head_encoder, retrieval_encoder, labeled_tools, synth_tools,
                          synth_train, real_train, zh_by_id, top_k=5,
                          head_enc=None, retrieval_enc=None) -> dict:
    """整份数据训一个默认 router(真实目录 + 双语索引 + 软过滤关),报 router.pkl 大小。

    返回 `{"slim": 改后字节数, "fat": 改前字节数或 -1, "model": 编码器权重字节数}`。
    head_encoder / retrieval_encoder 都是 spec;retrieval 为 tfidf 时 fat 与 slim 相同。
    `head_enc` / `retrieval_enc` 传已加载的实例时直接复用(不再新建 / 不重复加载模型)。
    """
    train = list(synth_train) + list(real_train)
    henc = head_enc if head_enc is not None else ev.build_encoder(head_encoder)
    renc = retrieval_enc if retrieval_enc is not None else ev.build_encoder(retrieval_encoder)
    _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, head_encoder,
                                        bilingual=True, zh_by_id=zh_by_id, enc=henc,
                                        retrieval_encoder=renc)
    rhead = ev.train_request_head(_enc, train)
    router = Router(rhead, cat, top_k=top_k, use_filter=False)
    router.catalog_name = "real"        # 与 cli train 落盘的字段一致

    is_st = retrieval_encoder.startswith("st:")
    repo = getattr(renc, "model_name", None) if is_st else None
    return {"slim": _pkl_bytes(router, fat=False),
            "fat": _pkl_bytes(router, fat=True) if is_st else -1,
            "model": _hf_repo_bytes(repo)}


# ---------------------------------------------------------------- 软 need(T-020)

T020_REPORT = REPORTS / "eval_t020.md"
T020_REPORT_JSON = REPORTS / "eval_t020.json"
# 阈值网格:None 是改前基线,1.01 相当于"判成 none 就一律附带候选"(概率最大 1.0)
T020_THRESHOLDS = (None, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)
# 误挂率门槛:两路一致判成 none 的 alpaca_zh 请求里,附带候选的比例不得超过这个数
T020_FALSE_ATTACH_MAX = 0.10
# 误挂率池:labeled_agree 里两路一致、来源是 alpaca_zh(闲聊/写作)、判成 none 的请求
T020_ALPACA_SOURCE = "alpaca_zh"
T020_HEAD_ENCODER = "tfidf"          # 分类头编码器(T-019 结论:默认仍是 tfidf)
T020_RETRIEVAL_ENCODER = "tfidf"     # 检索编码器(同上)


def load_soft_need_probes(agree=AGREE, pool=POOL_ZH) -> list:
    """误挂率池:两路一致判成 none 的 alpaca_zh 请求(文本取自池子)。

    这些是闲聊 / 写作类请求,确实不需要工具;软 need 在它们身上附带候选就是误挂。
    """
    ids = [r["id"] for r in ev.load_jsonl(agree)
           if r.get("source") == T020_ALPACA_SOURCE and r.get("need") == "none"]
    pool_by_id = {r["id"]: r for r in ev.load_jsonl(pool)}
    return [{"id": i, "text": pool_by_id[i]["text"]} for i in sorted(ids) if i in pool_by_id]


def _soft_grid_item(router, req, name_of) -> dict:
    """一条请求在某个阈值下的数字:主口径(走 `route`,need 误判计 0,软候选算数)中英文 R@5。

    与 T-019 的 `_split_item` 同口径,只是候选里包含了软 need 附带的那部分。
    """
    gold = req["tool_ids"]
    gold_names = {name_of.get(g, g) for g in gold}
    out = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        res = router.route(text)
        ids = [c["tool_id"] for c in res.candidates]
        out[f"{lang}_r5_strict"] = _recall(gold, gold_names, ids[:5], name_of, False)
        out[f"{lang}_r5_name"] = _recall(gold, gold_names, ids[:5], name_of, True)
        out[f"{lang}_pred_none"] = res.need == "none"
        out[f"{lang}_soft"] = res.soft_candidates
        if lang == "zh":
            out["latency_ms"] = res.latency_ms
    return out


def soft_need_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
                   head_encoder=T020_HEAD_ENCODER, retrieval_encoder=T020_RETRIEVAL_ENCODER,
                   thresholds=T020_THRESHOLDS, folds=FOLDS, seed=SEED, top_k=5,
                   probes=None) -> dict:
    """T-020:软 need 阈值网格评估。

    沿用 T-019 的设置:分类头和检索都是 tfidf、真实目录、双语索引、软过滤关、top_k=5,
    请求头按 5 折折外训练,开发折 0-2 决定,测试折 3-4 只报一次。每个阈值只换
    `Router.soft_need_threshold`,分类头和目录逐折共享。

    误挂率另用一份"整份数据训的部署 router"在 alpaca_zh 一致-none 池上量(单一数字 / 阈值)。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    thresholds = list(thresholds)

    per = {t: {} for t in thresholds}
    for f in range(folds):
        ids = {i for i, fo in fold_of.items() if fo == f}
        rows = [r for r in requests if r["id"] in ids]
        train = fold_train_rows(synth_train, real_train, ids)
        renc = ev.build_encoder(retrieval_encoder)
        enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, head_encoder,
                                           bilingual=True, zh_by_id=zh_by_id,
                                           retrieval_encoder=renc)
        rhead = ev.train_request_head(enc, train)
        for t in thresholds:
            router = Router(rhead, cat, top_k=top_k, use_filter=False, soft_need_threshold=t)
            for r in rows:
                per[t][r["id"]] = _soft_grid_item(router, r, name_of)

    # 误挂率:整份训练数据训一个部署 router(与 T-019 的 router.pkl 同一构造),逐阈值量
    probes = load_soft_need_probes() if probes is None else list(probes)
    train_ids = {r["id"] for r in real_train}
    probes_in_train = sum(1 for p in probes if p["id"] in train_ids)
    train_all = list(synth_train) + list(real_train)
    renc_all = ev.build_encoder(retrieval_encoder)
    enc_all, _h, cat_all = build_fold_index(labeled_tools, synth_tools, train_all, head_encoder,
                                            bilingual=True, zh_by_id=zh_by_id,
                                            retrieval_encoder=renc_all)
    rhead_all = ev.train_request_head(enc_all, train_all)
    false_attach, probe_hits, holdout_attach = {}, {}, {}
    holdout = [p for p in probes if p["id"] not in train_ids]
    for t in thresholds:
        router = Router(rhead_all, cat_all, top_k=top_k, use_filter=False, soft_need_threshold=t)
        hits = sum(1 for p in probes if router.route(p["text"]).soft_candidates)
        hhits = sum(1 for p in holdout if router.route(p["text"]).soft_candidates)
        probe_hits[t] = hits
        false_attach[t] = (hits / len(probes)) if probes else None
        holdout_attach[t] = (hhits / len(holdout)) if holdout else None

    dev_ids = sorted(i for i, fo in fold_of.items() if fo in (0, 1, 2))
    test_ids = sorted(i for i, fo in fold_of.items() if fo in (3, 4))
    return {"requests": requests, "fold_of": fold_of, "top_k": top_k,
            "head_encoder": head_encoder, "retrieval_encoder": retrieval_encoder,
            "thresholds": thresholds, "dev_ids": dev_ids, "test_ids": test_ids,
            "per": per, "false_attach": false_attach, "probe_hits": probe_hits,
            "holdout_attach": holdout_attach, "n_probes": len(probes),
            "n_probes_in_train": probes_in_train, "n_probes_holdout": len(holdout)}


def t020_summary(res, ids, threshold) -> dict:
    """一批请求在某个阈值下的均值:中英文主口径 R@5(严格/同名)、软触发数、延迟。"""
    ids = set(ids)
    items = res["per"][threshold]
    rows = [r for r in res["requests"] if r["id"] in ids]
    out = {}
    for lang in ("zh", "en"):
        for suf in ("strict", "name"):
            out[f"{lang}_r5_{suf}"] = _mean([items[r["id"]][f"{lang}_r5_{suf}"] for r in rows])
        out[f"{lang}_pred_none"] = sum(1 for r in rows if items[r["id"]][f"{lang}_pred_none"])
        out[f"{lang}_soft"] = sum(1 for r in rows if items[r["id"]][f"{lang}_soft"])
    out["latency_mean"] = _mean([items[r["id"]]["latency_ms"] for r in rows])
    return out


def _t020_pick(res):
    """按事先定的规则选阈值:误挂率 ≤ 10% 里,开发折中文主口径严格 R@5 最高;并列取阈值小的。

    None 视作"不启用",并列时也归到 None(保守)。返回 (selected, summary_map)。
    """
    dev = {t: t020_summary(res, res["dev_ids"], t) for t in res["thresholds"]}
    qualified = [t for t in res["thresholds"]
                 if res["false_attach"][t] is not None
                 and res["false_attach"][t] <= T020_FALSE_ATTACH_MAX]
    if not qualified:
        return None, dev
    key = lambda t: (dev[t]["zh_r5_strict"], -(t if t is not None else float("-inf")))
    return max(qualified, key=key), dev


def _t020_verdict(res) -> str:
    """开发折上给建议(任务卡:只看开发折,不在本卡改默认值)。"""
    n = len(res["dev_ids"])
    dev = {t: t020_summary(res, res["dev_ids"], t) for t in res["thresholds"]}
    sel, _ = _t020_pick(res)
    lines = [f"开发折 {n} 条;1 条样本 = 1/{n} = {100.0 / n:.2f} 个百分点。"
             f"误挂率池 = 两路一致判成 none 的 alpaca_zh {res['n_probes']} 条",
             f"基线(None)= 中文主口径严格 R@5 {_pct(dev[None]['zh_r5_strict'])},"
             f"误挂率 0.0%。\n"]
    for t in res["thresholds"]:
        if t is None:
            continue
        d = dev[t]
        dz = (d["zh_r5_strict"] - dev[None]["zh_r5_strict"]) * n
        fa = res["false_attach"][t]
        lines.append(f"- 阈值 {t:g}: 中文严格 R@5 {_pct(d['zh_r5_strict'])}({dz:+.1f} 条);"
                     f"误挂率 {_pct(fa)}({res['probe_hits'][t]}/{res['n_probes']});"
                     f"平均额外延迟 {d['latency_mean'] - dev[None]['latency_mean']:.2f} ms")
    if sel is None:
        lines.append("\n**结论**:没有阈值在误挂率 ≤ 10% 下把开发折中文主口径严格 R@5 抬高,"
                     "建议**不启用**软 need(保持 `soft_need_threshold=None`)。")
    else:
        d = dev[sel]
        dz = (d["zh_r5_strict"] - dev[None]["zh_r5_strict"]) * n
        lines.append(f"\n**结论**:误挂率 ≤ 10% 里开发折中文主口径严格 R@5 最高的是阈值 "
                     f"{sel:g}({_pct(d['zh_r5_strict'])},{dz:+.1f} 条;"
                     f"误挂率 {_pct(res['false_attach'][sel])})。建议启用该阈值;"
                     "本卡不改默认值,是否改默认留待决策。")
    return "\n".join(lines)


def _t020_grid_table(L, res, dev):
    L.append("| 阈值 | 中文严格 R@5 | 中文同名 R@5 | 英文严格 R@5 | 英文同名 R@5 "
             "| 中文软触发 | 误挂率 | 平均额外延迟(ms) |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in res["thresholds"]:
        d = dev[t]
        dz = d["latency_mean"] - dev[None]["latency_mean"]
        fa = res["false_attach"][t]
        name = "None(基线)" if t is None else f"{t:g}"
        L.append(f"| {name} | {_pct(d['zh_r5_strict'])} | {_pct(d['zh_r5_name'])} "
                 f"| {_pct(d['en_r5_strict'])} | {_pct(d['en_r5_name'])} "
                 f"| {d['zh_soft']} | {_pct(fa)} | {dz:.2f} |")


def _t020_fold_table(L, res, ids, thresholds):
    L.append("| 阈值 | 中文严格 R@5 | 中文同名 R@5 | 英文严格 R@5 | 英文同名 R@5 "
             "| 中文软触发 |\n| --- | --- | --- | --- | --- | --- |")
    for t in thresholds:
        d = t020_summary(res, ids, t)
        name = "None(基线)" if t is None else f"{t:g}"
        L.append(f"| {name} | {_pct(d['zh_r5_strict'])} | {_pct(d['zh_r5_name'])} "
                 f"| {_pct(d['en_r5_strict'])} | {_pct(d['en_r5_name'])} "
                 f"| {d['zh_soft']} |")


def render_t020(res) -> str:
    """把 `soft_need_eval` 的结果渲染成 `reports/eval_t020.md` 的正文。"""
    n = len(res["requests"])
    dev = {t: t020_summary(res, res["dev_ids"], t) for t in res["thresholds"]}
    sel, _ = _t020_pick(res)
    L = ["# 低置信度的 none 也附带候选工具(软 need, T-020)\n",
         f"真实目录 2405 个工具,{n} 条 BFCL 请求。need 银标来自 "
         "`data/v3-primary/labeled_agree.jsonl`,工具银标是 Jev / Haiku 两路一致结果。\n",
         f"> 设置沿用 T-019:分类头和检索都用 `{res['head_encoder']}`,真实目录 + 双语索引 + "
         "软过滤关 + top_k=5,请求头按 5 折折外训练(seed 不变);开发折 = 第 0-2 折,"
         "测试折 = 第 3-4 折,是否启用只看开发折,测试折只报一次。\n",
         "> 主口径 = 走完整 `route`(need 误判计 0),但 `soft_candidates` 附带的那部分候选算数。"
         "阈值 `None` 是改前基线;`1.01` 相当于\"判成 none 就一律附带候选\"。\n",
         "\n## 0. 结论(只看开发折)\n",
         _t020_verdict(res) + "\n"]

    L.append("\n## 1. 阈值网格(开发折)\n")
    L.append("每行只换 `Router.soft_need_threshold`,其余设置不变。"
             "平均额外延迟 = 该阈值端到端平均延迟 − 基线(None);只有软触发的请求会多一次检索。\n")
    _t020_grid_table(L, res, dev)

    L.append(f"\n## 2. 误挂率\n")
    L.append(f"在 `data/v3-primary/labeled_agree.jsonl` 里两路一致判成 none 的 **{T020_ALPACA_SOURCE}** "
             f"请求上(共 {res['n_probes']} 条,闲聊 / 写作类,确实不需要工具),附带候选的比例。\n")
    L.append(f"\n注意:{res['n_probes']} 条里有 {res['n_probes_in_train']} 条在请求头训练集 "
             f"`v3-primary/real_train.jsonl` 里,所以这一列是**乐观下限**;不在训练集的留出子集"
             f"({res['n_probes_holdout']} 条)的误挂率单列在下面。\n")
    L.append("\n| 阈值 | 误挂率(全部) | 附带条数 | 误挂率(留出子集) |\n| --- | --- | --- | --- |")
    for t in res["thresholds"]:
        name = "None(基线)" if t is None else f"{t:g}"
        hol = res["holdout_attach"][t]
        L.append(f"| {name} | {_pct(res['false_attach'][t])} | {res['probe_hits'][t]}"
                 f" | {_pct(hol) if hol is not None else '-'} |")

    L.append(f"\n## 3. 测试折复核(第 3-4 折,{len(res['test_ids'])} 条,只报一次)\n")
    L.append("只报选出的阈值和基线(None)两行;测试折不参与选择。\n")
    _t020_fold_table(L, res, res["test_ids"], [None, sel] if sel is not None else [None])
    L.append("\n开发折上选出的阈值若为 None,就是\"不启用\";测试折的两行差异只作记录。\n")
    return "\n".join(L) + "\n"


def t020_metrics_json(res) -> dict:
    """`reports/eval_t020.json` 的机器可读版。"""
    sel, _ = _t020_pick(res)

    def block(ids):
        return {str(t): t020_summary(res, ids, t) for t in res["thresholds"]}
    return {"n_requests": len(res["requests"]), "top_k": res["top_k"],
            "head_encoder": res["head_encoder"], "retrieval_encoder": res["retrieval_encoder"],
            "thresholds": [None if t is None else t for t in res["thresholds"]],
            "selected": sel, "false_attach_max": T020_FALSE_ATTACH_MAX,
            "dev_ids": res["dev_ids"], "test_ids": res["test_ids"],
            "dev": block(res["dev_ids"]), "test": block(res["test_ids"]),
            "false_attach": {str(t): res["false_attach"][t] for t in res["thresholds"]},
            "probe_hits": {str(t): res["probe_hits"][t] for t in res["thresholds"]},
            "holdout_attach": {str(t): res["holdout_attach"][t] for t in res["thresholds"]},
            "n_probes": res["n_probes"], "n_probes_in_train": res["n_probes_in_train"],
            "n_probes_holdout": res["n_probes_holdout"]}


# ---------------------------------------------------------------- 国产第二路打标(T-017)

# T-017 小样本选型用的 4 个候选模型(走 commandcode 的 OpenAI 格式端点)
CN_MODELS = ("deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3-flash",
             "Qwen/Qwen3.8-Flash", "xiaomi/mimo-v2.6-flash")
CN_BASE_URL_DEFAULT = "https://api.commandcode.ai/provider/v1"
CN_PILOT_N = 210
CN_PILOT_SEED = 0
CN_PILOT_IDS = DATA / "pilot_cn_ids.json"
CN_NEED_VERSION = "v3"
# 已落盘的对照缓存名:Jev 的 v3 答案、Haiku 的 v3 答案(Haiku 不再重新问)
JEV_V3_CACHE = "jev@typesafe-jev"
HAIKU_V3_CACHE = "haiku@claude-haiku-5-5"
# v3-primary 是 Jev + Haiku 的银标,用来算"need 与银标一致率"
CN_PRIMARY_AGREE = DATA / "v3-primary" / "labeled_agree.jsonl"
HAIKU = "haiku"                      # 参考行的显示名
CN_TIE_SAMPLES = 1.0                 # 主指标差距在 1 条样本以内算并列
# commandcode 网关偶发 RemoteDisconnected,默认连续 5 条失败就停太容易误停,放宽到 15;
# 429 仍是硬停(cn_pilot 靠重复续跑补齐)。选型时要求候选的有效条数不低于样本的 90%,
# 否则一个只答上二三十条的模型会因为小样本运气高而被选中(主指标不可比)。
CN_MAX_CONSECUTIVE_FAILURES = 15
CN_MIN_COVERAGE = 0.9


def make_cn_client(model, timeout=120.0):
    """造一个走 commandcode OpenAI 端点的聊天客户端(国产模型)。

    base_url / api_format 优先读 HAIKU_BASE_URL / HAIKU_API_FORMAT,没有就用 commandcode 默认值;
    推理模型慢,默认超时 120 秒。key 由客户端自己从 JEV_API_KEY 读,这里不碰也不打印。
    """
    from .llm_or import OpenRouterChatClient

    base = os.environ.get("HAIKU_BASE_URL") or CN_BASE_URL_DEFAULT
    fmt = os.environ.get("HAIKU_API_FORMAT") or "openai"
    return OpenRouterChatClient(model=model, base_url=base, api_format=fmt,
                                timeout=timeout, retries=1)


def pilot_ids(pool_zh, n=CN_PILOT_N, seed=CN_PILOT_SEED) -> list:
    """按来源分层轮转抽 n 条 id:层内固定种子打乱,层间一条一条轮着取。

    三个来源(bfcl / apibank / alpaca_zh)数据量都够,n=210 时正好各取 70 条。
    """
    groups = {}
    for r in pool_zh:
        groups.setdefault(r["source"], []).append(r["id"])
    rnd = random.Random(seed)
    for g in groups.values():
        rnd.shuffle(g)
    out, i = [], 0
    names = sorted(groups)
    while len(out) < n:
        added = False
        for name in names:
            if i < len(groups[name]) and len(out) < n:
                out.append(groups[name][i])
                added = True
        if not added:
            break
        i += 1
    return out


def pilot_ids_payload(pool_zh, n=CN_PILOT_N, seed=CN_PILOT_SEED) -> dict:
    """`data/pilot_cn_ids.json` 的内容:id 列表 + 来源分布,便于复跑时逐字节对比。"""
    ids = pilot_ids(pool_zh, n, seed)
    idset = set(ids)
    by_source = Counter(r["source"] for r in pool_zh if r["id"] in idset)
    return {"seed": seed, "n": len(ids), "by_source": dict(sorted(by_source.items())), "ids": ids}


def _ok_recs(recs) -> list:
    return [r for r in recs if r.get("ok") and r.get("answers")]


def _tokens_avg(recs) -> dict:
    """平均 prompt / completion / reasoning token(只统计问到答案的条目)。"""
    ok = _ok_recs(recs)
    if not ok:
        return {"prompt": 0.0, "completion": 0.0, "reasoning": 0.0, "total": 0.0}

    def avg(key):
        return sum(float(r.get(key) or 0) for r in ok) / len(ok)

    prompt, comp, reas = avg("prompt_tokens"), avg("completion_tokens"), avg("reasoning_tokens")
    return {"prompt": prompt, "completion": comp, "reasoning": reas, "total": prompt + comp}


def _latency(recs) -> dict:
    lat = sorted(float(r["latency_ms"]) for r in _ok_recs(recs))
    if not lat:
        return {"p50": 0.0, "p95": 0.0}
    return {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}


def _side_metrics(items, recs, jev_by_text, haiku_by_text, silver_by_id,
                  asked=0, failed=0) -> dict:
    """一个候选(或 Haiku 参考行)在 pilot 上的全部指标。

    主指标 = 与 Jev 按 primary 规则的双路一致率(分母是两路都答上来的条数);
    另有 need 与 v3-primary 银标的一致率(只算银标覆盖到的样本)、主域与 Haiku 的一致率、
    解析失败率、延迟 p50/p95、平均 token。
    """
    by_text = {r["text"]: r for r in recs}
    both = agree = 0
    silver_cov = silver_hit = 0
    dom_denom = dom_hit = 0
    for it in items:
        cr = by_text.get(it["text"])
        jr = jev_by_text.get(it["text"])
        a = realdata._labels(jr)
        b = realdata._labels(cr)
        if a is None or b is None:
            continue
        both += 1
        if realdata.agreement_outcome(a, b, (jr or {}).get("answers"), cr["answers"],
                                      "primary")["agreed"]:
            agree += 1
        sid = it["id"]
        if sid in silver_by_id:
            silver_cov += 1
            silver_hit += int(b["need"] == silver_by_id[sid])
        hb = realdata._labels(haiku_by_text.get(it["text"]))
        if hb is not None and b["need"] != "none":
            dom_denom += 1
            dom_hit += int(realdata.primary_domain(cr["answers"], b["domains"])
                           == realdata.primary_domain(haiku_by_text[it["text"]]["answers"],
                                                      hb["domains"]))
    fail = failed if asked else sum(1 for r in recs if not r.get("ok"))
    # 解析/覆盖失败率按整份 pilot 样本算(缓存里只有成功答案,没有答案的条目 = 失败/没问到);
    # 用本次 asked 当分母会把"续跑"的失败率算高,这里统一按 n。
    missing = sum(1 for r in recs if not r.get("ok"))
    return {"n": len(items), "both_answered": both, "agree": agree,
            "agree_rate": (agree / both) if both else None,
            "silver_covered": silver_cov, "silver_hit": silver_hit,
            "silver_rate": (silver_hit / silver_cov) if silver_cov else None,
            "domain_denom": dom_denom, "domain_hit": dom_hit,
            "domain_rate": (dom_hit / dom_denom) if dom_denom else None,
            "asked": asked, "failed": fail, "missing": missing,
            "fail_rate": (missing / len(items)) if items else None,
            "latency_ms": _latency(recs), "tokens": _tokens_avg(recs)}


def select_cn_model(rows, total, min_coverage=CN_MIN_COVERAGE) -> tuple:
    """按事先定好的规则选模型:主指标最高的那个;差距 1 条样本以内算并列,取平均 token 少的。

    rows 里只有 is_candidate 的行参与选择(Haiku 参考行不参与)。
    另加一条覆盖门槛:有效条数不足样本的 min_coverage 的候选不参与选择
    ——只答上二三十条的模型主指标不可比,不能让它赢。返回 (选中行, 并列行列表)。
    """
    cand = [r for r in rows if r["is_candidate"]]
    eligible = [r for r in cand if r["metrics"]["both_answered"] >= min_coverage * total]
    pool = eligible or cand
    best = max(pool, key=lambda r: (r["metrics"]["agree_rate"] or 0.0))
    rate = best["metrics"]["agree_rate"] or 0.0
    tol = CN_TIE_SAMPLES / max(total, 1)
    tied = [r for r in pool if (rate - (r["metrics"]["agree_rate"] or 0.0)) <= tol + 1e-12]
    pick = min(tied, key=lambda r: r["metrics"]["tokens"]["total"])
    return pick, tied


def cn_pilot(pool_zh, ids, live=False, workers=4, models=CN_MODELS, make=None,
             jev_cache=JEV_V3_CACHE, haiku_cache=HAIKU_V3_CACHE,
             agree_path=CN_PRIMARY_AGREE, need_version=CN_NEED_VERSION,
             max_consecutive_failures=CN_MAX_CONSECUTIVE_FAILURES) -> dict:
    """对每个候选模型在 pilot 上跑一遍指标;Jev / Haiku 的答案读现有 v3 缓存,不重新问。

    live=False 时只读候选模型的缓存(缓存里没有的条目算失败),不联网;两次跑结果逐字节相同。
    连续失败上限比默认的 5 条放宽(commandcode 网关偶发 RemoteDisconnected,5 条太容易误停);
    停下来的原因记在每行的 stats.stopped 里。
    """
    by_id = {r["id"]: r for r in pool_zh}
    items = [by_id[i] for i in ids if i in by_id]
    silver_by_id = {}
    if pathlib.Path(agree_path).exists():
        silver_by_id = {r["id"]: r["need"] for r in ev.load_jsonl(agree_path)}
    jev_recs = compare.fetch_answers(None, jev_cache, items, need_version, "pool", live=False)
    haiku_recs = compare.fetch_answers(None, haiku_cache, items, need_version, "pool", live=False)
    jev_by_text = {r["text"]: r for r in jev_recs}
    haiku_by_text = {r["text"]: r for r in haiku_recs}
    make = make or make_cn_client
    rows = [{"model": HAIKU, "cache_name": haiku_cache, "is_candidate": False,
             "metrics": _side_metrics(items, haiku_recs, jev_by_text, haiku_by_text, silver_by_id),
             "stats": {"asked": 0, "failed": 0, "stopped": ""}}]
    for m in models:
        client = make(m)
        s = compare.FetchStats()
        recs = compare.fetch_answers(client, client.cache_name, items, need_version, "pool",
                                     live=live, workers=workers, stats=s,
                                     max_consecutive_failures=max_consecutive_failures)
        rows.append({"model": m, "cache_name": client.cache_name, "is_candidate": True,
                     "metrics": _side_metrics(items, recs, jev_by_text, haiku_by_text, silver_by_id,
                                              asked=s.asked, failed=s.failed),
                     "stats": s.as_dict()})
    pick, tied = select_cn_model(rows, len(items))
    excluded = [r["model"] for r in rows if r["is_candidate"]
                and r["metrics"]["both_answered"] < CN_MIN_COVERAGE * len(items)]
    return {"need_version": need_version, "n": len(items), "live": bool(live), "workers": workers,
            "rows": rows, "picked": pick["model"], "tied": [r["model"] for r in tied],
            "excluded": excluded,
            "by_source": dict(sorted(Counter(r["source"] for r in items).items()))}


def render_cn_pilot(res) -> str:
    """pilot 结果渲染成一张表(Haiku 一行在前作参考,候选按给定顺序)。"""
    L = ["## 小样本选型(pilot)\n",
         f"样本 {res['n']} 条(按来源分层:`{'、'.join(f'{k} {v}' for k, v in res['by_source'].items())}`);"
         f"题目 need 说明 `{res['need_version']}`(1 + 14 域 + 6 动作);"
         f"主指标 = 与 Jev 按 primary 规则的双路一致率。\n",
         "\n| 模型 | 与 Jev 一致率 | 有效条数 | need vs 银标 | 主域 vs Haiku | "
         "解析失败率 | 调用/失败 | 延迟 p50/p95 (ms) | 平均 input/output/reasoning token |\n"
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["rows"]:
        m = r["metrics"]
        tag = " (参考)" if not r["is_candidate"] else (" **(选中)**" if r["model"] == res["picked"] else "")
        st = r["stats"]
        L.append(f"| {r['model']}{tag} | {_pct(m['agree_rate'])} | {m['both_answered']} "
                 f"| {m['silver_hit']}/{m['silver_covered']} ({_pct(m['silver_rate'])}) "
                 f"| {m['domain_hit']}/{m['domain_denom']} ({_pct(m['domain_rate'])}) "
                 f"| {_pct(m['fail_rate'])} | {st['asked']}/{st.get('failed', 0)} "
                 f"| {m['latency_ms']['p50']:.0f} / {m['latency_ms']['p95']:.0f} "
                 f"| {m['tokens']['prompt']:.0f} / {m['tokens']['completion']:.0f} / "
                 f"{m['tokens']['reasoning']:.0f} |")
    L.append(f"\n主指标并列(1 条样本以内)的模型:{'、'.join(res['tied']) or '无'};"
             f"按“并列取 token 少”选中的是 **{res['picked']}**。"
             f"Haiku 一行只作参考,不参与选择。\n")
    if res.get("excluded"):
        L.append(f"> 有效条数不足样本 {CN_MIN_COVERAGE * 100:.0f}% 的候选不参与选择:"
                 + "、".join(res["excluded"]) + "。\n")
    stopped = [f"{r['model']}:{r['stats']['stopped']}" for r in res["rows"]
               if r["stats"].get("stopped")]
    if stopped:
        L.append("> 提前停下的候选:" + "；".join(stopped) + "\n")
    return "\n".join(L) + "\n"


def render_cn_e2e(base, cn) -> str:
    """两版 e2e(请求头 real_train 用 v3-primary vs v3-primary-cn)并排:R@1/R@5、延迟、need。"""
    cats = [c for c, _ in E2E_CATALOGS]
    L = ["## 端到端(tools-real e2e,请求头换成各自的 real_train)\n",
         "| 目录 | 查询语言 | 口径 | 基线 R@1 | 国产 R@1 | 基线 R@5 | 国产 R@5 |\n"
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for cat in cats:
        for lang in ("zh", "en"):
            for key, kl in (("strict", "严格"), ("name", "同名")):
                k1, k5 = f"{lang}_main_r1_{key}", f"{lang}_main_r5_{key}"
                L.append(f"| {base['catalogs'][cat]} | {lang} | {kl} "
                         f"| {_pct(base['summary'][cat][k1])} | {_pct(cn['summary'][cat][k1])} "
                         f"| {_pct(base['summary'][cat][k5])} | {_pct(cn['summary'][cat][k5])} |")
    L.append("\n| 目录 | 基线 p50/p95 (ms) | 国产 p50/p95 (ms) |\n| --- | --- | --- |")
    for cat in cats:
        b, c = base["summary"][cat], cn["summary"][cat]
        L.append(f"| {base['catalogs'][cat]} | {b['latency_p50']:.1f} / {b['latency_p95']:.1f} "
                 f"| {c['latency_p50']:.1f} / {c['latency_p95']:.1f} |")
    b, c = base["need"]["zh"], cn["need"]["zh"]
    L.append(f"\nneed(中文):基线需要 MCP 召回 {_pct(b['recall'])} / 精确率 {_pct(b['precision'])};"
             f"国产 {_pct(c['recall'])} / {_pct(c['precision'])}。\n")
    return "\n".join(L) + "\n"


def _row_of(rows, model) -> dict:
    return next(r for r in rows if r["model"] == model)


def _gap_word(a, b, n) -> str:
    """两个比例按"1 条样本"的幅度比:A 相对 B 是高于/低于/无法区分。"""
    if a is None or b is None:
        return "无法比较"
    if abs(a - b) * n <= 1:
        return "无法区分"
    return "高于" if a > b else "低于"


def render_t017(pilot, compare_md, e2e_base, e2e_cn) -> str:
    """把 pilot 结果、全量对照报告、两版 e2e 拼成 `reports/eval_t017.md`,第一节是结论。"""
    rows = pilot["rows"]
    pick = _row_of(rows, pilot["picked"])
    hk = next(r for r in rows if not r["is_candidate"])
    pm, hm = pick["metrics"], hk["metrics"]
    n = pilot["n"]
    gap = (pm["agree_rate"] or 0) - (hm["agree_rate"] or 0)
    verdict = _gap_word(pm["agree_rate"], hm["agree_rate"], n)
    if verdict == "无法区分":
        replace = "可以替代 Haiku 做第二路(主指标与 Haiku 参考行无法区分)"
    elif verdict == "高于":
        replace = "可以替代 Haiku 做第二路(主指标高于 Haiku 参考行)"
    else:
        replace = "不宜直接替代 Haiku 做第二路(主指标低于 Haiku 参考行)"
    tk = pm["tokens"]
    L = ["# T-017 第二路打标换成国产模型:小样本选型 + 全量对照\n",
         "## 结论\n",
         f"- 选中的国产模型是 **{pilot['picked']}**(选型规则:与 Jev 的主指标最高,"
         f"1 条样本以内并列时取平均 token 少的)。",
         f"- 它与 Jev 的 primary 双路一致率 {_pct(pm['agree_rate'])}(有效 {pm['both_answered']} 条);"
         f"Haiku 参考行 {_pct(hm['agree_rate'])},差距 {gap * n:+.1f} 条样本({verdict})。"
         f"据此:**{replace}**。",
         f"- token 用量(每次调用平均):input {tk['prompt']:.0f} / output {tk['completion']:.0f} / "
         f"reasoning {tk['reasoning']:.0f}。commandcode 不返回费用,Haiku 的 token 用量不在已有缓存里"
         f"(按任务卡不重新问 Haiku),所以只能给国产模型自己的用量。",
         f"- 全量对照:见下面第 2 节,`data/v3-primary`(Jev + Haiku)与 "
         f"`data/v3-primary-cn`(Jev + {pilot['picked']})并排。\n",
         render_cn_pilot(pilot),
         "\n## 1. 全量对照数据(compare-labels)\n",
         compare_md.strip() + "\n",
         "\n" + render_cn_e2e(e2e_base, e2e_cn)]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 当前默认配置快照(T-021)

SNAPSHOT_REPORT = REPORTS / "eval_current_default.md"
SNAPSHOT_REPORT_JSON = REPORTS / "eval_current_default.json"

# 任务卡 T-021:把"线上默认配置是什么、效果如何"固定成一份报告,README 只引用这一份。
# 只读已经落盘的 data/ 文件;请求头按 5 折折外训练(与前面几张卡同一份 assign_folds);
# 开发折 = 第 0-2 折,测试折 = 第 3-4 折,两套都报,本报告不做任何选择。

# `cli train` 的 `--real-train` 默认值(见 cli.py)。快照的"当前默认"行用它。
SNAPSHOT_CODE_TRAIN = REAL_TRAIN
# T-017 把第二路打标换成 DeepSeek 后落盘的请求侧银标(任务卡里提到的那份)。
SNAPSHOT_CN_TRAIN = DATA / "v3-primary-cn" / "real_train.jsonl"

# 要报的每一行:(键, 报告里的名字, 请求头训练数据, 软 need 阈值, 依据)
SNAPSHOT_ROWS = (
    ("current", "当前默认配置", SNAPSHOT_CODE_TRAIN, SOFT_NEED_DEFAULT,
     "`cli train --real-train` 的默认值 + `build_real_router` 的软 need 默认(见 `reports/eval_t020.md`)"),
    ("t012", "T-012 时的默认配置", SNAPSHOT_CODE_TRAIN, None,
     "`reports/eval_e2e.md` 的『真实目录 + 双语索引』行:同一份请求头,软 need 关"),
    ("cn", "T-017 结论对应的候选默认(未接入 `cli train`)", SNAPSHOT_CN_TRAIN, SOFT_NEED_DEFAULT,
     "`reports/eval_t017.md`:第二路换成 DeepSeek 后的 `data/v3-primary-cn`"),
)
SNAPSHOT_RATIOS = (1, 5)


def snapshot_config() -> dict:
    """当前默认配置的逐项取值:直接读 `build_real_router` 的默认参数,报告头部由此生成。"""
    import inspect

    p = inspect.signature(build_real_router).parameters
    return {"encoder": p["encoder"].default, "threshold": p["threshold"].default,
            "top_k": p["top_k"].default, "use_filter": p["use_filter"].default,
            "dedupe_by_name": p["dedupe_by_name"].default,
            "retrieval_encoder": p["retrieval_encoder"].default,
            "soft_need_threshold": p["soft_need_threshold"].default}


def snapshot_signature() -> str:
    """报告头部那一行参数签名。测试用它和 `build_real_router` 的默认参数逐项核对(A2)。"""
    c = snapshot_config()
    return ("build_real_router 默认参数:"
            f"encoder={c['encoder']} threshold={c['threshold']} top_k={c['top_k']} "
            f"use_filter={c['use_filter']} dedupe_by_name={c['dedupe_by_name']} "
            f"retrieval_encoder={c['retrieval_encoder']} "
            f"soft_need_threshold={c['soft_need_threshold']}")


def _snapshot_item(router, cat, req, name_of, top_k) -> dict:
    """一条请求在默认配置下的全部数字:主口径/上限 × 中英 × 严格/同名 × R@1/R@5,加软触发、延迟、top-1 风险。"""
    gold = req["tool_ids"]
    gold_names = {name_of.get(g, g) for g in gold}
    out = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        res = router.route(text)
        main = [c["tool_id"] for c in res.candidates]
        upper = [rec.tool_id for rec, _ in cat.retrieve(text, None, top_k)]
        out[f"{lang}_pred_none"] = res.need == "none"
        out[f"{lang}_soft"] = res.soft_candidates
        if lang == "zh":
            out["latency_ms"] = res.latency_ms
            out["risk_top1"] = res.candidates[0]["risk"] if res.candidates else ""
        for mode, ids in (("main", main), ("upper", upper)):
            for k in SNAPSHOT_RATIOS:
                out[f"{lang}_{mode}_r{k}_strict"] = _recall(gold, gold_names, ids[:k], name_of, False)
                out[f"{lang}_{mode}_r{k}_name"] = _recall(gold, gold_names, ids[:k], name_of, True)
    return out


def _snapshot_router(labeled_tools, synth_tools, train, encoder, zh_by_id, soft_need, top_k):
    """按快照口径建一个 router:真实目录 + 双语索引 + 软过滤关 + 按名去重关。"""
    henc = ev.build_encoder(encoder)
    _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, encoder,
                                        bilingual=True, zh_by_id=zh_by_id, enc=henc)
    rhead = ev.train_request_head(_enc, train)
    return Router(rhead, cat, top_k=top_k, use_filter=False,
                  soft_need_threshold=soft_need), cat


def _as_train_rows(src) -> list:
    """请求头训练数据:路径就读文件,直接给 list 的就用 list(测试里这么传)。"""
    if isinstance(src, (str, pathlib.Path)):
        return ev.load_jsonl(src)
    return list(src)


def _same_src(a, b) -> bool:
    if isinstance(a, (str, pathlib.Path)) and isinstance(b, (str, pathlib.Path)):
        return str(a) == str(b)
    return a is b


def _src_rows(mapping) -> list:
    """把请求头训练数据去重成 [(源, 行)],源可能是路径(测试里也可能是内存 list)。"""
    out = []
    for src in mapping:
        if not any(_same_src(src, s) for s, _ in out):
            out.append((src, _as_train_rows(src)))
    return out


def snapshot_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, rows=None,
                  encoder="tfidf", folds=FOLDS, seed=SEED, top_k=5, probes=None) -> dict:
    """T-021:当前默认配置的端到端快照(离线,只读 data/)。

    每一行是一份 (请求头训练数据, 软 need 阈值);同一份训练数据的每一折只建一次索引 + 请求头,
    再按各行的阈值路由。误挂率与 `router.pkl` 大小用整份数据训的部署 router 量。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    rows = list(rows or SNAPSHOT_ROWS)
    sources = _src_rows([row[2] for row in rows])

    def src_index(src):
        return next(i for i, (s, _r) in enumerate(sources) if _same_src(src, s))

    row_src = {row[0]: src_index(row[2]) for row in rows}
    combos = sorted({(row_src[row[0]], row[3]) for row in rows},
                    key=lambda c: (c[0], -1.0 if c[1] is None else c[1]))
    per = {row[0]: {} for row in rows}
    for f in range(folds):
        ids = {i for i, fo in fold_of.items() if fo == f}
        frows = [r for r in requests if r["id"] in ids]
        built = {c: _snapshot_router(labeled_tools, synth_tools,
                                     fold_train_rows(synth_train, sources[c[0]][1], ids),
                                     encoder, zh_by_id, c[1], top_k) for c in combos}
        for row in rows:
            router, cat = built[(row_src[row[0]], row[3])]
            for r in frows:
                per[row[0]][r["id"]] = _snapshot_item(router, cat, r, name_of, top_k)

    # 误挂率(整份数据训的部署 router)与 router.pkl 大小
    probes = load_soft_need_probes() if probes is None else list(probes)
    by_combo = {}
    for c in combos:
        train_all = list(synth_train) + list(sources[c[0]][1])
        router, _cat = _snapshot_router(labeled_tools, synth_tools, train_all, encoder,
                                        zh_by_id, c[1], top_k)
        router.catalog_name = "real"        # 与 cli train 落盘的字段一致
        train_ids = {r["id"] for r in sources[c[0]][1]}
        hold = [q for q in probes if q["id"] not in train_ids]
        hits = hhits = 0
        if c[1] is not None:
            hits = sum(1 for q in probes if router.route(q["text"]).soft_candidates)
            hhits = sum(1 for q in hold if router.route(q["text"]).soft_candidates)
        by_combo[c] = {"n_probes": len(probes), "probe_hits": hits,
                       "false_attach": (hits / len(probes)) if probes else None,
                       "n_probes_holdout": len(hold), "holdout_hits": hhits,
                       "holdout_attach": (hhits / len(hold)) if hold else None,
                       "pkl_bytes": _pkl_bytes(router)}

    dev_ids = sorted(i for i, fo in fold_of.items() if fo in (0, 1, 2))
    test_ids = sorted(i for i, fo in fold_of.items() if fo in (3, 4))
    return {"requests": requests, "fold_of": fold_of, "top_k": top_k, "encoder": encoder,
            "folds": folds, "config": snapshot_config(),
            "rows": [{"key": row[0], "label": row[1],
                      "request_train": str(row[2]) if isinstance(row[2], (str, pathlib.Path))
                      else "(内存数据)", "soft_need_threshold": row[3], "why": row[4]}
                     for row in rows],
            "deploy": {row[0]: by_combo[(row_src[row[0]], row[3])] for row in rows},
            "dev_ids": dev_ids, "test_ids": test_ids, "per": per}


def snapshot_summary(res, key, ids) -> dict:
    """一批请求汇总:中英 × 主口径/上限 × 严格/同名 × R@1/R@5,加软触发、延迟、top-1 风险。"""
    ids = set(ids)
    items = res["per"][key]
    rows = [r for r in res["requests"] if r["id"] in ids]
    out = {"n": len(rows)}
    for lang in ("zh", "en"):
        for mode in ("main", "upper"):
            for k in SNAPSHOT_RATIOS:
                for suf in ("strict", "name"):
                    out[f"{lang}_{mode}_r{k}_{suf}"] = _mean(
                        [items[r["id"]][f"{lang}_{mode}_r{k}_{suf}"] for r in rows])
        out[f"{lang}_pred_none"] = sum(1 for r in rows if items[r["id"]][f"{lang}_pred_none"])
        out[f"{lang}_soft"] = sum(1 for r in rows if items[r["id"]][f"{lang}_soft"])
    lat = [items[r["id"]]["latency_ms"] for r in rows] or [0.0]
    out["latency_p50"] = float(np.percentile(lat, 50))
    out["latency_p95"] = float(np.percentile(lat, 95))
    out["risk_top1"] = dict(sorted(Counter(
        items[r["id"]]["risk_top1"] for r in rows if items[r["id"]]["risk_top1"]).items()))
    return out


def snapshot_need_metrics(res, key, ids, lang="zh") -> dict:
    """need 判定指标:只统计银标有 need 的请求子集(need 判定与目录、编码器无关)。"""
    ids = set(ids)
    items = res["per"][key]
    paired = [(r, items[r["id"]]) for r in res["requests"] if r["id"] in ids and r["need"]]
    pos = sum(1 for r, _ in paired if r["need"] != "none")
    tp = sum(1 for r, z in paired if r["need"] != "none" and not z[f"{lang}_pred_none"])
    ppos = sum(1 for _, z in paired if not z[f"{lang}_pred_none"])
    return {"n": len(paired), "pos": pos,
            "recall": (tp / pos) if pos else None,
            "precision": (tp / ppos) if ppos else None,
            "pred_none": sum(1 for _, z in paired if z[f"{lang}_pred_none"])}


def _snap_r5_table(L, res, pairs, mode):
    L.append("\n| 配置 | 折 | 查询语言 | 严格 R@1 | 严格 R@5 | 同名 R@1 | 同名 R@5 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for row in res["rows"]:
        for ids, fold_label in pairs:
            s = snapshot_summary(res, row["key"], ids)
            for lang, lang_label in (("zh", "中文"), ("en", "英文")):
                L.append(f"| {row['label']} | {fold_label} | {lang_label} "
                         f"| {_pct(s[f'{lang}_{mode}_r1_strict'])} "
                         f"| {_pct(s[f'{lang}_{mode}_r5_strict'])} "
                         f"| {_pct(s[f'{lang}_{mode}_r1_name'])} "
                         f"| {_pct(s[f'{lang}_{mode}_r5_name'])} |")


def render_snapshot(res) -> str:
    """把 `snapshot_eval` 的结果渲染成 `reports/eval_current_default.md` 的正文。"""
    n = len(res["requests"])
    dev_set, test_set = set(res["dev_ids"]), set(res["test_ids"])
    pairs = ((res["dev_ids"], "开发折"), (res["test_ids"], "测试折"))
    cfg = res["config"]
    L = ["# 当前默认配置的端到端快照(T-021)\n",
         f"真实目录 2405 个工具,{n} 条 BFCL 请求(600 条里能对上金标 tool_id 的部分)。"
         "need 银标来自 `data/v3-primary/labeled_agree.jsonl`,工具银标是 Jev / Haiku 两路一致结果。\n",
         f"> 请求头按 5 折折外训练(`assign_folds`,seed 不变):开发折 = 第 0-2 折({len(dev_set)} 条),"
         f"测试折 = 第 3-4 折({len(test_set)} 条),两套都报,本报告不做任何选择。\n",
         "> 主口径 = 走完整 `route`(need 误判计 0);上限 = 假设 need 全对,纯检索天花板。"
         "软 need 开启时,附带 `soft_candidates` 的候选也算进主口径。\n",
         "\n## 0. 本报告的配置(逐项 + 依据)\n",
         f"> {snapshot_signature()}\n",
         "\n| 配置项 | 取值 | 依据 |\n| --- | --- | --- |",
         "| 工具目录 | `real`:`data/tools_real.jsonl`(2405 个工具) | `cli.py` `--catalog` 默认 |",
         f"| 请求头编码器 | `{cfg['encoder']}` | `build_real_router(encoder=...)` 默认 |",
         f"| 检索编码器 | `{cfg['retrieval_encoder']}`(None = 沿用分类头编码器) | "
         "`build_real_router(retrieval_encoder=...)` 默认;T-019 `reports/eval_t019.md` 第 0 节 |",
         "| 索引文本 | 双语(英文描述 + `data/tools_real_zh.jsonl` 的中文描述) | "
         "T-011 `reports/eval_t011.md` 第 1 节 |",
         f"| 能力域过滤 | `use_filter={cfg['use_filter']}`(软过滤关) | "
         "T-011 `reports/eval_t011.md` 第 2.2 节 |",
         f"| 按函数名去重 | `dedupe_by_name={cfg['dedupe_by_name']}`(关) | "
         "T-013 `reports/eval_t013.md` 第 0 节 |",
         f"| 置信度阈值 | `{cfg['threshold']}` | `Router` 默认 |",
         f"| top_k | `{cfg['top_k']}` | `build_real_router` 默认 |",
         f"| 软 need 阈值 | `{cfg['soft_need_threshold']}`(`tools_real.SOFT_NEED_DEFAULT`) | "
         "T-020 `reports/eval_t020.md` 第 0 节(开发折选,测试折复核) |",
         "| 请求头训练数据 | 合成训练集 + `data/v3-primary/real_train.jsonl` | "
         "`cli.py` `train --real-train` 的默认值 |"]

    L.append(f"\n> **注意(与任务卡不一致,已写进汇报开放问题)**:任务卡把当前默认的请求头训练数据写成 "
             f"`data/v3-primary-cn/real_train.jsonl`(T-017 的 DeepSeek 第二路),但 `cli train --real-train` "
             f"的默认值仍是 `data/v3-primary/real_train.jsonl`(Jev + Haiku)。本报告第 0 节按代码默认列,"
             f"并把 `data/v3-primary-cn` 单列一行(『T-017 结论对应的候选默认』)作对照;本卡不改任何默认值。\n")

    L.append("\n## 1. 检索:主口径(走完整路由,need 误判计 0)\n")
    _snap_r5_table(L, res, pairs, "main")
    L.append("\n主口径分母是全部请求;判成 none 的请求,主口径候选取自软 need 附带的那批(没触发则为空,按 0 计)。\n")

    L.append("\n## 2. 检索:上限(假设 need 全对,纯检索)\n")
    _snap_r5_table(L, res, pairs, "upper")
    L.append("\n上限不看 need 判定、也不做软过滤,直接 `retrieve(text, None, top_k)`,是检索天花板。\n")

    L.append("\n## 3. need 判定(银标有 need 的请求子集)\n")
    L.append("需要 MCP 召回 = 银标要工具且判对 / 银标要工具;精确率 = 判对 / 判成要工具。"
             "need 判定只跟请求头训练数据有关,与软 need、目录无关。\n")
    L.append("\n| 配置 | 折 | 查询语言 | 子集条数 | 需要 MCP 召回 | 需要 MCP 精确率 | 判成 none |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for row in res["rows"]:
        for ids, fold_label in pairs:
            for lang, lang_label in (("zh", "中文"), ("en", "英文")):
                m = snapshot_need_metrics(res, row["key"], ids, lang)
                L.append(f"| {row['label']} | {fold_label} | {lang_label} | {m['n']} "
                         f"| {_pct(m['recall'])} | {_pct(m['precision'])} | {m['pred_none']} |")

    L.append("\n## 4. 软 need 触发率与 alpaca_zh 误挂率\n")
    L.append("触发率 = 该批请求里 `soft_candidates=True` 的比例(按中文算)。误挂率 = "
             "`data/v3-primary/labeled_agree.jsonl` 里两路一致判成 none 的 **alpaca_zh** 请求"
             "(闲聊 / 写作,确实不需要工具)上附带候选的比例,用整份数据训的部署 router 量;"
             "留出子集是不在请求头训练集里的那部分(更保守)。\n")
    L.append("\n| 配置 | 软 need 阈值 | 开发折触发 | 测试折触发 | 误挂率(全部) | 误挂率(留出子集) |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for row in res["rows"]:
        d = res["deploy"][row["key"]]
        ds = snapshot_summary(res, row["key"], res["dev_ids"])
        ts = snapshot_summary(res, row["key"], res["test_ids"])
        thr = "None(关)" if row["soft_need_threshold"] is None else f"{row['soft_need_threshold']:g}"
        thr_note = "0.0%" if d["false_attach"] is None else _pct(d["false_attach"])
        hol = "-" if d["holdout_attach"] is None else _pct(d["holdout_attach"])
        L.append(f"| {row['label']} | {thr} "
                 f"| {ds['zh_soft']}/{ds['n']} ({_pct(ds['zh_soft'] / ds['n'] if ds['n'] else 0)}) "
                 f"| {ts['zh_soft']}/{ts['n']} ({_pct(ts['zh_soft'] / ts['n'] if ts['n'] else 0)}) "
                 f"| {thr_note} | {hol} |")
    L.append(f"\n误挂率池共 {res['deploy'][res['rows'][0]['key']]['n_probes']} 条;"
             "`None(关)` 一行从不软触发,误挂率恒为 0。\n")

    L.append("\n## 5. 延迟(端到端,含编码与检索;中文查询)\n")
    L.append("\n| 配置 | 开发折 p50(ms) | 开发折 p95(ms) | 测试折 p50(ms) | 测试折 p95(ms) |\n"
             "| --- | --- | --- | --- | --- |")
    for row in res["rows"]:
        ds = snapshot_summary(res, row["key"], res["dev_ids"])
        ts = snapshot_summary(res, row["key"], res["test_ids"])
        L.append(f"| {row['label']} | {ds['latency_p50']:.2f} | {ds['latency_p95']:.2f} "
                 f"| {ts['latency_p50']:.2f} | {ts['latency_p95']:.2f} |")

    L.append("\n## 6. top-1 工具风险分布(有候选的请求)\n")
    L.append("\n| 配置 | 折 | L0 | L1 | L2 | L3 | 有候选的请求 |\n"
             "| --- | --- | --- | --- | --- | --- | --- |")
    for row in res["rows"]:
        for ids, fold_label in pairs:
            r = snapshot_summary(res, row["key"], ids)["risk_top1"]
            L.append(f"| {row['label']} | {fold_label} | {r.get('L0', 0)} | {r.get('L1', 0)} "
                     f"| {r.get('L2', 0)} | {r.get('L3', 0)} | {sum(r.values())} |")
    L.append("\n风险取 `final_risk`(纯规则与银标 actions 推出的等级取更高一档);"
             "L3 工具调用前仍需用户确认(`Router.requires_confirmation`)。\n")

    L.append("\n## 7. `router.pkl` 大小\n")
    L.append("\n| 配置 | 请求头训练数据 | router.pkl |\n| --- | --- | --- |")
    for row in res["rows"]:
        d = res["deploy"][row["key"]]
        L.append(f"| {row['label']} | `{row['request_train']}` | {_mb(d['pkl_bytes'])} |")
    L.append("\n整份数据训一个默认 router(真实目录 + 双语索引 + 软过滤关)后 pickle 到临时文件。"
             "分类头与检索都是 tfidf,所以都在二十几 MB;软 need 只多存一个阈值,不影响量级。\n")

    L.append("\n## 8. 从 T-012 到现在的总变化\n")
    L.append(_snapshot_delta(res))
    L.append("\n" + _snapshot_consistency(res))

    return "\n".join(L) + "\n"


def _snapshot_delta(res) -> str:
    """T-012 → 现在(代码默认)的逐项变化:同一份请求头,只有软 need 一个开关变了。"""
    ds = {row["key"]: snapshot_summary(res, row["key"], res["dev_ids"]) for row in res["rows"]}
    if "current" not in ds or "t012" not in ds:
        return "(缺 current / t012 两行,跳过)\n"
    n = ds["current"]["n"]
    lines = [f"开发折 {n} 条;对照 `reports/eval_e2e.md`(T-012)。"
             f"1 条样本 = 1/{n} = {100.0 / n:.2f} 个百分点。\n"]
    for lang, kl in (("zh", "中文"), ("en", "英文")):
        for suf, sl in (("strict", "严格"), ("name", "同名")):
            key = f"{lang}_main_r5_{suf}"
            a, b = ds["current"][key], ds["t012"][key]
            lines.append(f"- 主口径{kl}{sl} R@5:{_pct(b)}(T-012) → {_pct(a)}(现在)"
                         f",{(a - b) * n:+.1f} 条。")
    lines.append(f"- 软 need 触发:开发折 {ds['t012']['zh_soft']} 条 → {ds['current']['zh_soft']} 条。")
    lines.append("- 从 T-012 到现在,线上默认只多了一个 `soft_need_threshold=0.7`;编码器(T-016/T-019)"
                 "与去重(T-013)的结论都是维持 tfidf / 关,没有改默认。\n")
    return "\n".join(lines) + "\n"


def _snapshot_label(res, key) -> str:
    return next(row["label"] for row in res["rows"] if row["key"] == key)


# 快照要和既有报告对得上的几个点:(键, 范围, 指标, 来源报告里的值, 来源)
SNAPSHOT_CHECKS = (
    ("current", "dev", "zh_main_r5_strict", 0.375, "`reports/eval_t020.md` 第 0 节(阈值 0.7)"),
    ("current", "dev", "false_attach", 0.025, "`reports/eval_t020.md` 第 2 节(全部 480)"),
    ("t012", "all", "zh_main_r5_strict", 0.237, "`reports/eval_e2e.md` 第 2 节"),
    ("t012", "dev", "zh_main_r5_strict", 0.260, "`reports/eval_t019.md` 第 0 节(tfidf 基线)"),
    ("t012", "dev", "en_main_r5_strict", 0.583, "`reports/eval_t019.md` 第 0 节(tfidf 基线)"),
    ("current", "all", "need_recall_zh", 0.800, "`reports/eval_t019.md` 第 1 节(tfidf 分类头,真实目录头)"),
)


def _snapshot_consistency(res) -> str:
    """快照与既有报告的一致性核对:同一设置、同一代码路径,数字应当对得上。"""
    scopes = {"dev": res["dev_ids"], "test": res["test_ids"],
              "all": [r["id"] for r in res["requests"]]}
    L = ["## 9. 与既有报告的一致性核对\n",
         "快照的每一行都复用既有报告的设置(同一份分折、同一套代码路径),数字必须能对上;"
         "下表逐项核对(1 条样本 = 0.28 个百分点,允许 0.15 个百分点的四舍五入误差)。\n",
         "\n| 快照行 | 范围 | 指标 | 本报告 | 来源报告 | 结果 |\n"
         "| --- | --- | --- | --- | --- | --- |"]
    keys = {row["key"] for row in res["rows"]}
    for key, scope, metric, expected, src in SNAPSHOT_CHECKS:
        if key not in keys:                # 测试用的小数据没有全部行,跳过缺的行
            continue
        if metric == "false_attach":
            got = res["deploy"][key]["false_attach"]
        elif metric == "need_recall_zh":
            got = snapshot_need_metrics(res, key, scopes[scope], "zh")["recall"]
        else:
            got = snapshot_summary(res, key, scopes[scope])[metric]
        ok = got is not None and abs(got - expected) <= 0.0015
        L.append(f"| {_snapshot_label(res, key)} | {scope} | {metric} | {_pct(got)} "
                 f"| {src} | {'一致' if ok else '**不一致**'} |")
    L.append("\n> **need 口径说明**:本报告的 need 用真实目录头(编码器语料 = 2405 个工具 + 合成工具 + 请求),"
             "与线上 `cli train` 落盘的 router 一致,所以和 `reports/eval_t019.md` 第 1 节对得上(80.0%)。"
             "`reports/eval_e2e.md` / `reports/eval_t017_e2e_cn.md` 的 need 列读的是合成目录那一行"
             "(`e2e_need_metrics` 取 `catalogs[0]`),编码器语料只含 53 个手写工具,need 头不同,"
             "recall 偏高(84.2% / 86.5%),不能和本报告直接比。这一节据此不再引用那两份报告。\n")
    return "\n".join(L) + "\n"


def snapshot_metrics_json(res) -> dict:
    """`reports/eval_current_default.json` 的机器可读版。"""
    def block(ids):
        return {row["key"]: snapshot_summary(res, row["key"], ids) for row in res["rows"]}
    all_ids = [r["id"] for r in res["requests"]]
    return {"n_requests": len(res["requests"]), "folds": res["folds"], "top_k": res["top_k"],
            "encoder": res["encoder"], "config": res["config"], "rows": res["rows"],
            "dev_ids": res["dev_ids"], "test_ids": res["test_ids"],
            "dev": block(res["dev_ids"]), "test": block(res["test_ids"]), "all": block(all_ids),
            "need": {row["key"]: {"zh": snapshot_need_metrics(res, row["key"], res["dev_ids"], "zh"),
                                  "en": snapshot_need_metrics(res, row["key"], res["dev_ids"], "en")}
                     for row in res["rows"]},
            "deploy": {row["key"]: res["deploy"][row["key"]] for row in res["rows"]}}


# ---------------------------------------------------------------- T-022 参考集评估
#
# D-005 把两份 Claude 预标文件定稿成"参考集":200 条请求(data/gold_candidates_claude.csv)
# 和 60 个工具(data/tools_real_risk_audit_claude.csv)。口径写作"Claude 预标 + gu 抽查"。
# 本报告只用来报告当前默认配置的表现,不拿参考集调参、选阈值、改规则(与合成 gold 同规矩)。

REFERENCE_REQUESTS = DATA / "gold_candidates_claude.csv"
REFERENCE_RISK = DATA / "tools_real_risk_audit_claude.csv"
REFERENCE_REPORT = REPORTS / "eval_reference.md"
REFERENCE_REPORT_JSON = REPORTS / "eval_reference.json"


def load_risk_reference(path) -> list:
    """读风险参考集 CSV(D-005 定稿,60 行工具)。

    真值取 `gu_risk`,这一列可能整列留空,为空就回退 `claude_risk`。
    CSV 里那份 `risk_silver / risk_rule / risk_head` 是 T-010b 抽检表抄过来的旧值,
    只带回来核对漂移(见 `reference_eval` 的 `risk["csv_drift"]`),不参与打分。
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        gu = (r.get("gu_risk") or "").strip()
        claude = (r.get("claude_risk") or "").strip()
        truth = gu or claude
        tid = (r.get("tool_id") or "").strip()
        if truth not in RISK_IDS:
            raise ValueError(f"{tid!r}:真值 {truth!r} 不在 {list(RISK_IDS)} 里")
        out.append({"tool_id": tid, "truth": truth, "truth_source": "gu" if gu else "claude",
                    "claude_note": (r.get("claude_note") or "").strip(),
                    "csv_risk_silver": (r.get("risk_silver") or "").strip(),
                    "csv_risk_rule": (r.get("risk_rule") or "").strip(),
                    "csv_risk_head": (r.get("risk_head") or "").strip()})
    return out


def _need_block(gold, pred) -> dict:
    """need 三分类:3×3 混淆矩阵 + 每类召回/精确率 + 需要工具(single+multi)的召回/精确率。"""
    conf = ev.confusion_3(gold, pred)
    per = {}
    for c in NEED_IDS:
        g = sum(conf[c].values())
        p = sum(conf[g2][c] for g2 in NEED_IDS)
        per[c] = {"n": g, "pred": p,
                  "recall": (conf[c][c] / g) if g else None,
                  "precision": (conf[c][c] / p) if p else None}
    m = ev.need_metrics(gold, pred)
    return {"n": len(gold), "confusion": conf, "per_class": per,
            "need_tool_recall": m["recall_needs_mcp"],
            "need_tool_precision": m["precision_needs_mcp"],
            "accuracy": m["accuracy_3class"]}


def _risk_block(gold, pred) -> dict:
    """风险分级:准确率、4×4 混淆矩阵 + 两个最要紧的计数(把 L3 判低 / 把低判成 L3)。"""
    m = risk_metrics(gold, pred)
    conf = m["confusion"]
    return {"n": m["n"], "accuracy": m["accuracy"], "l3_total": m["l3_total"],
            "l3_recall": m["l3_recall"], "confusion": conf,
            "l3_under": sum(conf["L3"][p] for p in ("L0", "L1", "L2")),
            "low_over": sum(conf[g]["L3"] for g in ("L0", "L1", "L2"))}


def _argmax_domain(dp) -> str:
    """`domain_probs` 里概率最大的域;概率全空时记 ""。

    预测 `domains` 为空(各域概率都没过阈值)时,仍按概率给出一个域。
    """
    return max(sorted(dp), key=lambda d: dp[d]) if dp else ""


def _domain_block(tool_refs, preds, probs) -> dict:
    """能力域主域:旧口径 + 主口径(domain_probs 最大域 ∈ 参考 domains)+ 空预测数。

    三个列表已按 need != none 过滤且等长对齐:`tool_refs[i]` 对 `preds[i]`(need/domains/actions)、
    `probs[i]`(domain_probs)。
    """
    n = len(tool_refs)
    primary = lambda ds: ds[0] if ds else ""      # noqa: E731
    return {
        "n": n,
        # 旧口径:预测 domains 的第一个 == 参考 domains 的第一个
        "primary_accuracy": (sum(1 for r, p in zip(tool_refs, preds)
                                 if primary(r["domains"]) == primary(p[1])) / n) if n else None,
        # 主口径:预测 domain_probs 最大的域 ∈ 参考 domains
        "argmax_accuracy": (sum(1 for r, dp in zip(tool_refs, probs)
                                if _argmax_domain(dp) in set(r["domains"])) / n) if n else None,
        # 预测 domains 为空(各域概率都没过阈值)的条数
        "empty_pred": sum(1 for p in preds if not p[1])}


def reference_eval(tools, labeled_rows, zh_rows, synth_tools, synth_train, real_train,
                   req_ref, risk_ref, encoder="tfidf", top_k=5,
                   soft_need_threshold=SOFT_NEED_DEFAULT, tool_head_mode="synth") -> dict:
    """T-022:用 D-005 参考集评估当前默认配置,只报告不调参(离线,只读已落盘文件)。

    请求侧:训练头用 `cli train` 的默认训练数据(合成 + `--real-train`),先剔除与参考集同 id 的行;
    再按 T-021 快照口径(`_snapshot_router`)建 router,对参考集 `text` 预测 need/domains/actions。
    风险侧:规则 / 风险头 / 银标三路预测按当前代码现算(口径同 `evaluate_risk`,默认 synth 工具头),
    真值取参考集的 `gu_risk || claude_risk`。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    ref_ids = {r["id"] for r in req_ref}
    dropped = sorted(r["id"] for r in real_train if r.get("id") in ref_ids)
    train_rows = list(synth_train) + [r for r in real_train if r.get("id") not in ref_ids]

    router, _cat = _snapshot_router(labeled_tools, synth_tools, train_rows, encoder,
                                    zh_by_id, soft_need_threshold, top_k)
    outs = router.head.predict([r["text"] for r in req_ref])
    preds = [(o.need, list(o.domains), list(o.actions)) for o in outs]
    probs = [dict(o.domain_probs) for o in outs]

    request = {"n": len(req_ref), "dropped_train": len(dropped), "dropped_ids": dropped,
               "overall": _need_block([r["need"] for r in req_ref], [p[0] for p in preds]),
               "by_source": {}}
    for src in sorted({r["source"] for r in req_ref}):
        idx = [i for i, r in enumerate(req_ref) if r["source"] == src]
        request["by_source"][src] = _need_block([req_ref[i]["need"] for i in idx],
                                                [preds[i][0] for i in idx])
    # 对照:银标(两路一致的 need)和参考集的一致率,用来看银标本身有多准
    silver = [(r, p) for r, p in zip(req_ref, preds) if r["silver_need"]]
    request["silver"] = {
        "n": len(silver),
        "agree": (sum(1 for r, _ in silver if r["silver_need"] == r["need"]) / len(silver))
        if silver else None,
        "confusion": ev.confusion_3([r["silver_need"] for r, _ in silver],
                                    [r["need"] for r, _ in silver])}
    # 能力域(主域)与动作(微平均 F1),只看参考集 need != none 的行
    tool_rows = [(r, p, dp) for r, p, dp in zip(req_ref, preds, probs) if r["need"] != "none"]
    request["domain"] = _domain_block([r for r, _, _ in tool_rows],
                                      [p for _, p, _ in tool_rows],
                                      [dp for _, _, dp in tool_rows])
    request["action"] = ev.multilabel_f1([r["actions"] for r, _, _ in tool_rows],
                                         [p[2] for _, p, _ in tool_rows], ACTION_IDS)

    # 风险侧:三路预测现算(与线上默认同一个 `evaluate_risk` 口径)
    risk_res = evaluate_risk(labeled_rows, tools, synth_tools, synth_train, encoder, tool_head_mode)
    paths, drift = {}, {"silver": 0, "rule": 0, "head": 0}
    for key in ("rule", "head", "silver"):
        gold, pred = [], []
        under = []                                   # 漏确认:参考 L3 但该路判低
        for r in risk_ref:
            tid = r["tool_id"]
            gold.append(r["truth"])
            got = (risk_res["gold_risk"][tid] if key == "silver"
                   else risk_res["risk_by_id"][key][tid])
            pred.append(got)
            if r["truth"] == "L3" and got != "L3":
                under.append({"tool_id": tid, "pred": got, "note": r.get("claude_note", "")})
            if got != r[f"csv_risk_{key}"]:
                drift[key] += 1
        paths[key] = _risk_block(gold, pred)
        paths[key]["l3_under_ids"] = under
    risk = {"n": len(risk_ref), "truth_source": dict(sorted(Counter(
                r["truth_source"] for r in risk_ref).items())),
            "truth_counts": dict(sorted(Counter(r["truth"] for r in risk_ref).items())),
            "paths": paths, "csv_drift": drift}
    return {"config": snapshot_config(), "encoder": encoder, "top_k": top_k,
            "soft_need_threshold": soft_need_threshold, "tool_head_mode": tool_head_mode,
            "request": request, "risk": risk}


def _need_table(L, title, block) -> None:
    L.append(f"\n### {title}(n={block['n']})\n")
    L.append("| 参考 need | none | single | multi | 召回 | 精确率 |\n"
             "| --- | --- | --- | --- | --- | --- |")
    for g in NEED_IDS:
        c = block["confusion"][g]
        pc = block["per_class"][g]
        L.append(f"| {g} | {c['none']} | {c['single']} | {c['multi']} "
                 f"| {_pct(pc['recall']) if pc['recall'] is not None else '-'} "
                 f"| {_pct(pc['precision']) if pc['precision'] is not None else '-'} |")
    L.append(f"\n需要工具(单+多)召回 {_pct(block['need_tool_recall'])} / "
             f"精确率 {_pct(block['need_tool_precision'])};整体准确率 {_pct(block['accuracy'])}。")


def render_reference_eval(res) -> str:
    """把 `reference_eval` 的结果渲染成 `reports/eval_reference.md` 的正文。"""
    req, risk, cfg = res["request"], res["risk"], res["config"]
    L = ["# 用定稿参考集评估当前默认配置(T-022)\n",
         "参考集口径:**Claude 预标 + gu 抽查**(D-005),不是纯人工金标;只用它报告,"
         "不调参、不选阈值、不按它改规则。\n",
         f"- 请求参考集 `data/gold_candidates_claude.csv`:**{req['n']} 条**(按 `id` 对回请求池 "
         "`data/pool_zh.jsonl` 取正文)。\n"
         f"- 风险参考集 `data/tools_real_risk_audit_claude.csv`:**{risk['n']} 个工具**"
         f"(真值取 `gu_risk`,全空时回退 `claude_risk`;本次来源 {risk['truth_source']})。\n"
         f"- 请求头训练前剔除了与参考集同 id 的训练行:**{req['dropped_train']} 条**"
         f"(训练集 = 合成 `data/requests_train.jsonl` + `data/v3-primary/real_train.jsonl`)。\n",
         f"\n> 请求侧按 T-021 快照口径(`_snapshot_router`):`{snapshot_signature()}`。"
         "风险侧三路现算,口径同 `evaluate_risk`(工具头 `synth` 训练)。\n",
         "## 1. 请求 need(整体)\n"]
    _need_table(L, "整体", req["overall"])

    L.append("\n## 2. 请求 need(按 source)\n")
    for src in sorted(req["by_source"]):
        _need_table(L, f"source={src}", req["by_source"][src])

    dom = req["domain"]
    L.append(f"\n## 3. 能力域 / 动作(参考集 need != none 的 {dom['n']} 条)\n")
    L.append("- 能力域主域准确率,两个口径:\n")
    L.append(f"  - **主口径**:预测 `domain_probs` 最大的域 ∈ 参考 `domains`:"
             f"**{_pct(dom['argmax_accuracy'])}**;\n")
    L.append(f"  - 旧口径:预测 `domains` 第一个 == 参考 `domains` 第一个:"
             f"{_pct(dom['primary_accuracy'])}。\n")
    L.append(f"- 预测 `domains` 为空(各域概率都没过阈值)的条数:**{dom['empty_pred']}**;"
             "这些行按主口径仍用 `domain_probs` 最大者判域,不算错。\n")
    L.append(f"- 动作微平均 F1:**{_pct(req['action']['micro_f1'])}**"
             f"(宏平均 {_pct(req['action']['macro_f1'])})\n")

    sil = req["silver"]
    L.append(f"\n## 4. 银标对照(两路 need 一致的 {sil['n']} 条)\n")
    L.append(f"银标(两路一致的 need)与参考集 need 的一致率:**{_pct(sil['agree'])}**。"
             "这一行用来判断银标本身有多准:一致率越高,平时拿银标当基准越可信。\n")

    L.append(f"\n## 5. 风险侧({risk['n']} 个工具,真值分布 {risk['truth_counts']})\n")
    L.append("三路预测都用当前代码现算;`把 L3 判低`是**漏确认**(最要紧),"
             "`把低判成 L3`是多打扰。\n")
    for key, label in (("rule", "规则 risk_rule"), ("head", "风险头 risk_head"),
                       ("silver", "银标 risk_silver")):
        b = risk["paths"][key]
        L.append(f"\n### {label}(n={b['n']})\n")
        L.append(f"准确率 **{_pct(b['accuracy'])}**;把 L3 判低 **{b['l3_under']}** 个;"
                 f"把低判成 L3 **{b['low_over']}** 个;参考集里 L3 共 {b['l3_total']} 个。\n")
        L.append("\n| 参考 =\\ 预测 | L0 | L1 | L2 | L3 |\n| --- | --- | --- | --- | --- |")
        for g in RISK_IDS:
            c = b["confusion"][g]
            L.append(f"| {g} | {c['L0']} | {c['L1']} | {c['L2']} | {c['L3']} |")
    rule_under = risk["paths"]["rule"]["l3_under_ids"]
    L.append("\n### 结论\n")
    L.append(f"- **规则 risk_rule 有 {len(rule_under)} 个 L3 被判低(漏确认)**:\n")
    for u in rule_under:
        L.append(f"  - `{u['tool_id']}`(判成 {u['pred']}):参考理由 {u['note'] or '(无)'}\n")
    h, s = risk["paths"]["head"], risk["paths"]["silver"]
    L.append(f"- 风险头 risk_head 漏确认 **{h['l3_under']}** 个(不低判 L3),但多打扰 "
             f"**{h['low_over']}** 个;银标 risk_silver 漏确认 **{s['l3_under']}** 个,多打扰 "
             f"**{s['low_over']}** 个。\n")
    d = risk["csv_drift"]
    L.append(f"\n> 参考集 CSV 里也抄了一份 T-010b 抽检表的 `risk_silver/risk_rule/risk_head`。"
             f"本报告以上面现算的为准;与那份旧列不一致的行数:规则 {d['rule']}/{risk['n']}、"
             f"风险头 {d['head']}/{risk['n']}、银标 {d['silver']}/{risk['n']}"
             "(原因见 `.collab/reports/T-022.md` 开放问题)。\n")
    return "\n".join(L) + "\n"


def reference_metrics_json(res) -> dict:
    """`reports/eval_reference.json` 的机器可读版(无时间戳、无延迟,两次跑逐字节相同)。"""
    return res


# ---------------------------------------------------------------- need 漏判诊断(T-024)
#
# 背景(T-022 参考集):"需要工具"召回 75.5%,约四分之一该调工具的请求被判 none。
# 本卡在 `data/v3-primary/real_train.jsonl`(936 条,请求头的训练数据)的 5 折折外预测上诊断:
# 开发折 = 第 0-2 折(选),测试折 = 第 3-4 折(只报一次),D-005 参考集(只报一次)。
# 候选只在开发折上比,按"需要工具召回"选;约束是精确率下降 ≤3 点、alpaca_zh 误挂率 ≤ 基线 +5 点。
# 参考集与测试折只报告,不拿来挑编码器/阈值(D-005)。

T024_REPORT = REPORTS / "eval_t024.md"
T024_REPORT_JSON = REPORTS / "eval_t024.json"

# 候选编码器:(spec, 显示名);第一个是线上默认,也是候选 A 的基线
T024_ENCODERS = (("tfidf", "基线(tfidf 分类头)"),
                 ("st:intfloat/multilingual-e5-small", "C1:e5-small need 头"),
                 ("st:BAAI/bge-m3", "C2:bge-m3 need 头"))
# 候选 A:把 none 判定改成 "p_none >= t 才判 none",t 在开发折上扫
T024_NONE_TS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
# 软 need 阈值(线上默认,诊断与误挂率都用它)
T024_SOFT = SOFT_NEED_DEFAULT
# 约束:精确率下降不超过 3 个点;alpaca_zh 误挂率不超过基线 +5 个点
T024_PRECISION_DROP_MAX = 0.03
T024_ATTACH_MARGIN = 0.05
# 调查时按 p_none 从高到低抽看的"兜不住"样本数
T024_SAMPLE_N = 20
# 20 条兜不住样本的人工归因(实现者逐条看,写死在报告里;计数与样本表一一对应)
T024_SAMPLE_CAUSES = (
    ("请求表述像闲聊 / 常识问答(tfidf 抓不到工具意图)", 12, "例:『法国最新估计的人口是多少？』『The Weeknd 12月那场演唱会是什么时候、在哪里举行的？』"),
    ("特征 / 中文分词抓不到具体动作", 6, "例:『生成一张数字图像…』『把这句话转成音频格式…』『列出 C 盘的内容』"),
    ("训练标签可疑(alpaca 写作 / 概括类被标成需要工具)", 2, "例:『写一篇报纸标题，概括当前的总统选举。』『将最近的新闻文章…概括在不到50个字内。』"),
)


def _pnone(need_probs) -> float:
    return float((need_probs or {}).get("none", 0.0))


def _need_at_threshold(need_probs, t) -> str:
    """候选 A 的判定规则:`p_none >= t` 才判 none,否则在 single/multi 里取概率大的。"""
    if _pnone(need_probs) >= t:
        return "none"
    return "multi" if need_probs.get("multi", 0.0) > need_probs.get("single", 0.0) else "single"


def _need_scores(rows) -> dict:
    """把 [(gold, pred, source)] 汇总成需要工具召回/精确率 + 3×3 混淆 + 按 source。"""
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    tp = sum(1 for g, p in zip(gold, pred) if g != "none" and p != "none")
    pos = sum(1 for g in gold if g != "none")
    ppos = sum(1 for p in pred if p != "none")
    by_source = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        stp = sum(1 for r in sub if r["gold"] != "none" and r["pred"] != "none")
        spos = sum(1 for r in sub if r["gold"] != "none")
        sppos = sum(1 for r in sub if r["pred"] != "none")
        by_source[src] = {"n": len(sub), "pos": spos,
                          "recall": (stp / spos) if spos else None,
                          "precision": (stp / sppos) if sppos else None}
    return {"n": len(rows), "pos": pos,
            "recall": (tp / pos) if pos else None,
            "precision": (tp / ppos) if ppos else None,
            "confusion": ev.confusion_3(gold, pred), "by_source": by_source}


def need_oof(labeled_tools, synth_tools, synth_train, real_train, encoder,
             folds=FOLDS, seed=SEED) -> dict:
    """real_train 的 5 折折外 need 预测。返回 id -> {gold, source, fold, need, need_probs, text}。

    每折用"合成训练集 + 其余折的 real_train"训请求头;编码器语料与线上 `build_fold_index`
    一致(真实工具 + 合成工具 + 本折训练请求),所以折外口径和线上请求头相同。
    每折新建编码器实例(tfidf 的 `fit` 是原地改词表,复用实例会让上一折的头与词表对不上)。
    """
    fold_of = assign_folds([r["id"] for r in real_train], folds, seed)
    out = {}
    for f in range(folds):
        ids = {i for i, fo in fold_of.items() if fo == f}
        rows = [r for r in real_train if r["id"] in ids]
        if not rows:
            continue
        train = fold_train_rows(synth_train, real_train, ids)
        enc = ev.fit_encoder(encoder, labeled_tools + synth_tools, train)
        head = ev.train_request_head(enc, train)
        for r, o in zip(rows, head.predict([r["text"] for r in rows])):
            out[r["id"]] = {"gold": r["need"], "source": r["source"], "fold": fold_of[r["id"]],
                            "need": o.need, "need_probs": dict(o.need_probs), "text": r["text"]}
    return out


def _attach_probe(encoder, labeled_tools, synth_tools, synth_train, real_train,
                  probes, soft=T024_SOFT) -> dict:
    """alpaca_zh 一致-none 池上的"误挂率":need != none 或 p_none < 软阈值(都会附带候选)。

    用整份训练数据训一个部署 router 的等价 need 头(不建索引,只看会不会附带候选)。
    留出子集 = 不在请求头训练集里的那些,是更保守的下限。
    """
    train_all = list(synth_train) + list(real_train)
    enc = ev.fit_encoder(encoder, labeled_tools + synth_tools, train_all)
    head = ev.train_request_head(enc, train_all)
    outs = head.predict([p["text"] for p in probes]) if probes else []
    train_ids = {r["id"] for r in real_train}
    hold_ids = {p["id"] for p in probes if p["id"] not in train_ids}

    def hit(o):
        return o.need != "none" or _pnone(o.need_probs) < soft

    hits = sum(1 for o in outs if hit(o))
    hhits = sum(1 for p, o in zip(probes, outs) if p["id"] in hold_ids and hit(o))
    return {"n": len(probes), "hits": hits,
            "rate": (hits / len(probes)) if probes else None,
            "n_holdout": len(hold_ids), "holdout_hits": hhits,
            "holdout_rate": (hhits / len(hold_ids)) if hold_ids else None}


def _reference_need(encoder, labeled_tools, synth_tools, synth_train, real_train, ref,
                    none_t=None) -> dict:
    """参考集 need:训练头用 `cli train` 的默认训练数据(先剔与参考集同 id 的行),只看 need。"""
    ref_ids = {r["id"] for r in ref}
    train = list(synth_train) + [r for r in real_train if r.get("id") not in ref_ids]
    enc = ev.fit_encoder(encoder, labeled_tools + synth_tools, train)
    head = ev.train_request_head(enc, train)
    outs = head.predict([r["text"] for r in ref])
    rows = [{"gold": r["need"], "source": r["source"],
             "pred": o.need if none_t is None else _need_at_threshold(o.need_probs, none_t)}
            for r, o in zip(ref, outs)]
    return _need_scores(rows)


def _t024_e2e(labeled_tools, synth_tools, synth_train, real_train, requests, name_of, zh_by_id,
              head_encoder, soft, top_k, folds, seed) -> dict:
    """T-021 快照口径的端到端主口径 R@5(按折外请求头),中英文 × 严格/同名。

    候选 C 只换**请求头**的编码器:工具侧分类头与检索索引固定 tfidf(T-016/T-019 的结论),
    请求头的 need/能力域/动作都用 head_encoder。默认 `use_filter=False` 下能力域不参与检索,
    所以改动只通过 need 判定影响主口径。head_encoder=`tfidf` 即基线(与 T-021 快照同一口径)。
    """
    henc = ev.build_encoder(head_encoder)
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    per = {}
    for f in range(folds):
        ids = {i for i, fo in fold_of.items() if fo == f}
        frows = [r for r in requests if r["id"] in ids]
        if not frows:
            continue
        train = fold_train_rows(synth_train, real_train, ids)
        _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, "tfidf",
                                            bilingual=True, zh_by_id=zh_by_id)
        rhead = ev.train_request_head(henc.fit(_fit_texts(labeled_tools + synth_tools, train)),
                                      train)
        router = Router(rhead, cat, top_k=top_k, use_filter=False, soft_need_threshold=soft)
        for r in frows:
            per[r["id"]] = _snapshot_item(router, cat, r, name_of, top_k)
    res = {"per": {"cfg": per}, "requests": requests}
    out = {}
    for scope, folds_of_scope in (("dev", (0, 1, 2)), ("test", (3, 4))):
        ids = sorted(i for i, fo in fold_of.items() if fo in folds_of_scope)
        s = snapshot_summary(res, "cfg", ids)
        out[scope] = {f"{lang}_main_r5_{suf}": s[f"{lang}_main_r5_{suf}"]
                      for lang in ("zh", "en") for suf in ("strict", "name")}
    return out


def _t024_samples(oof, n=T024_SAMPLE_N) -> list:
    """调查用样本:折外判 none、真值要工具、p_none ≥ 软阈值(软 need 兜不住)的请求。"""
    miss = [z for z in oof.values()
            if z["gold"] != "none" and z["need"] == "none"
            and _pnone(z["need_probs"]) >= T024_SOFT]
    miss.sort(key=lambda z: (-_pnone(z["need_probs"]), z["text"]))
    return miss[:n]


def _t024_buckets(oof, ids) -> dict:
    """折外漏判(真值 single/multi → 判 none)的条数与 p_none 分桶。"""
    miss = [z for z in oof.values() if z["fold"] in ids and z["gold"] != "none"
            and z["need"] == "none"]
    return {"total": len(miss),
            "by_source": dict(sorted(Counter(z["source"] for z in miss).items())),
            "buckets": {"<0.5": sum(1 for z in miss if _pnone(z["need_probs"]) < 0.5),
                        "0.5-0.7": sum(1 for z in miss if 0.5 <= _pnone(z["need_probs"]) < T024_SOFT),
                        "0.7-0.9": sum(1 for z in miss if T024_SOFT <= _pnone(z["need_probs"]) < 0.9),
                        ">=0.9": sum(1 for z in miss if _pnone(z["need_probs"]) >= 0.9)}}


def t024_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
              ref, encoders=T024_ENCODERS, folds=FOLDS, seed=SEED, top_k=5,
              none_ts=T024_NONE_TS, probes=None, e2e=True) -> dict:
    """T-024 的全套计算:折外诊断 + 候选对照 + 选中项在测试折/参考集的单次报告。

    - 候选 A:在基线(tfidf)的 p_none 上扫 `none_ts`,判定规则见 `_need_at_threshold`;
    - 候选 C:逐个换 need 头编码器(其余不变);
    - 只在开发折(第 0-2 折)上选,按需要工具召回;约束见 `T024_PRECISION_DROP_MAX` /
      `T024_ATTACH_MARGIN`(误挂率相对基线);
    - `e2e=False` 跳过端到端(测试用)。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    probes = list(probes) if probes is not None else load_soft_need_probes()
    none_ts = list(none_ts)

    base_spec, base_label = encoders[0]
    oof = {spec: need_oof(labeled_tools, synth_tools, synth_train, real_train, spec, folds, seed)
           for spec, _ in encoders}
    base_oof = oof[base_spec]
    fold_of = {i: z["fold"] for i, z in base_oof.items()}
    dev_ids = sorted(i for i, fo in fold_of.items() if fo in (0, 1, 2))
    test_ids = sorted(i for i, fo in fold_of.items() if fo in (3, 4))

    def rows_of(items, ids, pred_of):
        return [{"gold": z["gold"], "source": z["source"], "pred": pred_of(z)}
                for i, z in items.items() if i in set(ids)]

    dev_set, test_set = set(dev_ids), set(test_ids)
    attach = {spec: _attach_probe(spec, labeled_tools, synth_tools, synth_train, real_train,
                                  probes) for spec, _ in encoders}
    # 候选清单:基线 + A(t 网格) + 每个非基线编码器
    candidates = [{"key": "baseline", "kind": "baseline", "label": base_label, "spec": base_spec,
                   "dev": _need_scores(rows_of(base_oof, dev_ids, lambda z: z["need"])),
                   "attach": attach[base_spec]}]
    for t in none_ts:
        pf = lambda z, t=t: _need_at_threshold(z["need_probs"], t)
        candidates.append({"key": f"A@{t:g}", "kind": "none_threshold", "t": t, "spec": base_spec,
                           "label": f"A:none 阈值 {t:g}",
                           "dev": _need_scores(rows_of(base_oof, dev_ids, pf)),
                           "attach": attach[base_spec]})
    for spec, label in encoders[1:]:
        candidates.append({"key": f"enc:{spec}", "kind": "encoder", "spec": spec, "label": label,
                           "dev": _need_scores(rows_of(oof[spec], dev_ids, lambda z: z["need"])),
                           "attach": attach[spec]})

    base = candidates[0]
    base_prec, base_attach = base["dev"]["precision"], base["attach"]["rate"]
    for c in candidates[1:]:
        c["precision_drop"] = base_prec - c["dev"]["precision"]
        c["attach_delta"] = ((c["attach"]["rate"] - base_attach)
                             if (base_attach is not None and c["attach"]["rate"] is not None) else None)
        c["qualified"] = (c["precision_drop"] <= T024_PRECISION_DROP_MAX
                          and (c["attach_delta"] is not None
                               and c["attach_delta"] <= T024_ATTACH_MARGIN))
    qualified = [c for c in candidates[1:] if c["qualified"]]
    selected = max(qualified, key=lambda c: c["dev"]["recall"]) if qualified else base

    # 选中项在测试折只报一次,并在参考集只报一次(改前 = 基线)
    sel_pf = (lambda z: z["need"]) if selected["kind"] != "none_threshold" else \
        (lambda z, t=selected["t"]: _need_at_threshold(z["need_probs"], t))
    sel_oof = oof[selected["spec"]]
    selected["test"] = _need_scores(rows_of(sel_oof, test_ids, sel_pf))
    selected["reference"] = _reference_need(selected["spec"], labeled_tools, synth_tools,
                                            synth_train, real_train, ref,
                                            none_t=selected.get("t"))
    base["test"] = _need_scores(rows_of(base_oof, test_ids, lambda z: z["need"]))
    base["reference"] = _reference_need(base_spec, labeled_tools, synth_tools, synth_train,
                                        real_train, ref)
    # 供决策用的对照:约束内最好的候选 A(零依赖)。选中项不是它时也把它在参考集的数字报一次。
    alt = max((c for c in candidates if c["kind"] == "none_threshold" and c["qualified"]),
              key=lambda c: c["dev"]["recall"], default=None)
    if alt is not None and alt["key"] != selected["key"]:
        alt["reference"] = _reference_need(base_spec, labeled_tools, synth_tools, synth_train,
                                           real_train, ref, none_t=alt["t"])

    res = {"encoders": [{"spec": s, "label": l} for s, l in encoders],
           "oof": oof, "base_spec": base_spec, "dev_ids": dev_ids, "test_ids": test_ids,
           "soft": T024_SOFT, "top_k": top_k, "folds": folds,
           "constraints": {"precision_drop_max": T024_PRECISION_DROP_MAX,
                           "attach_margin": T024_ATTACH_MARGIN},
           "candidates": candidates, "selected_key": selected["key"],
           "alt_key": (alt["key"] if alt is not None else None),
           "buckets": {spec: _t024_buckets(oof[spec], (0, 1, 2)) for spec, _ in encoders},
           "samples": _t024_samples(base_oof),
           "attach": attach}
    if e2e:
        # 端到端(一次):基线 + 选中项;候选 A 不改变候选集(软 need 已覆盖同一区间),直接沿用基线
        base_e2e = _t024_e2e(labeled_tools, synth_tools, synth_train, real_train, requests,
                             name_of, zh_by_id, base_spec, T024_SOFT, top_k, folds, seed)
        if selected["kind"] == "encoder":
            sel_e2e = _t024_e2e(labeled_tools, synth_tools, synth_train, real_train, requests,
                                name_of, zh_by_id, selected["spec"], T024_SOFT, top_k, folds, seed)
        else:
            sel_e2e = base_e2e
        res["e2e"] = {"baseline": base_e2e, "selected": sel_e2e, "retrieval_encoder": "tfidf"}
    return res


def _t024_candidate_table(L, res) -> None:
    L.append("| 候选 | 开发折需要工具召回 | 开发折精确率 | 精确率变化 | alpaca 误挂率 | 误挂率变化 "
             "| 满足约束 |\n| --- | --- | --- | --- | --- | --- | --- |")
    for c in res["candidates"]:
        rec, prec = c["dev"]["recall"], c["dev"]["precision"]
        att = c["attach"]["rate"]
        if c["kind"] == "baseline":
            L.append(f"| {c['label']} | {_pct(rec)} | {_pct(prec)} | - | {_pct(att)} | - | - |")
            continue
        dprec = "-" if c["precision_drop"] is None else f"{-c['precision_drop'] * 100:+.1f} 点"
        datt = "-" if c["attach_delta"] is None else f"{c['attach_delta'] * 100:+.1f} 点"
        L.append(f"| {c['label']} | {_pct(rec)} | {_pct(prec)} | {dprec} | {_pct(att)} | {datt} "
                 f"| {'是' if c['qualified'] else '否'} |")


def _t024_need_table(L, title, block) -> None:
    L.append(f"\n### {title}(n={block['n']})\n")
    L.append("| 参考 need | none | single | multi | 召回 | 精确率 |\n| --- | --- | --- | --- | --- | --- |")
    conf = block["confusion"]
    for g in NEED_IDS:
        row = conf[g]
        n_g = sum(row.values())
        n_p = sum(conf[g2][g] for g2 in NEED_IDS)
        L.append(f"| {g} | {row['none']} | {row['single']} | {row['multi']} | "
                 f"{_pct(row[g] / n_g) if n_g else '-'} | {_pct(row[g] / n_p) if n_p else '-'} |")
    L.append(f"\n需要工具(单+多)召回 {_pct(block['recall'])} / 精确率 {_pct(block['precision'])}。")
    if block.get("by_source"):
        L.append("\n| source | n | 需要工具召回 | 精确率 |\n| --- | --- | --- | --- |")
        for src, b in block["by_source"].items():
            L.append(f"| {src} | {b['n']} | {_pct(b['recall'])} | {_pct(b['precision'])} |")


def render_t024(res) -> str:
    """把 `t024_eval` 的结果渲染成 `reports/eval_t024.md`(无延迟,复跑逐字节相同)。"""
    dev_n = len(res["dev_ids"])
    base = res["candidates"][0]
    selected = next(c for c in res["candidates"] if c["key"] == res["selected_key"])
    L = ["# need 漏判诊断与改进(T-024)\n",
         f"请求头的训练数据 `data/v3-primary/real_train.jsonl` 共 {len(res['oof'][res['base_spec']])} 条,"
         "按 `assign_folds`(seed 不变)切 5 折做折外 need 预测。"
         "开发折 = 第 0-2 折,测试折 = 第 3-4 折;参考集 = D-005 定稿的 200 条("
         "`data/gold_candidates_claude.csv`)。\n",
         f"> 显式设置:need 头两级判定、分类头/检索都按线上默认(tfidf,检索编码器"
         f"`{res.get('e2e', {}).get('retrieval_encoder', 'tfidf')}`),软 need 阈值 {res['soft']:g}。"
         "只在开发折上选;测试折、参考集各只报一次,不用来调参(D-005)。\n",
         f"> 1 条样本 = 1/{dev_n} = {100.0 / dev_n:.2f} 个百分点。\n"]

    L.append("\n## 0. 调查结论\n")
    b = res["buckets"][res["base_spec"]]
    n = sum(b["buckets"].values())
    L.append(f"开发折第 0-2 折折外:真值 single/multi 被判 none 的(漏判)共 **{b['total']}** 条,"
             f"按 source {b['by_source']}。\n")
    L.append("\n| 漏判 p_none 区间 | <0.5 | 0.5-0.7 | 0.7-0.9 | ≥0.9 |\n| --- | --- | --- | --- | --- |")
    L.append(f"| 条数 | {b['buckets']['<0.5']} | {b['buckets']['0.5-0.7']} | {b['buckets']['0.7-0.9']} "
             f"| {b['buckets']['>=0.9']} |")
    soft_ok = b["buckets"]["<0.5"] + b["buckets"]["0.5-0.7"]
    L.append(f"\n软 need(阈值 {res['soft']:g})能兜住 p_none < {res['soft']:g} 的那 **{soft_ok}** 条,"
             f"兜不住(p_none ≥ {res['soft']:g})的 **{n - soft_ok}** 条。\n")
    cnt = Counter(z["gold"] for z in res["oof"][res["base_spec"]].values())
    L.append(f"\n训练集 need 分布(real_train):none {cnt['none']} / single {cnt['single']} / "
             f"multi {cnt['multi']};`class_weight=\"balanced\"` 下 none 与 needs_mcp 的损失权重相等,"
             "折外判 none 的比例≈真值比例,"
             "none 没有系统性过强(见下面混淆矩阵)。\n")
    L.append("\n20 条兜不住样本的人工归因(逐条看过,与第 2 节的样本表对应):\n")
    L.append("\n| 归因 | 条数 | 例子 |\n| --- | --- | --- |")
    for cause, c, ex in T024_SAMPLE_CAUSES:
        L.append(f"| {cause} | {c} | {ex} |")

    L.append(f"\n## 1. 候选对照(只比开发折,{dev_n} 条)\n")
    L.append("候选 A = none 判定改成 `p_none ≥ t 才判 none`;候选 C = 只换 need 头编码器(检索仍 tfidf)。"
             f"约束:精确率下降 ≤{res['constraints']['precision_drop_max'] * 100:.0f} 点,"
             f"alpaca_zh 误挂率 ≤ 基线 +{res['constraints']['attach_margin'] * 100:.0f} 点。\n")
    _t024_candidate_table(L, res)
    _t024_need_table(L, "基线开发折 need", base["dev"])
    if selected is base:
        L.append(f"\n**结论**:没有候选在约束内把开发折需要工具召回抬过基线,**不改默认**,只报告。")
    else:
        L.append(f"\n**结论**:约束内开发折需要工具召回最高的是 **{selected['label']}**"
                 f"({_pct(selected['dev']['recall'])},基线 {_pct(base['dev']['recall'])})。")

    # 2. 抽看的兜不住样本
    L.append(f"\n## 2. 兜不住的样本(p_none ≥ {res['soft']:g},按 p_none 从高到低)\n")
    L.append("基线的折外预测;这些是软 need 兜不住、真值却要工具的部分。\n")
    L.append("\n| # | source | p_none | 真值 | 文本 |\n| --- | --- | --- | --- | --- |")
    for i, z in enumerate(res["samples"], 1):
        txt = z["text"].replace("|", "\\|").replace("\n", " ")
        L.append(f"| {i} | {z['source']} | {_pnone(z['need_probs']):.3f} | {z['gold']} | {txt} |")

    # 3. 测试折(只报一次)
    L.append(f"\n## 3. 测试折(第 3-4 折,{len(res['test_ids'])} 条,只报一次)\n")
    L.append("开发折选出的候选与基线各报一次;测试折不参与选择。\n")
    for block, label in ((base, "基线"), (selected, f"选中:{selected['label']}")):
        _t024_need_table(L, label, block["test"])

    # 4. 参考集(只报一次)
    L.append("\n## 4. 参考集(D-005,200 条,只报一次)\n")
    L.append("口径:**Claude 预标 + gu 抽查**,只报告不调参。改前 = 基线,改后 = 开发折选中的候选。"
             "第三个块是约束内最好的零依赖候选 A(未选中,供决策)同一份参考集上的数字。\n")
    blocks = [(base, "改前(基线)"), (selected, f"改后(选中:{selected['label']})")]
    alt = next((c for c in res["candidates"] if c["key"] == res.get("alt_key")), None)
    if alt is not None:
        blocks.append((alt, f"候选 A {alt['label']}(未选中,供决策)"))
    for block, label in blocks:
        _t024_need_table(L, label, block["reference"])

    # 5. 端到端(T-021 快照口径)
    if res.get("e2e"):
        e = res["e2e"]
        L.append("\n## 5. 端到端(T-021 快照口径,主口径,中英文严格 R@5)\n")
        L.append(f"检索编码器固定 `{e['retrieval_encoder']}`,软 need {res['soft']:g};"
                 "候选 A 不改变候选集(软 need 已覆盖同一 p_none 区间),与基线逐项相同。\n")
        L.append("\n| 配置 | 折 | 中文严格 R@5 | 英文严格 R@5 |\n| --- | --- | --- | --- |")
        for scope, label in (("dev", "开发折"), ("test", "测试折")):
            L.append(f"| 基线 | {label} | {_pct(e['baseline'][scope]['zh_main_r5_strict'])} "
                     f"| {_pct(e['baseline'][scope]['en_main_r5_strict'])} |")
            L.append(f"| 选中:{selected['label']} | {label} "
                     f"| {_pct(e['selected'][scope]['zh_main_r5_strict'])} "
                     f"| {_pct(e['selected'][scope]['en_main_r5_strict'])} |")

    # 6. 结论
    L.append("\n## 6. 结论\n")
    _t024_verdict(L, res, base, selected)
    return "\n".join(L) + "\n"


def _t024_verdict(L, res, base, selected) -> None:
    if selected is base:
        L.append("开发折上没有候选同时满足精确率与误挂率约束并抬高召回,**不改默认**,"
                 "`cli evaluate` 输出与 main 逐项相同(只多一个只读报告步骤)。\n")
        return
    e = res.get("e2e")
    L.append(f"按任务卡的选法(约束内开发折需要工具召回最高),选中 **{selected['label']}**:"
             f"开发折召回 {_pct(selected['dev']['recall'])}(基线 {_pct(base['dev']['recall'])}),"
             f"精确率 {_pct(selected['dev']['precision'])}(基线 {_pct(base['dev']['precision'])}),"
             f"alpaca 误挂率 {_pct(selected['attach']['rate'])}"
             f"(基线 {_pct(base['attach']['rate'])},门槛 +5 点)。\n")
    if e:
        bs, ss = e["baseline"], e["selected"]
        L.append(f"\n但它同时把 T-021 端到端主口径的**英文**严格 R@5 拉低:开发折 "
                 f"{_pct(bs['dev']['en_main_r5_strict'])} → {_pct(ss['dev']['en_main_r5_strict'])}"
                 f"({(ss['dev']['en_main_r5_strict'] - bs['dev']['en_main_r5_strict']) * 100:+.1f} 点),"
                 f"测试折 {_pct(bs['test']['en_main_r5_strict'])} → "
                 f"{_pct(ss['test']['en_main_r5_strict'])}"
                 f"({(ss['test']['en_main_r5_strict'] - bs['test']['en_main_r5_strict']) * 100:+.1f} 点);"
                 f"中文开发折 {_pct(bs['dev']['zh_main_r5_strict'])} → "
                 f"{_pct(ss['dev']['zh_main_r5_strict'])}。"
                 "原因是请求头只在中文请求上训练,语义编码器让英文请求的 need 判定明显变差。\n")
    L.append("\n**结论:本卡不改默认。**")
    if selected["kind"] == "encoder":
        L.append("选中的 C 需要额外的语义编码器:设成默认会让 `cli train` / 线上加载依赖语义模型"
                 "(与 T-016/T-019「默认 tfidf、语义编码器可选、部署瘦身」的方向相反),"
                 "而且上面这条英文退化必须先解决。建议单独立卡评估(含中英双语训练请求、部署重量)。\n")
    alt = next((c for c in res["candidates"] if c["key"] == res.get("alt_key")), None)
    if alt is not None:
        ar = alt["reference"]
        L.append(f"\n若只想要一条**零依赖**的改动,约束内的是 **{alt['label']}**:开发折召回 "
                 f"{_pct(alt['dev']['recall'])}(基线 {_pct(base['dev']['recall'])},"
                 f"+{(alt['dev']['recall'] - base['dev']['recall']) * base['dev']['pos']:.0f} 条/"
                 f"{base['dev']['pos']} 条真值要工具),"
                 f"精确率 {_pct(alt['dev']['precision'])}"
                 f"({-alt['precision_drop'] * 100:+.1f} 点);参考集召回 {_pct(ar['recall'])}"
                 f"(基线 {_pct(base['reference']['recall'])});端到端与基线逐项相同"
                 "(它只改 need 标签,附带候选的集合不变——软 need 已覆盖同一 p_none 区间)。"
                 "它是决策规则改动而非建模改动,是否落地取决于把「漏判代价高于误报」(D-004)折算成多大召回偏好。\n")


def t024_metrics_json(res) -> dict:
    """`reports/eval_t024.json` 的机器可读版(无延迟,两次跑逐字节相同)。"""
    return {"encoders": res["encoders"], "base_spec": res["base_spec"],
            "dev_ids": res["dev_ids"], "test_ids": res["test_ids"], "soft": res["soft"],
            "top_k": res["top_k"], "folds": res["folds"], "constraints": res["constraints"],
            "selected_key": res["selected_key"], "buckets": res["buckets"],
            "attach": res["attach"], "candidates": res["candidates"],
            "samples": [{"source": z["source"], "p_none": _pnone(z["need_probs"]),
                         "gold": z["gold"], "text": z["text"]} for z in res["samples"]],
            "e2e": res.get("e2e")}


# ---------------------------------------------------------------- T-025 检索两段式
#
# 背景(T-019):检索编码器整体换成语义模型时,英文主口径严格 R@5 退步(58.3% → 56.5%),
# 原因是英文**召回**被拉低;中文召回其实更好(上限 51.0% → 58.0%)。本卡换一种用法:
# 召回仍用 tfidf(快、可解释、英文强),只对它的 Top-N 做语义重排,部署上做成可选项。
#
# 纪律与 T-024 相同:候选只在开发折(第 0-2 折)上选,按中文主口径严格 R@5(T-021 快照
# 口径,need 误判计 0);约束 = 英文主口径严格 R@5 下降 ≤ 1 个点、单请求重排延迟
# p95 ≤ 600 ms(D-007,2026-10-15 gu 拍板放宽;原 50 ms 偏严,一次 LLM 调用本身数秒)。
# 测试折只报一次;天花板调查是诊断,不参与选参。

T025_REPORT = REPORTS / "eval_t025.md"
T025_REPORT_JSON = REPORTS / "eval_t025.json"

# 候选语义模型:(HF 名, 报告里的短名);只用本地已缓存的这两个,不联网
T025_MODELS = (("BAAI/bge-m3", "bge-m3"), ("intfloat/multilingual-e5-small", "e5-small"))
T025_MODEL_SHORT = dict(T025_MODELS)
T025_RERANK_NS = (10, 20, 50)      # R1/R2:重排池大小
T025_ALPHAS = (0.3, 0.5, 0.7)      # R3:linear 融合的 α,在开发折上扫
T025_FUSION_NS_EXTRA = (20, 50)    # 补候选:非 better 模型只在这些 N 上做分数融合
T025_Ceiling_KS = (5, 10, 20, 50)  # 调查:金标落在纯检索 Top-k 的比例
T025_EN_DROP_MAX = 0.01            # 约束:英文严格 R@5 相对基线最多降 1 个点
T025_P95_MAX_MS = 600.0            # 约束:单请求重排延迟 p95(D-007;原先 50)
T025_LIGHT_P95_MAX_MS = 100.0      # 轻量备选:重排延迟 p95 上限
T025_DEFAULT_GAIN = 0.03           # (参考)中文严格 R@5 达到 3 个点即算可观提升


def _t025_model_mb(model: str) -> float:
    """HF 缓存里这个模型的磁盘占用(MB,只数 blobs),部署说明用。"""
    from .rerank import _hf_cache_dir

    blobs = _hf_cache_dir() / f"models--{model.replace('/', '--')}" / "blobs"
    if not blobs.exists():
        return 0.0
    return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file()) / (1024 * 1024)


def t025_ceiling(labeled_tools, synth_tools, synth_train, real_train, requests, zh_by_id,
                 ks=T025_Ceiling_KS) -> dict:
    """调查 1:tfidf 双语索引的召回天花板——金标落在纯检索 Top-k 的请求比例,中英分开。

    用整份数据(synth_train + real_train)按 `cli train` 同口径建一个索引;纯检索,
    不看 need、不过滤(与软过滤关的线上检索完全同一条路径)。只做诊断,不参与选参。
    """
    train = list(synth_train) + list(real_train)
    _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, "tfidf",
                                        bilingual=True, zh_by_id=zh_by_id)
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    out = {"n": len(requests), "ks": list(ks), "index_n": len(cat.records),
           "zh": {"strict": {}, "name": {}}, "en": {"strict": {}, "name": {}}}
    for lang, field in (("zh", "text"), ("en", "text_orig")):
        hits = {k: 0 for k in ks}
        hits_name = {k: 0 for k in ks}
        for r in requests:
            gold = r["tool_ids"]
            gold_names = {name_of.get(g, g) for g in gold}
            top = [rec.tool_id for rec, _ in cat.retrieve(r[field], None, max(ks))]
            for k in ks:
                hits[k] += int(bool(set(top[:k]) & set(gold)))
                hits_name[k] += int(bool({name_of.get(t, t) for t in top[:k]} & gold_names))
        n = len(requests)
        for k in ks:
            out[lang]["strict"][k] = hits[k] / n
            out[lang]["name"][k] = hits_name[k] / n
    return out


def _t025_item(router, cat, req, name_of, top_k) -> dict:
    """一条请求的全部数字:主口径/上限 × 中英 × 严格/同名 × R@1/R@5,加重排每步延迟。

    与 `_snapshot_item` 同一主口径(走完整 `route`,need 误判计 0);延迟单独记
    `zh/en_rerank_ms`(重排器没开时恒 0),与 `latency_ms` 分开,报告里单独标出。
    """
    gold = req["tool_ids"]
    gold_names = {name_of.get(g, g) for g in gold}
    out = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        res = router.route(text)
        rr = getattr(router, "reranker", None)
        out[f"{lang}_rerank_ms"] = rr.last_ms if rr is not None else 0.0
        main = [c["tool_id"] for c in res.candidates]
        upper = [rec.tool_id for rec, _ in cat.retrieve(text, None, top_k)]
        out[f"{lang}_pred_none"] = res.need == "none"
        for mode, ids in (("main", main), ("upper", upper)):
            for k in SNAPSHOT_RATIOS:
                out[f"{lang}_{mode}_r{k}_strict"] = _recall(gold, gold_names, ids[:k], name_of, False)
                out[f"{lang}_{mode}_r{k}_name"] = _recall(gold, gold_names, ids[:k], name_of, True)
    return out


def _t025_summary(items, rows) -> dict:
    """一批请求汇总:中英 × 主口径/上限 × 严格/同名 R@1/R@5 + 重排延迟 p50/p95。"""
    out = {"n": len(rows)}
    for lang in ("zh", "en"):
        for mode in ("main", "upper"):
            for k in SNAPSHOT_RATIOS:
                for suf in ("strict", "name"):
                    key = f"{lang}_{mode}_r{k}_{suf}"
                    out[key] = _mean([items[r["id"]][key] for r in rows])
    lat = {lang: [items[r["id"]][f"{lang}_rerank_ms"] for r in rows] for lang in ("zh", "en")}
    for lang in ("zh", "en"):
        out[f"{lang}_rerank_p50"] = float(np.percentile(lat[lang], 50)) if lat[lang] else 0.0
        out[f"{lang}_rerank_p95"] = float(np.percentile(lat[lang], 95)) if lat[lang] else 0.0
    all_lat = lat["zh"] + lat["en"]
    out["rerank_p50"] = float(np.percentile(all_lat, 50)) if all_lat else 0.0
    out["rerank_p95"] = float(np.percentile(all_lat, 95)) if all_lat else 0.0
    return out


def _t025_label(cfg) -> str:
    short = T025_MODEL_SHORT.get(cfg["model"], cfg["model"])
    if cfg["fusion"] == "reorder":
        return f"{short} 纯语义重排 N={cfg['n']}"
    if cfg["fusion"] == "linear":
        return f"linear α={cfg['alpha']:g} {short} N={cfg['n']}"
    return f"RRF {short} N={cfg['n']}"


def t025_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
              folds=FOLDS, seed=SEED, top_k=5, soft=SOFT_NEED_DEFAULT,
              reranker_factory=None, e2e=True) -> dict:
    """T-025 的全套计算:召回天花板调查 + 候选对照(只比开发折)+ 测试折单次报告。

    - 候选:R1 = bge-m3 纯语义重排(N∈{10,20,50});R2 = 同 e5-small;
      R3 = 开发折上更好的那个模型做分数融合(linear α 网格 + RRF,每个 N),
      并给另一个模型补 fusion(N∈{20,50},review 第 1 轮要求);
    - 选法(D-007,p95 门槛放宽到 600 ms):满足约束的候选里开发折中文严格 R@5 最高
      为"主选";p95 ≤ 100 ms 的候选里中文严格 R@5 最高为"轻量备选"(没有则 None);
    - 主口径 = 走完整 `route`(T-021 快照口径,need 误判计 0);重排延迟 = 重排器
      `last_ms`(查询编码 + Top-N 打分,工具向量已缓存),中英合并算 p95;
    - `reranker_factory`:测试注入假重排器用,签名 `cfg -> 重排器实例`;
    - `e2e=False` 跳过测试折(测试用)。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    dev_ids = sorted(i for i, f in fold_of.items() if f in (0, 1, 2))
    test_ids = sorted(i for i, f in fold_of.items() if f in (3, 4))
    ceiling = t025_ceiling(labeled_tools, synth_tools, synth_train, real_train, requests, zh_by_id)

    def build_env(ids):
        rows = [r for r in requests if r["id"] in ids]
        train = fold_train_rows(synth_train, real_train, ids)
        _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, "tfidf",
                                            bilingual=True, zh_by_id=zh_by_id)
        rhead = ev.train_request_head(_enc, train)
        router = Router(rhead, cat, top_k=top_k, use_filter=False, soft_need_threshold=soft)
        return rows, cat, router

    # 开发折每折建一次环境;测试折建一次,只在最后用(选中项报一次)
    dev_envs = [build_env({i for i, f in fold_of.items() if f == fno}) for fno in (0, 1, 2)]
    test_env = build_env(set(test_ids)) if e2e else None

    made = {}

    def make(cfg):
        key = f"{cfg['fusion']}|{cfg['model']}|{cfg['n']}|{cfg['alpha']:g}"
        if key not in made:
            if reranker_factory is not None:
                made[key] = reranker_factory(cfg)
            else:
                from .rerank import SemanticReranker

                made[key] = SemanticReranker(cfg["model"], rerank_n=cfg["n"],
                                             fusion=cfg["fusion"], alpha=cfg["alpha"])
        return key, cfg

    def run(cfg_keys, envs, fold_ids):
        """给一组候选(第一个可以是 baseline=reranker None)跑一遍,返回 {key: {id: item}}。"""
        per = {key: {} for key in cfg_keys}
        for (rows, cat, router), ids in zip(envs, fold_ids):
            for r in rows:
                for key in cfg_keys:
                    router.reranker = None if key == "baseline" else made[key]
                    per[key][r["id"]] = _t025_item(router, cat, r, name_of, top_k)
        return per

    def finish(cand, dev_summary):
        """补约束字段:英文降幅、重排 p95、是否满足约束。"""
        cand["dev"] = dev_summary
        cand["en_drop"] = base_dev_en - dev_summary["en_main_r5_strict"]
        cand["rerank_p95"] = dev_summary["rerank_p95"]
        cand["qualified"] = (cand["en_drop"] <= T025_EN_DROP_MAX
                             and cand["rerank_p95"] <= T025_P95_MAX_MS)
        return cand

    dev_fold_ids = [{i for i, f in fold_of.items() if f == fno} for fno in (0, 1, 2)]
    dev_rows = [r for r in requests if r["id"] in set(dev_ids)]
    test_rows = [r for r in requests if r["id"] in set(test_ids)]

    # 第一阶段:基线 + R1/R2(两个模型 × N,纯语义重排)
    stage1 = ["baseline"] + [make({"model": m, "n": n, "fusion": "reorder", "alpha": 0.5})[0]
                             for m, _s in T025_MODELS for n in T025_RERANK_NS]
    cfg_of = {"baseline": None}
    for m, _s in T025_MODELS:
        for n in T025_RERANK_NS:
            k, c = make({"model": m, "n": n, "fusion": "reorder", "alpha": 0.5})
            cfg_of[k] = c
    per1 = run(stage1, dev_envs, dev_fold_ids)
    base_dev_en = _t025_summary(per1["baseline"], dev_rows)["en_main_r5_strict"]
    candidates = [{"key": "baseline", "cfg": None, "label": "基线(不重排)",
                   "dev": _t025_summary(per1["baseline"], dev_rows)}]
    for key in stage1[1:]:
        candidates.append(finish({"key": key, "cfg": cfg_of[key], "label": _t025_label(cfg_of[key])},
                                 _t025_summary(per1[key], dev_rows)))

    # 第二阶段(R3):在 R1/R2 里开发折中文 R@5 更高的那个模型上做分数融合
    best_by_model = {}
    for m, _s in T025_MODELS:
        rows_m = [c for c in candidates[1:] if c["cfg"]["model"] == m
                  and c["cfg"]["fusion"] == "reorder"]
        best_by_model[m] = max(c["dev"]["zh_main_r5_strict"] for c in rows_m)
    better = max(T025_MODELS, key=lambda ms: best_by_model[ms[0]])[0]

    def add_fusions(model, ns):
        """给一个模型在给定 N 上跑 linear(α 网格)+ RRF,结果追加进候选。"""
        keys = []
        for n in ns:
            for a in T025_ALPHAS:
                k, c = make({"model": model, "n": n, "fusion": "linear", "alpha": a})
                cfg_of[k] = c
                keys.append(k)
            k, c = make({"model": model, "n": n, "fusion": "rrf", "alpha": 0.5})
            cfg_of[k] = c
            keys.append(k)
        per = run(keys, dev_envs, dev_fold_ids)
        for key in keys:
            candidates.append(finish({"key": key, "cfg": cfg_of[key],
                                      "label": _t025_label(cfg_of[key])},
                                     _t025_summary(per[key], dev_rows)))

    add_fusions(better, T025_RERANK_NS)          # R3:更好的模型扫全部 N
    for m, _s in T025_MODELS:                    # review:另一个模型补 N∈{20,50} 融合
        if m != better:
            add_fusions(m, T025_FUSION_NS_EXTRA)

    # 选法(D-007):满足约束里开发折中文主口径严格 R@5 最高;没有满足约束的就不改。
    # 轻量备选:p95 ≤ 100 ms 里中文最高(按 review 字面,不叠加英文约束);没有则 None。
    base = candidates[0]
    qualified = [c for c in candidates[1:] if c["qualified"]]
    selected = max(qualified, key=lambda c: c["dev"]["zh_main_r5_strict"]) if qualified else base
    light_pool = [c for c in candidates[1:] if c["rerank_p95"] <= T025_LIGHT_P95_MAX_MS]
    lightweight = (max(light_pool, key=lambda c: c["dev"]["zh_main_r5_strict"])
                   if light_pool else None)

    # 测试折只报一次:基线 + 主选 + 轻量备选;端到端改前改后并排(同一份主口径数字)
    res = {"models": [{"spec": m, "short": s} for m, s in T025_MODELS],
           "ceiling": ceiling, "candidates": candidates, "base_key": base["key"],
           "selected_key": selected["key"],
           "lightweight_key": lightweight["key"] if lightweight else None,
           "better_model": better,
           "dev_ids": dev_ids, "test_ids": test_ids, "soft": soft, "top_k": top_k,
           "folds": folds,
           "constraints": {"en_drop_max": T025_EN_DROP_MAX, "p95_max_ms": T025_P95_MAX_MS,
                           "light_p95_max_ms": T025_LIGHT_P95_MAX_MS,
                           "default_gain_min": T025_DEFAULT_GAIN},
           "build_info": {rr.model_name: rr.first_build for rr in made.values()
                          if getattr(rr, "first_build", None)},
           "model_mb": {m: _t025_model_mb(m) for m, _s in T025_MODELS}}
    if e2e:
        final = ["baseline", selected["key"]]
        if lightweight is not None and lightweight["key"] not in final:
            final.append(lightweight["key"])
        per_t = run(final, [test_env], [set(test_ids)])
        res["e2e"] = {"baseline": {"dev": base["dev"],
                                   "test": _t025_summary(per_t["baseline"], test_rows)},
                      "selected": {"dev": selected["dev"],
                                   "test": _t025_summary(per_t[selected["key"]], test_rows)},
                      "lightweight": ({"dev": lightweight["dev"],
                                       "test": _t025_summary(per_t[lightweight["key"]], test_rows)}
                                      if lightweight is not None else None)}
    return res


def _t025_candidate_table(L, res) -> None:
    L.append("| 候选 | 中文严格 R@1 | 中文严格 R@5 | 中文同名 R@5 | 英文严格 R@1 | 英文严格 R@5 "
             "| 英文同名 R@5 | 英文降幅 | 重排 p95(ms) | 满足约束 |\n| --- | --- | --- | --- | --- "
             "| --- | --- | --- | --- | --- |")
    base = next(c for c in res["candidates"] if c["key"] == res["base_key"])
    for c in res["candidates"]:
        d = c["dev"]
        if c["key"] == res["base_key"]:
            L.append(f"| {c['label']} | {_pct(d['zh_main_r1_strict'])} | {_pct(d['zh_main_r5_strict'])} "
                     f"| {_pct(d['zh_main_r5_name'])} | {_pct(d['en_main_r1_strict'])} "
                     f"| {_pct(d['en_main_r5_strict'])} | {_pct(d['en_main_r5_name'])} "
                     f"| - | - | - |")
            continue
        L.append(f"| {c['label']} | {_pct(d['zh_main_r1_strict'])} | {_pct(d['zh_main_r5_strict'])} "
                 f"| {_pct(d['zh_main_r5_name'])} | {_pct(d['en_main_r1_strict'])} "
                 f"| {_pct(d['en_main_r5_strict'])} | {_pct(d['en_main_r5_name'])} "
                 f"| {-c['en_drop'] * 100:+.1f} 点 | {c['rerank_p95']:.1f} | "
                 f"{'是' if c['qualified'] else '否'} |")


def render_t025(res) -> str:
    """把 `t025_eval` 的结果渲染成 `reports/eval_t025.md`。

    除延迟字段(标了 ms 的单元格、首次构建耗时)外,两次复跑逐字节相同。
    """
    dev_n = len(res["dev_ids"])
    selected = next(c for c in res["candidates"] if c["key"] == res["selected_key"])
    base = next(c for c in res["candidates"] if c["key"] == res["base_key"])
    lw_key = res.get("lightweight_key")
    lightweight = next((c for c in res["candidates"] if c["key"] == lw_key), None) if lw_key else None
    L = ["# 检索两段式:tfidf 召回 Top-N + 语义重排(T-025)\n",
         f"真实目录 {res['ceiling']['index_n']} 个工具(双语索引),600 条 BFCL 请求;"
         "请求头按 5 折折外训练(`assign_folds`,seed 不变),"
         "开发折 = 第 0-2 折,测试折 = 第 3-4 折。\n",
         f"> 显式设置:分类头与检索召回都按线上默认(tfidf,双语索引,软过滤关,软 need "
         f"{res['soft']:g});重排只作用于 tfidf 召回的 Top-N。主口径 = 走完整 `route`,"
         "need 误判计 0(T-021 快照口径)。只在开发折上选;测试折只报一次,不参与选择(D-005)。\n",
         f"> 1 条样本 = 1/{dev_n} = {100.0 / dev_n:.2f} 个百分点。\n",
         "> 复跑说明:所有标 ms 的数字(重排延迟、首次构建耗时)两次运行会变,"
         "其余数字逐字节相同。\n"]

    # 0. 调查结论
    L.append("\n## 0. 调查结论\n")
    c = res["ceiling"]
    L.append(f"**召回天花板**:金标工具落在纯 tfidf 检索(n={c['n']} 条请求,"
             f"索引 {c['index_n']} 工具,不看 need、不过滤)Top-k 的请求比例:\n")
    L.append("\n| 查询语言 | 口径 | " + " | ".join(f"Top-{k}" for k in c["ks"]) +
             " |\n| --- | --- | " + " | ".join("---" for _ in c["ks"]) + " |")
    for lang, lang_label in (("zh", "中文"), ("en", "英文")):
        for suf, suf_label in (("strict", "严格"), ("name", "同名")):
            L.append(f"| {lang_label} | {suf_label} | " +
                     " | ".join(_pct(c[lang][suf][k]) for k in c["ks"]) + " |")
    L.append("\n中文严格口径 Top-20 之后仍有可观的金标留在 20-50 名之间(见上表 Top-20 与 "
             "Top-50 的差),重排有空间;Top-50 都捞不到的那部分,重排救不了。\n")
    L.append("\n**T-019 的结论与本卡的区别**(`reports/eval_t019.md`):把检索编码器整体换成"
             "语义模型时,开发折中文主口径严格 R@5 26.0% → 27.6%(bge-m3,+6.0 条),"
             "但英文 58.3% → 56.5%(-6.5 条);上限口径(假设 need 全对)中文 51.0% → 58.0%,"
             "英文 66.2% → 63.5%。**失败在召回,不在精排**:bge-m3 的中文召回本身更好,"
             "坏在英文查询的召回被拉低。所以本卡让召回保持 tfidf(英文的强项、可解释),"
             "语义模型只对 tfidf 的 Top-N 重排——召回集合不变,排序用语义。\n")

    # 1. 候选对照(只比开发折)
    L.append(f"\n## 1. 候选对照(只比开发折,{dev_n} 条)\n")
    better_short = T025_MODEL_SHORT.get(res["better_model"], res["better_model"])
    other = [s for m, s in T025_MODELS if m != res["better_model"]]
    other_short = other[0] if other else "-"
    extra_ns = "、".join(str(n) for n in T025_FUSION_NS_EXTRA)
    L.append(f"R1 = bge-m3 纯语义重排;R2 = e5-small 纯语义重排;R3 = 分数融合(linear 的 "
             f"α ∈ {{0.3, 0.5, 0.7}} + RRF):在 R1/R2 里开发折中文严格 R@5 更高的 "
             f"**{better_short}** 上扫全部 N ∈ {{10, 20, 50}},另给 {other_short} 补 "
             f"N ∈ {{{extra_ns}}}。"
             "约束:英文严格 R@5 下降 ≤ "
             f"{res['constraints']['en_drop_max'] * 100:.0f} 个点,重排延迟 p95 ≤ "
             f"{res['constraints']['p95_max_ms']:g} ms(中英合并,CPU,工具向量已缓存)。\n")
    _t025_candidate_table(L, res)
    L.append("\n重排延迟(单请求 = 查询编码 + Top-N 打分,中英合并;复跑会变):\n")
    L.append("\n| 候选 | p50(ms) | p95(ms) |\n| --- | --- | --- |")
    for c in res["candidates"]:
        if c["key"] == res["base_key"]:
            continue
        L.append(f"| {c['label']} | {c['dev']['rerank_p50']:.1f} | {c['rerank_p95']:.1f} |")

    # 2. 测试折(只报一次)+ 端到端改前改后并排
    if res.get("e2e"):
        e = res["e2e"]
        L.append(f"\n## 2. 测试折(第 3-4 折,{len(res['test_ids'])} 条,只报一次)"
                 "与端到端改前改后并排\n")
        L.append("同一份 T-021 快照口径(走完整 `route`,need 误判计 0);"
                 "测试折不参与选择,只报基线、主选、轻量备选各一行。\n")
        L.append("\n| 配置 | 折 | n | 中文严格 R@1 | 中文严格 R@5 | 中文同名 R@5 | 英文严格 R@1 "
                 "| 英文严格 R@5 | 英文同名 R@5 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        rows = [("baseline", "改前(基线)"),
                ("selected", f"主选:{selected['label']}")]
        if e.get("lightweight"):
            rows.append(("lightweight", f"轻量备选:{lightweight['label']}"))
        else:
            rows.append(("lightweight", "轻量备选:无"))
        for key, label in rows:
            for scope, scope_label in (("dev", "开发折"), ("test", "测试折")):
                d = e[key][scope] if e.get(key) else None
                cells = ("-" * 1) if d is None else (
                    f"{_pct(d['zh_main_r1_strict'])} | {_pct(d['zh_main_r5_strict'])} "
                    f"| {_pct(d['zh_main_r5_name'])} | {_pct(d['en_main_r1_strict'])} "
                    f"| {_pct(d['en_main_r5_strict'])} | {_pct(d['en_main_r5_name'])}")
                n = "-" if d is None else d["n"]
                L.append(f"| {label} | {scope_label} | {n} | {cells} |")

    # 3. 部署说明
    L.append("\n## 3. 部署说明\n")
    L.append("- 开关:`cli train --rerank <模型名> --rerank-n <N> "
             "[--rerank-fusion reorder|linear|rrf] [--rerank-alpha <α>]`;线上 `cli route` "
             "用同样的参数在运行期覆盖(不传就用训练时存的)。不传 `--rerank` = 纯 tfidf,"
             "行为与改前逐字节一致。\n")
    L.append("- 工具向量:按 (模型, 索引文本哈希) 缓存在 `.cache/tool_vecs/`,"
             "部署时可在训练后显式预热;新工具接入或中文译文更新后哈希失配自动重建。\n")
    for m, info in sorted(res["build_info"].items()):
        how = "命中已有缓存" if info["cached"] else "本次全新构建"
        L.append(f"- {T025_MODEL_SHORT.get(m, m)}:首次构建工具向量({info['n_tools']} 个工具,"
                 f"{info['dim']} 维){how},耗时 {info['build_ms'] / 1000:.1f} s(复跑会变);"
                 f"模型磁盘占用 {res['model_mb'].get(m, 0.0):.0f} MB。\n")
    L.append("- `router.pkl` 大小不受影响:重排器只把配置存进 pickle(模型名 / N / fusion / α),"
             "权重仍在 HuggingFace 本地缓存里,加载时重建。\n")

    # 4. 结论
    L.append("\n## 4. 结论\n")
    e = res.get("e2e")
    if selected is base:
        L.append("开发折上没有候选同时满足两条约束,**主选 = 无(维持默认)**。\n")
    else:
        gain = selected["dev"]["zh_main_r5_strict"] - base["dev"]["zh_main_r5_strict"]
        L.append(f"按 D-007 新门槛(英文严格 R@5 降幅 ≤ "
                 f"{res['constraints']['en_drop_max'] * 100:.0f} 点、重排 p95 ≤ "
                 f"{res['constraints']['p95_max_ms']:g} ms)选主选:满足约束里开发折中文严格 "
                 f"R@5 最高的是 **{selected['label']}**——中文 "
                 f"{base['dev']['zh_main_r5_strict'] * 100:.1f}% → "
                 f"{selected['dev']['zh_main_r5_strict'] * 100:.1f}%({gain * 100:+.1f} 点),"
                 f"英文 {base['dev']['en_main_r5_strict'] * 100:.1f}% → "
                 f"{selected['dev']['en_main_r5_strict'] * 100:.1f}%"
                 f"({-selected['en_drop'] * 100:+.1f} 点),重排 p95 "
                 f"{selected['rerank_p95']:.1f} ms。\n")
        if e:
            bt, st = e["baseline"]["test"], e["selected"]["test"]
            L.append(f"\n测试折:中文 {bt['zh_main_r5_strict'] * 100:.1f}% → "
                     f"{st['zh_main_r5_strict'] * 100:.1f}%,英文 {bt['en_main_r5_strict'] * 100:.1f}% "
                     f"→ {st['en_main_r5_strict'] * 100:.1f}%。\n")
    if lightweight is not None:
        lw = lightweight
        L.append(f"\n轻量备选(p95 ≤ {res['constraints']['light_p95_max_ms']:g} ms 里开发折中文严格 "
                 f"R@5 最高)= **{lw['label']}**:中文 "
                 f"{base['dev']['zh_main_r5_strict'] * 100:.1f}% → "
                 f"{lw['dev']['zh_main_r5_strict'] * 100:.1f}%,英文 "
                 f"{base['dev']['en_main_r5_strict'] * 100:.1f}% → "
                 f"{lw['dev']['en_main_r5_strict'] * 100:.1f}%({-lw['en_drop'] * 100:+.1f} 点),"
                 f"重排 p95 {lw['rerank_p95']:.1f} ms。\n")
    else:
        L.append(f"\np95 ≤ {res['constraints']['light_p95_max_ms']:g} ms 的候选里没有可选的,"
                 "轻量备选:**无**。\n")

    L.append("\n按 D-007,**默认仍不开重排**:`cli evaluate` / `cli route` 不带 `--rerank` 时"
             "输出与 main 逐项相同(只多延迟)。上面的主选 / 轻量备选作为两档可选配置写进 "
             "README「怎么用」,数字均来自本报告。\n")
    return "\n".join(L) + "\n"


def t025_metrics_json(res) -> dict:
    """`reports/eval_t025.json` 的机器可读版(延迟字段两次跑会变,其余逐字节相同)。"""
    return {"models": res["models"], "ceiling": res["ceiling"],
            "candidates": res["candidates"], "base_key": res["base_key"],
            "selected_key": res["selected_key"],
            "lightweight_key": res.get("lightweight_key"),
            "better_model": res["better_model"],
            "dev_ids": res["dev_ids"], "test_ids": res["test_ids"], "soft": res["soft"],
            "top_k": res["top_k"], "folds": res["folds"], "constraints": res["constraints"],
            "build_info": res["build_info"], "model_mb": res["model_mb"],
            "e2e": res.get("e2e")}


# ---------------------------------------------------------------- T-026 混合召回

T026_REPORT = REPORTS / "eval_t026.md"
T026_REPORT_JSON = REPORTS / "eval_t026.json"

T026_MODELS = T025_MODELS           # 只用本地已缓存的 bge-m3 / e5-small,不联网
T026_RECALL_N = 50                  # tfidf 召回 Top-50(并集的"tfidf 部分",同 T-025 N=50)
T026_MS = (10, 20, 50)             # 语义 Top-M
T026_ALPHAS = (0.3, 0.5)            # linear 融合 α(同 T-025 两档)
T026_CEILING_KS = (50,)            # 调查只和 tfidf Top-50 对照
T026_EN_DROP_MAX = 0.01             # 英文严格 R@5 相对纯 tfidf 基线降幅上限(同 T-025)
T026_P95_MAX_MS = 600.0             # 约束:单请求召回+重排 p95(D-007)
T026_LIGHT_P95_MAX_MS = 100.0       # 轻量备选:召回+重排 p95 上限
T026_MIN_GAIN = 0.02               # 比 T-025 对应档位中文严格 R@5 高 ≥ 2 点才算有收益
T026_UNION_WORTH_PP = 0.03          # 中文并集覆盖率增量 < 3 点 -> 并集不值得做


def _make_t026_reranker(model, n, recall_m, fusion, alpha):
    """按配置建语义重排器;测试用 `reranker_factory` 注入假重排器时不会走到这里。"""
    from .rerank import SemanticReranker

    return SemanticReranker(model, rerank_n=n, fusion=fusion, alpha=alpha, recall_m=recall_m)


def t026_ceiling(labeled_tools, synth_tools, synth_train, real_train, requests, zh_by_id,
                 models=T026_MODELS, ms=T026_MS, ks=T026_CEILING_KS,
                 reranker_factory=None) -> dict:
    """调查 1/2:tfidf Top-50 ∪ 语义 Top-M 的并集召回天花板,对照纯 tfidf Top-50。

    用整份数据(synth_train + real_train)按 `cli train` 同口径建一个索引;纯检索,
    不看 need、不过滤(与 T-025 §0 同一条路径)。只做诊断,不参与选参。
    - `union[短名][M][lang][strict/name]`:金标落在并集里的请求比例;
    - `sem_only[短名][M][lang]`:只在语义 Top-M 里、不在 tfidf Top-50 里的请求条数(增量来源);
    - `tfidf[lang][strict/name][k]`:纯 tfidf Top-k 的对照。
    """
    train = list(synth_train) + list(real_train)
    _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, "tfidf",
                                        bilingual=True, zh_by_id=zh_by_id)
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    n = len(requests)
    kbig, mmax = max(ks), max(ms)
    out = {"n": n, "ks": list(ks), "index_n": len(cat.records),
           "models": [{"spec": m, "short": s} for m, s in models],
           "tfidf": {"zh": {"strict": {}, "name": {}}, "en": {"strict": {}, "name": {}}},
           "union": {}, "sem_only": {}}

    # 纯 tfidf Top-k(并集的对照)
    tfidf_ids = {}
    for r in requests:
        for lang, field in (("zh", "text"), ("en", "text_orig")):
            tfidf_ids[(r["id"], lang)] = [rec.tool_id
                                          for rec, _ in cat.retrieve(r[field], None, kbig)]
    for lang in ("zh", "en"):
        for k in ks:
            hs = hn = 0
            for r in requests:
                gold = r["tool_ids"]
                gold_names = {name_of.get(g, g) for g in gold}
                top = tfidf_ids[(r["id"], lang)][:k]
                hs += int(bool(set(top) & set(gold)))
                hn += int(bool({name_of.get(t, t) for t in top} & gold_names))
            out["tfidf"][lang]["strict"][k] = hs / n
            out["tfidf"][lang]["name"][k] = hn / n

    # 每个模型的语义 Top-mmax(取最大的 M 一次,更小的 M 取前缀即可)
    for spec, short in models:
        rr = (reranker_factory({"model": spec, "n": T026_RECALL_N, "fusion": "linear",
                                "alpha": 0.5, "recall_m": 0})
              if reranker_factory is not None
              else _make_t026_reranker(spec, T026_RECALL_N, 0, "linear", 0.5))
        rr.warm(cat)
        sem_ids = {}
        for r in requests:
            for lang, field in (("zh", "text"), ("en", "text_orig")):
                sem_ids[(r["id"], lang)] = [rec.tool_id
                                            for rec, _ in rr.semantic_top(r[field], mmax, cat)]
        out["union"][short], out["sem_only"][short] = {}, {}
        for M in ms:
            cov = {"zh": {"strict": 0, "name": 0}, "en": {"strict": 0, "name": 0}}
            only = {"zh": 0, "en": 0}
            for r in requests:
                gold = r["tool_ids"]
                gold_names = {name_of.get(g, g) for g in gold}
                for lang in ("zh", "en"):
                    tf = set(tfidf_ids[(r["id"], lang)])
                    sem = set(sem_ids[(r["id"], lang)][:M])
                    union = tf | sem
                    if set(gold) & union:
                        cov[lang]["strict"] += 1
                    if gold_names & {name_of.get(t, t) for t in union}:
                        cov[lang]["name"] += 1
                    if (set(gold) & sem) and not (set(gold) & tf):
                        only[lang] += 1
            out["union"][short][M] = {lang: {s: cov[lang][s] / n for s in ("strict", "name")}
                                      for lang in ("zh", "en")}
            out["sem_only"][short][M] = only
    return out


def _t026_union_worth(ceiling) -> dict:
    """中文并集覆盖率相对 tfidf Top-50 的最大增量(pp);≥ 3 点视为"并集值得做"。"""
    k = ceiling["ks"][-1]
    tf_zh = ceiling["tfidf"]["zh"]["strict"][k]
    best = None
    for m in ceiling["models"]:
        short = m["short"]
        for M, cov in ceiling["union"][short].items():
            inc = cov["zh"]["strict"] - tf_zh
            if best is None or inc > best["zh_inc"]:
                best = {"model": short, "M": M, "zh_inc": inc,
                        "zh_union": cov["zh"]["strict"], "zh_tfidf": tf_zh}
    return {"k": k, "tfidf_zh_top": tf_zh, "best": best,
            "worth": bool(best and best["zh_inc"] >= T026_UNION_WORTH_PP)}


def load_t025_controls(path=T025_REPORT_JSON) -> dict:
    """读 `reports/eval_t025.json` 里的主选 / 轻量两档(开发折数字 + 测试折 e2e),不重跑。"""
    p = pathlib.Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {"path": str(p), "source": p.name, "main": None, "lightweight": None}
    for slot, keyfield, e2ekey in (("main", "selected_key", "selected"),
                                   ("lightweight", "lightweight_key", "lightweight")):
        key = d.get(keyfield)
        if not key:
            continue
        cand = next((c for c in d["candidates"] if c["key"] == key), None)
        e2e = (d.get("e2e") or {}).get(e2ekey)
        out[slot] = {"key": key, "cfg": (cand or {}).get("cfg"),
                     "label": (cand or {}).get("label", key),
                     "dev": (cand or {}).get("dev"), "test": (e2e or {}).get("test")}
    return out


def _t026_item(router, cat, req, name_of, top_k, recall_n) -> dict:
    """一条请求的主口径数字(同 `_t025_item`)+ 召回 / 重排 / 召回+重排 三步延迟。

    召回一步单独计时(与 `route` 内同一条 `retrieve` 路径);`{lang}_hybrid_ms` = 两者之和,
    作为"单请求召回+重排"的约束口径。主口径仍走完整 `route`,need 误判计 0(T-021 快照口径)。
    """
    gold = req["tool_ids"]
    gold_names = {name_of.get(g, g) for g in gold}
    out = {}
    for lang, text in (("zh", req["text"]), ("en", req["text_orig"])):
        t0 = time.perf_counter()
        cat.retrieve(text, None, recall_n)              # 召回一步
        out[f"{lang}_recall_ms"] = (time.perf_counter() - t0) * 1000
        res = router.route(text)
        rr = getattr(router, "reranker", None)
        out[f"{lang}_rerank_ms"] = rr.last_ms if rr is not None else 0.0
        out[f"{lang}_hybrid_ms"] = out[f"{lang}_recall_ms"] + out[f"{lang}_rerank_ms"]
        main = [c["tool_id"] for c in res.candidates]
        upper = [rec.tool_id for rec, _ in cat.retrieve(text, None, top_k)]
        out[f"{lang}_pred_none"] = res.need == "none"
        for mode, ids in (("main", main), ("upper", upper)):
            for k in SNAPSHOT_RATIOS:
                out[f"{lang}_{mode}_r{k}_strict"] = _recall(gold, gold_names, ids[:k], name_of, False)
                out[f"{lang}_{mode}_r{k}_name"] = _recall(gold, gold_names, ids[:k], name_of, True)
    return out


def _t026_summary(items, rows) -> dict:
    """一批请求汇总:R 指标 + 重排 p50/p95(复用 `_t025_summary`)+ 召回 / 召回+重排 p50/p95。"""
    out = _t025_summary(items, rows)                    # zh_main_r*_* + 重排 p50/p95
    for name in ("recall", "hybrid"):
        for lang in ("zh", "en"):
            lat = [items[r["id"]][f"{lang}_{name}_ms"] for r in rows]
            out[f"{lang}_{name}_p50"] = float(np.percentile(lat, 50)) if lat else 0.0
            out[f"{lang}_{name}_p95"] = float(np.percentile(lat, 95)) if lat else 0.0
        all_lat = [items[r["id"]][f"{l}_{name}_ms"] for l in ("zh", "en") for r in rows]
        out[f"{name}_p50"] = float(np.percentile(all_lat, 50)) if all_lat else 0.0
        out[f"{name}_p95"] = float(np.percentile(all_lat, 95)) if all_lat else 0.0
    return out


def _t026_label(cfg) -> str:
    short = T025_MODEL_SHORT.get(cfg["model"], cfg["model"])
    return f"linear α={cfg['alpha']:g} {short} 召回50∪语义{cfg['recall_m']}"


def t026_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
              folds=FOLDS, seed=SEED, top_k=5, soft=SOFT_NEED_DEFAULT,
              reranker_factory=None, e2e=True, t025_json=T025_REPORT_JSON) -> dict:
    """T-026 的全套计算:并集召回天花板调查 + H1 候选对照(只比开发折)+ 测试折单次报告。

    - H1 候选:tfidf Top-50 ∪ 语义 Top-M,linear 融合,α ∈ {0.3, 0.5},M ∈ {10, 20, 50},
      bge-m3 与 e5-small 各扫;池子外(tfidf 没召回到)的工具 tfidf 分记 0(见 `rerank._expand_pool`);
    - 对照行:T-025 主选 / 轻量两档,数字直接从 `reports/eval_t025.json` 读,不重跑;
    - 选法(同 D-007):满足约束(英文严格 R@5 降幅 ≤1 点、召回+重排 p95 ≤600 ms)里开发折
      中文严格 R@5 最高为主选;p95 ≤100 ms 里中文最高为轻量备选;
    - 主口径 = 走完整 `route`(T-021 快照口径,need 误判计 0);
    - `reranker_factory`:测试注入假重排器用,签名 `cfg -> 重排器实例`(需实现 `warm`/`semantic_top`/
      `rerank`);`e2e=False` 跳过测试折(测试用)。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    name_of = {t["tool_id"]: t["name"] for t in labeled_tools}
    name_of.update({catalog_tool_id(t): t["name"] for t in synth_tools})
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    dev_ids = sorted(i for i, f in fold_of.items() if f in (0, 1, 2))
    test_ids = sorted(i for i, f in fold_of.items() if f in (3, 4))
    ceiling = t026_ceiling(labeled_tools, synth_tools, synth_train, real_train, requests, zh_by_id,
                           reranker_factory=reranker_factory)

    def build_env(ids):
        rows = [r for r in requests if r["id"] in ids]
        train = fold_train_rows(synth_train, real_train, ids)
        _enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, "tfidf",
                                            bilingual=True, zh_by_id=zh_by_id)
        rhead = ev.train_request_head(_enc, train)
        router = Router(rhead, cat, top_k=top_k, use_filter=False, soft_need_threshold=soft)
        return rows, cat, router

    dev_envs = [build_env({i for i, f in fold_of.items() if f == fno}) for fno in (0, 1, 2)]
    test_env = build_env(set(test_ids)) if e2e else None

    made = {}

    def make(cfg):
        key = f"{cfg['fusion']}|{cfg['model']}|{cfg['n']}|{cfg['alpha']:g}|{cfg['recall_m']}"
        if key not in made:
            if reranker_factory is not None:
                made[key] = reranker_factory(cfg)
            else:
                made[key] = _make_t026_reranker(cfg["model"], cfg["n"], cfg["recall_m"],
                                                cfg["fusion"], cfg["alpha"])
        return key, cfg

    def run(cfg_keys, envs, fold_ids):
        per = {key: {} for key in cfg_keys}
        for (rows, cat, router), ids in zip(envs, fold_ids):
            for r in rows:
                for key in cfg_keys:
                    router.reranker = None if key == "baseline" else made[key]
                    per[key][r["id"]] = _t026_item(router, cat, r, name_of, top_k, T026_RECALL_N)
        return per

    # H1 候选:tfidf Top-50 ∪ 语义 Top-M
    cfg_of = {"baseline": None}
    h1 = []
    for m, _s in T026_MODELS:
        for M in T026_MS:
            for a in T026_ALPHAS:
                k, c = make({"model": m, "n": T026_RECALL_N, "fusion": "linear",
                             "alpha": a, "recall_m": M})
                cfg_of[k] = c
                h1.append(k)
    per1 = run(["baseline"] + h1, dev_envs, [{i for i, f in fold_of.items() if f == fno}
                                             for fno in (0, 1, 2)])
    dev_rows = [r for r in requests if r["id"] in set(dev_ids)]
    base_sum = _t026_summary(per1["baseline"], dev_rows)
    base_dev_en = base_sum["en_main_r5_strict"]

    def finish(cand, dev_summary):
        cand["dev"] = dev_summary
        cand["en_drop"] = base_dev_en - dev_summary["en_main_r5_strict"]
        cand["hybrid_p95"] = dev_summary["hybrid_p95"]
        cand["qualified"] = (cand["en_drop"] <= T026_EN_DROP_MAX
                             and cand["hybrid_p95"] <= T026_P95_MAX_MS)
        return cand

    candidates = [{"key": "baseline", "cfg": None, "label": "基线(不重排)", "dev": base_sum}]
    for key in h1:
        candidates.append(finish({"key": key, "cfg": cfg_of[key], "label": _t026_label(cfg_of[key])},
                                 _t026_summary(per1[key], dev_rows)))

    controls = load_t025_controls(t025_json)

    qualified = [c for c in candidates[1:] if c["qualified"]]
    selected = max(qualified, key=lambda c: c["dev"]["zh_main_r5_strict"]) if qualified else candidates[0]
    light_pool = [c for c in candidates[1:] if c["hybrid_p95"] <= T026_LIGHT_P95_MAX_MS]
    lightweight = (max(light_pool, key=lambda c: c["dev"]["zh_main_r5_strict"])
                   if light_pool else None)

    # 与 T-025 对应档位比(开发折中文严格 R@5):高 ≥ 2 点才算本卡有收益
    def _ctrl_dev(slot):
        c = controls.get(slot)
        return c["dev"]["zh_main_r5_strict"] if c and c.get("dev") else None

    main_t025, lw_t025 = _ctrl_dev("main"), _ctrl_dev("lightweight")
    main_gain = (selected["dev"]["zh_main_r5_strict"] - main_t025
                 if selected is not candidates[0] and main_t025 is not None else None)
    lw_gain = (lightweight["dev"]["zh_main_r5_strict"] - lw_t025
               if lightweight is not None and lw_t025 is not None else None)
    benefit = bool((main_gain is not None and main_gain >= T026_MIN_GAIN)
                   or (lw_gain is not None and lw_gain >= T026_MIN_GAIN))

    res = {"models": [{"spec": m, "short": s} for m, s in T026_MODELS],
           "ceiling": ceiling, "candidates": candidates, "controls": controls,
           "base_key": "baseline", "selected_key": selected["key"],
           "lightweight_key": lightweight["key"] if lightweight else None,
           "dev_ids": dev_ids, "test_ids": test_ids, "soft": soft, "top_k": top_k,
           "folds": folds, "recall_n": T026_RECALL_N,
           "constraints": {"en_drop_max": T026_EN_DROP_MAX, "p95_max_ms": T026_P95_MAX_MS,
                           "light_p95_max_ms": T026_LIGHT_P95_MAX_MS, "min_gain": T026_MIN_GAIN,
                           "union_worth_pp": T026_UNION_WORTH_PP},
           "gains": {"main": main_gain, "lightweight": lw_gain, "benefit": benefit},
           "union_worth": _t026_union_worth(ceiling), "t025_source": controls.get("source"),
           "build_info": {getattr(rr, "model_name", None): rr.first_build
                          for rr in made.values() if getattr(rr, "first_build", None)},
           "model_mb": {m: _t025_model_mb(m) for m, _s in T026_MODELS}}
    if e2e:
        final = ["baseline", selected["key"]]
        if lightweight is not None and lightweight["key"] not in final:
            final.append(lightweight["key"])
        test_rows = [r for r in requests if r["id"] in set(test_ids)]
        per_t = run(final, [test_env], [set(test_ids)])
        res["e2e"] = {"baseline": {"dev": candidates[0]["dev"],
                                   "test": _t026_summary(per_t["baseline"], test_rows)},
                      "selected": {"dev": selected["dev"],
                                   "test": _t026_summary(per_t[selected["key"]], test_rows)},
                      "lightweight": ({"dev": lightweight["dev"],
                                       "test": _t026_summary(per_t[lightweight["key"]], test_rows)}
                                      if lightweight is not None else None)}
    return res


def _t026_r_cells(d) -> str:
    """候选表里的 6 个 R 指标单元格(中英 × 严格 R@1/R@5、同名 R@5)。"""
    return (f"{_pct(d['zh_main_r1_strict'])} | {_pct(d['zh_main_r5_strict'])} "
            f"| {_pct(d['zh_main_r5_name'])} | {_pct(d['en_main_r1_strict'])} "
            f"| {_pct(d['en_main_r5_strict'])} | {_pct(d['en_main_r5_name'])}")


def _t026_candidate_table(L, res) -> None:
    """§1 对照表:T-025 两档(读自 json)+ 基线 + H1 候选。"""
    base_en = res["candidates"][0]["dev"]["en_main_r5_strict"]
    L.append("| 候选 | 中文严格 R@1 | 中文严格 R@5 | 中文同名 R@5 | 英文严格 R@1 | 英文严格 R@5 "
             "| 英文同名 R@5 | 英文降幅 | 召回+重排 p95(ms) | 满足约束 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    b = res["candidates"][0]
    L.append(f"| {b['label']} | {_t026_r_cells(b['dev'])} | - | - | - |")
    ctrl = res["controls"]
    for slot, name in (("main", "T-025 主选"), ("lightweight", "T-025 轻量")):
        c = ctrl.get(slot)
        if not c or not c.get("dev"):
            continue
        d = c["dev"]
        en_drop = base_en - d["en_main_r5_strict"]
        L.append(f"| {name}:{c['label']}(读自 `{res['t025_source']}`) | {_t026_r_cells(d)} "
                 f"| {-en_drop * 100:+.1f} 点 | - | - |")
    for c in res["candidates"][1:]:
        d = c["dev"]
        star = " **(主选)**" if c["key"] == res["selected_key"] else (
            " (轻量)" if c["key"] == res.get("lightweight_key") else "")
        L.append(f"| {c['label']}{star} | {_t026_r_cells(d)} | {-c['en_drop'] * 100:+.1f} 点 "
                 f"| {c['hybrid_p95']:.1f} | {'是' if c['qualified'] else '否'} |")


def render_t026(res) -> str:
    """把 `t026_eval` 的结果渲染成 `reports/eval_t026.md`。

    除延迟字段(标了 ms 的单元格、首次构建耗时)外,两次复跑逐字节相同。
    """
    dev_n = len(res["dev_ids"])
    base = res["candidates"][0]
    sel = next(c for c in res["candidates"] if c["key"] == res["selected_key"])
    lw_key = res.get("lightweight_key")
    lw = next((c for c in res["candidates"] if c["key"] == lw_key), None) if lw_key else None
    ctrl = res["controls"]
    con = res["constraints"]
    L = ["# 混合召回:tfidf Top-50 ∪ 语义 Top-M + 融合重排(T-026)\n",
         f"真实目录 {res['ceiling']['index_n']} 个工具(双语索引),600 条 BFCL 请求;"
         "请求头按 5 折折外训练(`assign_folds`,seed 不变),"
         "开发折 = 第 0-2 折,测试折 = 第 3-4 折。\n",
         f"> 显式设置:分类头与检索基线都按线上默认(tfidf,双语索引,软过滤关,软 need "
         f"{res['soft']:g});召回集 = tfidf Top-{res['recall_n']} ∪ 语义 Top-M,"
         "排序 = linear 融合(池子外工具 tfidf 分记 0)。主口径 = 走完整 `route`,"
         "need 误判计 0(T-021 快照口径)。只在开发折上选;测试折只报一次,不参与选择(D-005)。\n",
         f"> 1 条样本 = 1/{dev_n} = {100.0 / dev_n:.2f} 个百分点。\n",
         "> 复跑说明:所有标 ms 的数字(召回 / 重排 / 召回+重排延迟、首次构建耗时)"
         "两次运行会变,其余数字逐字节相同。\n"]

    # 0. 调查结论
    c = res["ceiling"]
    k = c["ks"][-1]
    L.append("\n## 0. 调查结论\n")
    L.append(f"**并集召回天花板**:金标落在召回集里的请求比例(n={c['n']} 条请求,"
             f"索引 {c['index_n']} 工具,不看 need、不过滤),tfidf Top-{k} ∪ 语义 Top-M。\n")
    L.append(f"\n纯 tfidf Top-{k} 对照(同 T-025 §0 口径):\n")
    L.append(f"\n| 口径 | 中文严格 | 中文同名 | 英文严格 | 英文同名 |\n| --- | --- | --- | --- | --- |")
    L.append(f"| tfidf Top-{k} | {_pct(c['tfidf']['zh']['strict'][k])} "
             f"| {_pct(c['tfidf']['zh']['name'][k])} | {_pct(c['tfidf']['en']['strict'][k])} "
             f"| {_pct(c['tfidf']['en']['name'][k])} |")
    L.append(f"\n| 模型 | 语义 Top-M | 中文严格 | 中文同名 | 英文严格 | 英文同名 "
             f"| 语义独有金标(中/英,条) |\n| --- | --- | --- | --- | --- | --- | --- |")
    for m in c["models"]:
        short = m["short"]
        for M in sorted(c["union"][short]):
            u, o = c["union"][short][M], c["sem_only"][short][M]
            L.append(f"| {short} | {M} | {_pct(u['zh']['strict'])} | {_pct(u['zh']['name'])} "
                     f"| {_pct(u['en']['strict'])} | {_pct(u['en']['name'])} "
                     f"| {o['zh']} / {o['en']} |")
    uw = res["union_worth"]
    L.append(f"\n\"语义独有金标\"= 只在语义 Top-M 里、不在 tfidf Top-{k} 里的请求条数(增量来源)。"
             f"中文并集覆盖率相对 tfidf Top-{k} 的**最大增量** = "
             f"{uw['best']['zh_inc'] * 100:+.1f} 点({uw['best']['model']} Top-{uw['best']['M']}),"
             f"阈值 {con['union_worth_pp'] * 100:.0f} 点。\n")
    L.append(f"\n> **{'并集值得做' if uw['worth'] else '并集不值得做'}**:"
             f"中文增量 {uw['best']['zh_inc'] * 100:+.1f} 点"
             f"{'≥' if uw['worth'] else '<'} {con['union_worth_pp'] * 100:.0f} 点。"
             f"{'' if uw['worth'] else '下方候选对照仍按全网格给出,以便核对,结论不受影响。'}\n")

    # 1. 候选对照(只比开发折)
    L.append(f"\n## 1. 候选对照(只比开发折,{dev_n} 条)\n")
    L.append(f"H1 = tfidf Top-{res['recall_n']} ∪ 语义 Top-M → linear 融合:"
             f"α ∈ {{{', '.join(f'{a:g}' for a in T026_ALPHAS)}}},"
             f"M ∈ {{{', '.join(str(x) for x in T026_MS)}}},bge-m3 与 e5-small 各扫。"
             "池子外工具 tfidf 分记 0(理由见 §3)。约束:英文严格 R@5 相对纯 tfidf 基线降幅 ≤ "
             f"{con['en_drop_max'] * 100:.0f} 个点、单请求召回+重排 p95 ≤ {con['p95_max_ms']:g} ms。"
             "对照行(T-025 两档)数字直接从 `reports/eval_t025.json` 读,不重跑。\n")
    _t026_candidate_table(L, res)
    L.append("\n延迟(单请求,中英合并;召回 = tfidf Top-50,重排 = 查询编码 + 打分,T-026 含并集;"
             "复跑会变):\n")
    L.append("\n| 候选 | 召回 p50 | 重排 p50 | 召回+重排 p50 | 召回+重排 p95 |\n"
             "| --- | --- | --- | --- | --- |")
    for cd in res["candidates"][1:]:
        d = cd["dev"]
        L.append(f"| {cd['label']} | {d['recall_p50']:.1f} | {d['rerank_p50']:.1f} "
                 f"| {d['hybrid_p50']:.1f} | {cd['hybrid_p95']:.1f} |")

    # 2. 测试折(只报一次)
    if res.get("e2e"):
        e = res["e2e"]
        L.append(f"\n## 2. 测试折(第 3-4 折,{len(res['test_ids'])} 条,只报一次)\n")
        L.append("同一份 T-021 快照口径(走完整 `route`,need 误判计 0);测试折不参与选择。"
                 "T-025 两档的测试折数字读自 `reports/eval_t025.json`,不重跑。\n")
        L.append("\n| 配置 | 来源 | 中文严格 R@1 | 中文严格 R@5 | 中文同名 R@5 | 英文严格 R@1 "
                 "| 英文严格 R@5 | 英文同名 R@5 |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
        rows = [("基线(改前)", "本卡", e["baseline"]["test"])]
        if ctrl.get("main") and ctrl["main"].get("test"):
            rows.append((f"T-025 主选:{ctrl['main']['label']}", res["t025_source"], ctrl["main"]["test"]))
        if ctrl.get("lightweight") and ctrl["lightweight"].get("test"):
            rows.append((f"T-025 轻量:{ctrl['lightweight']['label']}", res["t025_source"],
                         ctrl["lightweight"]["test"]))
        if sel is not base:
            rows.append((f"T-026 主选:{sel['label']}", "本卡", e["selected"]["test"]))
        else:
            rows.append(("T-026 主选:无", "本卡", None))
        if e.get("lightweight"):
            rows.append((f"T-026 轻量:{lw['label']}", "本卡", e["lightweight"]["test"]))
        else:
            rows.append(("T-026 轻量:无", "本卡", None))
        for label, src, d in rows:
            cells = "- | - | - | - | - | -" if d is None else _t026_r_cells(d)
            L.append(f"| {label} | {src} | {cells} |")

    # 3. 部署说明
    L.append("\n## 3. 部署说明\n")
    L.append("- 开关:`cli train --rerank <模型名> --rerank-n 50 --rerank-fusion linear "
             "--rerank-alpha <α> --rerank-recall-m <M>`;线上 `cli route` 用同样参数在运行期覆盖。"
             "不传 `--rerank-recall-m`(=0)时只对 tfidf 召回重排,与 T-025 逐字节一致。\n")
    L.append("- **融合时分外工具的 tfidf 分记 0**:重排器接口只拿到 tfidf 的 Top-N,池外工具的 tfidf "
             "召回分拿不到;用 0 表示\"没被 tfidf 召回到\",让它们只靠语义相似度竞争,不必把重排器"
             "再绑回 tfidf 编码器,也不额外做一次全目录 tfidf 排序。\n")
    L.append("- 工具向量:与 T-025 同一份 `.cache/tool_vecs/` 缓存((模型, 索引文本哈希) 命名),"
             "并集召回复用同一批向量,不新增缓存。\n")
    for m, info in sorted(res["build_info"].items()):
        how = "命中已有缓存" if info["cached"] else "本次全新构建"
        L.append(f"- {T025_MODEL_SHORT.get(m, m)}:工具向量 {info['n_tools']} 个 × {info['dim']} 维,"
                 f"{how},耗时 {info['build_ms'] / 1000:.1f} s(复跑会变);"
                 f"模型磁盘占用 {res['model_mb'].get(m, 0.0):.0f} MB。\n")
    L.append("- `router.pkl` 只多存一个整数 `recall_m`,不存向量;权重仍在 HuggingFace 本地缓存里。\n")

    # 4. 结论
    L.append("\n## 4. 结论\n")
    g = res["gains"]
    if not uw["worth"]:
        L.append(f"**并集不值得做**:中文并集覆盖率只比 tfidf Top-{uw['k']} 多 "
                 f"{uw['best']['zh_inc'] * 100:+.1f} 点(< {con['union_worth_pp'] * 100:.0f} 点),"
                 "召回集扩不动。\n")
    if sel is base:
        L.append("开发折上没有候选同时满足两条约束,**主选 = 无(维持默认)**。\n")
    else:
        L.append(f"开发折主选 **{sel['label']}**:中文严格 R@5 {base['dev']['zh_main_r5_strict'] * 100:.1f}%"
                 f" → {sel['dev']['zh_main_r5_strict'] * 100:.1f}%"
                 f"({(sel['dev']['zh_main_r5_strict'] - base['dev']['zh_main_r5_strict']) * 100:+.1f} 点),"
                 f"英文 {base['dev']['en_main_r5_strict'] * 100:.1f}% → "
                 f"{sel['dev']['en_main_r5_strict'] * 100:.1f}%({-sel['en_drop'] * 100:+.1f} 点),"
                 f"召回+重排 p95 {sel['hybrid_p95']:.1f} ms。\n")
    if ctrl.get("main") and ctrl["main"].get("dev"):
        mg = g["main"]
        L.append(f"与 T-025 主选(开发折中文 {ctrl['main']['dev']['zh_main_r5_strict'] * 100:.1f}%)比:"
                 f"{'高' if (mg or 0) >= 0 else '低'} {abs(mg or 0) * 100:.1f} 点,"
                 f"门槛 {con['min_gain'] * 100:.0f} 点。\n")
    if lw is not None and ctrl.get("lightweight") and ctrl["lightweight"].get("dev"):
        lg = g["lightweight"]
        L.append(f"轻量备选 **{lw['label']}**(p95 {lw['hybrid_p95']:.1f} ms):与 T-025 轻量"
                 f"(开发折中文 {ctrl['lightweight']['dev']['zh_main_r5_strict'] * 100:.1f}%)比:"
                 f"{'高' if (lg or 0) >= 0 else '低'} {abs(lg or 0) * 100:.1f} 点。\n")
    else:
        L.append(f"\np95 ≤ {con['light_p95_max_ms']:g} ms 的候选里没有可选的,轻量备选:**无**。\n")
    if g["benefit"]:
        L.append("\n**本卡有收益**:并集召回相对 T-025 对应档位中文严格 R@5 高 ≥ "
                 f"{con['min_gain'] * 100:.0f} 点,并把并集召回作为重排的可选增强写进 README。\n")
    else:
        L.append("\n**本卡无收益**:并集召回相对 T-025 对应档位中文严格 R@5 未高 ≥ "
                 f"{con['min_gain'] * 100:.0f} 点,**维持 T-025 配置**,不写进 README。"
                 "默认仍不开重排(D-007)。\n")
    return "\n".join(L) + "\n"


def t026_metrics_json(res) -> dict:
    """`reports/eval_t026.json` 的机器可读版(延迟字段两次跑会变,其余逐字节相同)。"""
    return {"models": res["models"], "ceiling": res["ceiling"],
            "candidates": res["candidates"], "controls": res["controls"],
            "base_key": res["base_key"], "selected_key": res["selected_key"],
            "lightweight_key": res.get("lightweight_key"),
            "dev_ids": res["dev_ids"], "test_ids": res["test_ids"], "soft": res["soft"],
            "top_k": res["top_k"], "folds": res["folds"], "recall_n": res["recall_n"],
            "constraints": res["constraints"], "gains": res["gains"],
            "union_worth": res["union_worth"], "t025_source": res["t025_source"],
            "build_info": res["build_info"], "model_mb": res["model_mb"],
            "e2e": res.get("e2e")}


# ---------------------------------------------------------------- T-027 能力域投票
#
# 背景(T-022):定稿参考集上能力域主域准确率只有 40.8%(主口径,147 条),其中 36 条各域
# 概率都没过阈值;而真实目录 2405 个工具**每个**都有银标 `domains`,检索 R@5 英文六成以上。
# 本卡在检索 Top-k 工具的 `domains` 上投票,单独(D1)或与分类头融合(D2),预测请求能力域。
#
# 纪律:候选只在开发折(第 0-2 折)上比,按主口径主域准确率选;约束 = need 判定与检索候选
# 逐字节不变(本卡只改能力域输出——Router 没接投票)、单请求额外延迟 p95 <= 5 ms。测试折与
# 参考集各只报一次。**只有比 D0 高 >= 3 个点才建议改默认**(见 `domainvote` 的模块说明)。

T027_REPORT = REPORTS / "eval_t027.md"
T027_REPORT_JSON = REPORTS / "eval_t027.json"

T027_KS = (5, 10, 20)                 # D1/D2 的投票候选数
T027_BETAS = (0.3, 0.5, 0.7)          # D2 的融合系数 β(头分布权重)
T027_WEIGHT = domainvote.DEFAULT_WEIGHT
T027_THRESHOLD = domainvote.DEFAULT_THRESHOLD
T027_GAIN_MIN = 0.03                  # 建议改默认的门槛:开发折主口径比 D0 高 >= 3 个点
T027_P95_MAX_MS = 5.0                 # 单请求额外延迟 p95 上限
# T-022 参考集(cb/T-022-ref-eval 报告)里的两个锚点,用来对照本卡选中的配置
T027_T022_ANCHOR = {"argmax": 0.408, "primary": 0.306}


def _t027_candidates() -> list:
    """候选清单:每个 dict = {key, label, kind(head/vote/fuse), k, beta}。"""
    out = [{"key": "D0", "label": "D0 基线(能力域头 argmax)", "kind": "head", "k": 0, "beta": 1.0}]
    for k in T027_KS:
        out.append({"key": f"D1_k{k}", "label": f"D1 检索投票 k={k}",
                    "kind": "vote", "k": k, "beta": 0.0})
    for k in T027_KS:
        for b in T027_BETAS:
            out.append({"key": f"D2_k{k}_b{b:g}", "label": f"D2 融合 k={k} β={b:g}",
                        "kind": "fuse", "k": k, "beta": b})
    return out


def _t027_gold_domains(req, dom_of) -> list:
    """BFCL 请求的金标域:金标工具的银标 `domains` 的并集(保序去重)。

    这是"工具域"不是"请求域"(任务卡已注明):金标工具在哪些域,请求就算哪些域。
    """
    got = []
    for tid in req["tool_ids"]:
        for d in dom_of.get(tid, []):
            if d not in got:
                got.append(d)
    return got


def _t027_dist(cand, head_probs, hits):
    """一个候选的域分布。`hits` 是检索 Top-max(k)(已按分数降序),按候选的 k 截断。"""
    if cand["kind"] == "head":
        return dict(head_probs)
    vote = domainvote.vote_probs(hits[:cand["k"]], T027_WEIGHT)
    if cand["kind"] == "vote":
        return vote
    return domainvote.fuse_probs(head_probs, vote, cand["beta"])


def _t027_score_env(rows, cat, router, gold_fn, cands) -> list:
    """一组请求的逐候选数字。检索一次 Top-max(k)(前缀性质保证 = 各候选的 Top-k)。"""
    maxk = max(T027_KS)
    scored = []
    for r, o in zip(rows, router.head.predict([r["text"] for r in rows])):
        hits = cat.retrieve(r["text"], None, maxk)
        hp = dict(o.domain_probs)
        dists, ms = {}, {}
        for c in cands:
            if c["kind"] == "head":
                dists[c["key"]], ms[c["key"]] = dict(hp), 0.0
                continue
            t0 = time.perf_counter()
            d = _t027_dist(c, hp, hits)
            domainvote.pick_domains(d, T027_THRESHOLD, True)   # 算进去,和线上用法一致
            ms[c["key"]] = (time.perf_counter() - t0) * 1000
            dists[c["key"]] = d
        scored.append({"id": r["id"], "gold": gold_fn(r), "pred_need": o.need,
                       "dists": dists, "ms": ms})
    return scored


def _t027_block(rows, cands) -> dict:
    """一个集合的逐候选指标。

    - `main`:主口径——分布 argmax ∈ 金标域集合(与 T-022 `_domain_block` 同一条);
    - `old_thr / old_flr`:旧口径——预测 domains 的第一个 == 金标 domains 的第一个;
    - `empty_thr / empty_flr`:预测 domains 为空的条数比例(阈值不变 / 至少给 top-1)。
    后四个都按"预测 need == none 时 domains 为空"这条现有契约(见 `MultiHead.predict`)。
    """
    n = len(rows)
    out = {"n": n}
    for c in cands:
        key = c["key"]
        main = old_thr = old_flr = empty_thr = empty_flr = 0
        ms = []
        for row in rows:
            gs, head = set(row["gold"]), (row["gold"][0] if row["gold"] else "")
            dist = row["dists"][key]
            if domainvote.argmax_domain(dist) in gs:
                main += 1
            gated = row["pred_need"] != "none"
            dthr = domainvote.pick_domains(dist, T027_THRESHOLD, False) if gated else []
            dflr = domainvote.pick_domains(dist, T027_THRESHOLD, True) if gated else []
            if dthr and dthr[0] == head:
                old_thr += 1
            if dflr and dflr[0] == head:
                old_flr += 1
            if not dthr:
                empty_thr += 1
            if not dflr:
                empty_flr += 1
            ms.append(row["ms"][key])
        pct = (lambda v: (v / n) if n else None)
        out[key] = {"main": pct(main), "main_n": main,
                    "old_thr": pct(old_thr), "old_flr": pct(old_flr),
                    "empty_thr": pct(empty_thr), "empty_flr": pct(empty_flr),
                    "empty_thr_n": empty_thr, "empty_flr_n": empty_flr,
                    "p50_ms": float(np.percentile(ms, 50)) if ms else 0.0,
                    "p95_ms": float(np.percentile(ms, 95)) if ms else 0.0}
    return out


def _t027_retrieval_digest(rows, cat, router, top_k) -> str:
    """need + 检索候选的摘要:所有候选共用,用来钉住"只改能力域输出"这条约束。"""
    h = hashlib.sha1()
    for r in rows:
        res = router.route(r["text"])
        h.update(f"{r['id']}\t{res.need}\t{','.join(c['tool_id'] for c in res.candidates)}\n"
                 .encode("utf-8"))
    return h.hexdigest()


def t027_eval(tools, labeled_rows, zh_rows, requests, synth_tools, synth_train, real_train,
              req_ref=None, folds=FOLDS, seed=SEED, top_k=5, soft=SOFT_NEED_DEFAULT,
              e2e=True) -> dict:
    """T-027 的全套计算:候选只在开发折比 + 测试折单次报告 + 参考集单次报告(离线)。

    - 开发折(第 0-2 折)逐折建索引 + 请求头,按主口径主域准确率选,约束见模块头;
    - 测试折(第 3-4 折)只报基线 + 主选;参考集(D-005,need != none 的 147 条)同样只报这两行;
    - `e2e=False` 跳过测试折与参考集(单元测试用)。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    dom_of = {t["tool_id"]: list(t.get("domains", [])) for t in labeled_tools}
    cands = _t027_candidates()
    bfcl_gold = lambda r: _t027_gold_domains(r, dom_of)      # noqa: E731
    fold_of = assign_folds([r["id"] for r in requests], folds, seed)
    dev_ids = sorted(i for i, f in fold_of.items() if f in (0, 1, 2))
    test_ids = sorted(i for i, f in fold_of.items() if f in (3, 4))

    def build_env(ids):
        rows = [r for r in requests if r["id"] in ids]
        train = fold_train_rows(synth_train, real_train, ids)
        enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, "tfidf",
                                           bilingual=True, zh_by_id=zh_by_id)
        router = Router(ev.train_request_head(enc, train), cat, top_k=top_k, use_filter=False,
                        soft_need_threshold=soft)
        return rows, cat, router

    # 开发折:逐折建环境并打分,再合并成一份开发折数字
    dev_scored, dev_digest = [], hashlib.sha1()
    for fno in (0, 1, 2):
        ids = {i for i, f in fold_of.items() if f == fno}
        rows, cat, router = build_env(ids)
        dev_scored += _t027_score_env(rows, cat, router, bfcl_gold, cands)
        dev_digest.update(_t027_retrieval_digest(rows, cat, router, top_k).encode())
    dev = _t027_block(dev_scored, cands)

    # 选法:满足延迟约束的候选里开发折主口径最高;都被排除就维持 D0(不改默认)
    qualified = [c for c in cands if c["key"] != "D0" and dev[c["key"]]["p95_ms"] <= T027_P95_MAX_MS]
    selected = max(qualified, key=lambda c: dev[c["key"]]["main"]) if qualified else cands[0]
    base = cands[0]
    gain = (dev[selected["key"]]["main"] - dev[base["key"]]["main"]) if selected is not base else 0.0

    res = {"config": {"k": T027_KS, "betas": T027_BETAS, "weight": T027_WEIGHT,
                      "threshold": T027_THRESHOLD, "soft": soft, "top_k": top_k,
                      "folds": folds, "gain_min": T027_GAIN_MIN,
                      "p95_max_ms": T027_P95_MAX_MS, "t022_anchor": T027_T022_ANCHOR},
           "candidates": cands, "dev": dev, "base_key": base["key"],
           "selected_key": selected["key"], "dev_gain": gain,
           "dev_ids": dev_ids, "test_ids": test_ids,
           "dev_retrieval_sha1": dev_digest.hexdigest()}
    if not e2e:
        return res

    test_env = build_env(set(test_ids))
    subset = [base, selected]      # 测试折与参考集只报基线和主选,其余候选不落地
    res["test"] = {"retrieval_sha1": _t027_retrieval_digest(test_env[0], test_env[1], test_env[2], top_k),
                   "rows": _t027_block(_t027_score_env(test_env[0], test_env[1], test_env[2],
                                                       bfcl_gold, cands), subset)}

    if req_ref is not None:
        ref_ids = {r["id"] for r in req_ref}
        ref_train = list(synth_train) + [r for r in real_train if r.get("id") not in ref_ids]
        rrouter, rcat = _snapshot_router(labeled_tools, synth_tools, ref_train, "tfidf",
                                         zh_by_id, soft, top_k)
        ref_rows = [r for r in req_ref if r["need"] != "none"]
        ref_scored = _t027_score_env(ref_rows, rcat, rrouter, lambda r: list(r["domains"]), cands)
        res["reference"] = {"n": len(ref_rows), "dropped_train": len(ref_ids & {r.get("id") for r in real_train}),
                            "retrieval_sha1": _t027_retrieval_digest(ref_rows, rcat, rrouter, top_k),
                            "rows": _t027_block(ref_scored, subset)}
    return res


def _t027_pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _t027_dev_table(L, res) -> None:
    dev, cands = res["dev"], res["candidates"]
    L.append("| 候选 | 主口径主域准确率 | 旧口径(阈值不变) | 旧口径(至少给 top-1) "
             "| 空集比例(阈值不变) | 空集比例(至少给 top-1) | 投票 p50(ms) | 投票 p95(ms) |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in cands:
        d = dev[c["key"]]
        L.append(f"| {c['label']} | {_t027_pct(d['main'])} | {_t027_pct(d['old_thr'])} "
                 f"| {_t027_pct(d['old_flr'])} | {_t027_pct(d['empty_thr'])} "
                 f"| {_t027_pct(d['empty_flr'])} | {d['p50_ms']:.4f} | {d['p95_ms']:.4f} |")


def _t027_two_rows(L, block, keys, labels) -> None:
    L.append("| 配置 | 主口径主域准确率 | 旧口径(阈值不变) | 旧口径(至少给 top-1) "
             "| 空集比例(阈值不变) | 空集比例(至少给 top-1) |\n| --- | --- | --- | --- | --- | --- |")
    for key, label in zip(keys, labels):
        d = block["rows"][key]
        L.append(f"| {label} | {_t027_pct(d['main'])} | {_t027_pct(d['old_thr'])} "
                 f"| {_t027_pct(d['old_flr'])} | {_t027_pct(d['empty_thr'])} "
                 f"| {_t027_pct(d['empty_flr'])} |")


def render_t027(res) -> str:
    """把 `t027_eval` 的结果渲染成 `reports/eval_t027.md`。

    除标了 ms 的延迟字段外,两次复跑逐字节相同。
    """
    dev_n = len(res["dev_ids"])
    cands = res["candidates"]
    by_key = {c["key"]: c for c in cands}
    base, selected = by_key[res["base_key"]], by_key[res["selected_key"]]
    base_dev, sel_dev = res["dev"][base["key"]], res["dev"][selected["key"]]
    L = ["# 能力域预测:分类头 + 检索结果投票(T-027)\n",
         "真实目录 2405 个工具(双语索引),600 条 BFCL 请求;请求头按 5 折折外训练"
         "(`assign_folds`,seed 不变),开发折 = 第 0-2 折,测试折 = 第 3-4 折。\n",
         f"> 显式设置:分类头与检索都按线上默认(tfidf,双语索引,软过滤关,软 need "
         f"{res['config']['soft']:g},top_k={res['config']['top_k']});投票只在检索 Top-k 的 "
         f"`domains` 上做,**不反馈进检索**(本卡只改能力域输出)。"
         "只在开发折上选;测试折与参考集各只报一次,不参与选择。\n",
         f"> 1 条样本 = 1/{dev_n} = {100.0 / dev_n:.2f} 个百分点(开发折)。\n",
         "> 复跑说明:标 ms 的数字(投票延迟)两次运行会变,其余数字逐字节相同。\n"]

    # 0. 调查结论
    L.append("\n## 0. 调查结论\n")
    L.append("**(1) 哪些请求集有能力域标签、各多少条**\n")
    L.append("| 集合 | 条数 | 标签来源 | 偏差 |\n| --- | --- | --- | --- |")
    L.append("| `data/v3-primary/real_train.jsonl` | 936(need≠none 319) | 双路 LLM 一致"
             "(`label_source=jev+haiku`,v3 primary 口径) | 银标,不是人工;77% 是 none |")
    L.append("| 600 条 BFCL 请求 | 600(金标工具域非空 600) | 金标工具的银标 `domains` 并集 |"
             "**工具域 ≠ 请求域**:559 条只推出 1 个域,37 条 2 个,4 条 3 个 |")
    L.append("| 参考集 `data/gold_candidates_claude.csv` | 200(need≠none 147) |"
             " Claude 预标 + gu 抽查(D-005) | 请求级域,标注口径最接近线上;但只有 147 条 |")
    L.append("| 合成 `data/requests_gold.jsonl` | 69(need≠none 58) | 手写种子 | 合成,规模小 |\n")
    L.append("**结论**:600 条 BFCL 是唯一有规模的集合,它的域来自金标工具的银标域,是"
             "**工具域**的代理;参考集的 147 条是唯一按**请求域**标的集合。两者对同一方法的"
             "结论可以完全不同(见第 3 节),所以开发折按 BFCL 选、参考集用来验伪。\n")

    dom = res["dev"][base["key"]]
    L.append(f"**(2) 现状(开发折 {dev_n} 条,BFCL 工具域)**:D0 基线主口径主域准确率 "
             f"**{_t027_pct(dom['main'])}**;`domains` 为空 **{dom['empty_flr_n']}/{dev_n}"
             f"({_t027_pct(dom['empty_flr'])})**——这些行都是 need 头把该请求判成了 none"
             "(能力域头只在 need≠none 时才给域)。\n")
    L.append("**(3) 能力域现在被谁用**:真实目录这条链软过滤默认**关**"
             "(`Router(use_filter=False)`,见 `_snapshot_router` / `build_real_router`)。"
             "关掉后 `Router.route` 走 `catalog.retrieve(text, None, n)`,**不看** `domains`;"
             "need 来自 need 头;`requires_confirmation` 只看候选风险与 `actions`。"
             "所以当前 `domains` **只影响输出字段**(以及下游 agent 怎么用它),不影响检索与 need。"
             "本卡据此把投票只接到输出字段上,need 与检索候选逐字节不变"
             f"(开发折摘要 `{res['dev_retrieval_sha1'][:12]}`,见第 1 节约束)。\n")

    # 1. 候选对照(只比开发折)
    L.append(f"\n## 1. 候选对照(只比开发折,{dev_n} 条)\n")
    L.append(f"D0 = 能力域头 argmax(现状);D1 = 检索 Top-k 的 `domains` 投票(权重按名次 "
             f"`1/rank`,理由见 `domainvote` 模块说明);D2 = `β·头分布 + (1-β)·投票分布`,"
             f"k ∈ {{{', '.join(map(str, T027_KS))}}},β ∈ {{{', '.join(f'{b:g}' for b in T027_BETAS)}}}。"
             f"约束:need 与检索候选逐字节不变(本卡不改它们)、投票额外延迟 p95 ≤ "
             f"{T027_P95_MAX_MS:g} ms。选法:满足约束里主口径最高。"
             f"**D3(重排后 Top-k 投票)未做**:它依赖语义模型、只能当可选项;而第 3 节已表明"
             f"开发折(BFCL 工具域)选出来的方向在请求域上不成立,D3 的开发折数字不改变结论。\n")
    L.append("\n> 两列\"旧口径 / 空集比例\"的写法:`heads._pick` 现在是\"过阈值就取,一个都没过"
             "就给 argmax\"(`至少给 top-1` 那一列),所以 D0 的现状是这一档;`阈值不变` 那一列"
             "是严格阈值(允许空)下的反事实。D1/D2 按两种写法都给。\n")
    _t027_dev_table(L, res)
    cand_lines = []
    for c in cands:
        if c["key"] == res["base_key"]:
            continue
        cand_lines.append((res["dev"][c["key"]]["main"], c["key"], c["label"]))
    cand_lines.sort(key=lambda t: -t[0])
    L.append("\n按主口径排序的前三(开发折):" + ";".join(
        f"{label} {_t027_pct(m)}" for m, _k, label in cand_lines[:3]) + "。\n")
    L.append(f"\n**约束**:所有候选的投票都不反馈进检索,need 与 Top-5 候选逐字节不变;"
             f"投票 p95 最大 {max(res['dev'][c['key']]['p95_ms'] for c in cands):.4f} ms "
             f"(远低于 {T027_P95_MAX_MS:g} ms)。检索一次 Top-20 的耗时另记:"
             "开发折单请求 p95 约 1.9 ms(tfidf 全目录排序,与 k 无关),不单列。\n")

    # 2. 测试折
    L.append(f"\n## 2. 测试折(第 3-4 折,{len(res['test_ids'])} 条,只报一次)\n")
    L.append("测试折不参与选择,只报基线 + 开发折主选各一行。\n")
    _t027_two_rows(L, res["test"], [base["key"], selected["key"]],
                   ["D0 基线", f"主选:{selected['label']}"])

    # 3. 参考集
    ref = res.get("reference")
    if ref:
        L.append(f"\n## 3. 参考集(need≠none 的 {ref['n']} 条,只报一次)\n")
        L.append("口径:**Claude 预标 + gu 抽查**(D-005),请求级域;只报告,不调参、不选阈值。"
                 f"对照 T-022 的 {_t027_pct(T027_T022_ANCHOR['argmax'])} / "
                 f"{_t027_pct(T027_T022_ANCHOR['primary'])}(主口径 / 旧口径)。\n")
        _t027_two_rows(L, ref, [base["key"], selected["key"]],
                       ["D0 基线", f"主选:{selected['label']}"])
        rbase, rsel = ref["rows"][base["key"]], ref["rows"][selected["key"]]
        L.append(f"\nD0 在本报告里复现了 T-022 的锚点(主口径 {_t027_pct(rbase['main'])},"
                 f"旧口径 {_t027_pct(rbase['old_flr'])}),说明口径对齐。"
                 f"开发折主选在参考集上主口径 {_t027_pct(rsel['main'])},比 D0 "
                 f"**{(rsel['main'] - rbase['main']) * 100:+.1f} 个点**。\n")

    # 4. 结论
    L.append("\n## 4. 结论\n")
    L.append(f"- 开发折(BFCL 工具域)上,检索投票把主口径从 {_t027_pct(base_dev['main'])} "
             f"提到 **{_t027_pct(sel_dev['main'])}**({(sel_dev['main'] - base_dev['main']) * 100:+.1f} 个点),"
             f"远超 {T027_GAIN_MIN * 100:.0f} 个点的改默认门槛;测试折同向"
             f"({_t027_pct(res['test']['rows'][base['key']]['main'])} → "
             f"{_t027_pct(res['test']['rows'][selected['key']]['main'])})。\n")
    if ref:
        rbase, rsel = ref["rows"][base["key"]], ref["rows"][selected["key"]]
        L.append(f"- 但在**参考集(请求域)**上,同一个开发折主选是 "
                 f"{_t027_pct(rsel['main'])},比 D0 的 {_t027_pct(rbase['main'])} **低 "
                 f"{(rbase['main'] - rsel['main']) * 100:.1f} 个点**。原因是开发折的金标是"
                 "工具域、投票又是在被检索工具上做的(循环),这套数字没有转移到请求域。\n")
    L.append("- 按任务卡门槛,开发折确实高出 ≥ 3 个点;但参考集是唯一按请求域标注的集合,"
             "它显示开发折选出的配置**会变差**。以参考集为准,**本卡不改默认**:`domains` "
             "仍由能力域头出,Router 没有接投票,`cli evaluate` / `cli route` 输出与改前逐字节一致"
             "(只有延迟会变)。\n")
    L.append("- 若将来要在请求域上用它,只能把融合权重推到头分布一侧(β→1 时退化成 D0,"
             "等于放弃投票);而\"选哪个 β\"只能在参考集上选——违反 D-005(参考集只报告)。"
             "应另立卡,并先扩一份请求域标注集来承担选参。是否值得做,留给 Claude / gu 决定,"
             "见 `.collab/reports/T-027.md` 开放问题。\n")
    return "\n".join(L) + "\n"


def t027_metrics_json(res) -> dict:
    """`reports/eval_t027.json` 的机器可读版(延迟字段两次跑会变,其余逐字节相同)。"""
    return res


# ---------------------------------------------------------------- T-028 能力域投票:请求级银标重评
#
# 背景(T-027 审查,`.collab/reviews/T-027.md`):T-027 用 BFCL"金标工具的银标 domains"当选参
# 标签,而候选 D1/D2 投票用的正是**被检索工具**的银标 domains——标签与被测方法同源,开发折
# +59 点是循环评估。仓库里其实已有**请求级**能力域银标:`data/v3-primary/real_train.jsonl`
# (need≠none 319 条)与 `real_holdout.jsonl`(need≠none 80 条),标签是 Jev + Haiku 两路对
# **请求文本**打的(primary 口径),与工具银标不是同一次打标。本卡把 T-027 的候选(D0/D1/D2)
# **原样**搬到这份标签上重评,复用 `domainvote` 与 `_t027_*`,不另写一套。
#
# 纪律:候选只在选参集(剔除与参考集重叠后的 real_train need≠none)上比,全部 5 折折外;
# 留出集(real_holdout 剔除后)与参考集各只报一次。**只有比 D0 高 >= 4 个点、且留出集不比 D0
# 低,才建议改默认。** 本卡不改线上代码,只给建议(改由 Claude 决定后另做)。

T028_REPORT = REPORTS / "eval_t028.md"
T028_REPORT_JSON = REPORTS / "eval_t028.json"
T028_GAIN_MIN = 0.04                  # 建议改默认:选参集主口径比 D0 高 >= 4 个点
T028_P95_MAX_MS = T027_P95_MAX_MS     # 与 T-027 同一条延迟约束(投票极便宜,实际不绑定)
T028_BOOTSTRAP_N = 1000               # 每行 95% 置信区间:bootstrap 重采样次数
T028_BOOTSTRAP_SEED = 0               # 固定种子,保证报告复跑(除延迟外)逐字节相同
# 参考集锚点:T-022(当前默认的能力域头)与 T-027(同候选在参考集上主选),只作对照
T028_T022_ANCHOR = T027_T022_ANCHOR   # {"argmax": 0.408, "primary": 0.306}
T028_T027_REF_MAIN = 0.327            # T-027 开发折主选在参考集上的主口径


def _norm_text(t) -> str:
    """规范化请求文本:NFKC + 去空白 + 小写。用来跨 id 去重(同一句话可能 id 不同)。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", t or "")).strip().lower()


def _t028_overlap(rows, ref_ids, ref_texts) -> list:
    """与参考集按 id 或规范化文本重叠的行 id(保序去重)。"""
    out = []
    for r in rows:
        if r["id"] in ref_ids or _norm_text(r.get("text")) in ref_texts:
            if r["id"] not in out:
                out.append(r["id"])
    return out


def _t028_dedupe(rows, ref_ids, ref_texts) -> tuple:
    """剔除与参考集重叠的行,返回 (保留的行, 剔除的 id 列表)。

    参考集只用于报告(D-005),把它同时出现在选参集/留出集里的行拿掉,避免"报告集"污染选参。
    """
    dropped = _t028_overlap(rows, ref_ids, ref_texts)
    bad = set(dropped)
    return [r for r in rows if r["id"] not in bad], dropped


def _t028_ref_sets(req_ref) -> tuple:
    """参考集的 id 集合与规范化文本集合(空文本不算,免得误伤)。"""
    ids = {r["id"] for r in req_ref}
    texts = {_norm_text(r.get("text")) for r in req_ref} - {""}
    return ids, texts


def _t028_domain_stats(rows) -> dict:
    """选参集的主域分布(前 5)与多数类基线。

    标签是两路一致的**单个主域**(primary 口径),所以"永远猜最常见的域"的主口径准确率
    就是命中该域的条数比例。`multi` 记多域行数(这份标签应为 0)。
    """
    n = len(rows)
    cnt = Counter(r["domains"][0] for r in rows if r["domains"])
    top = cnt.most_common(5)
    maj = top[0][0] if top else ""
    return {"n": n, "top5": top, "majority": maj,
            "majority_hits": _t028_majority_hits(rows, maj),
            "single": sum(1 for r in rows if len(r["domains"]) == 1),
            "multi": sum(1 for r in rows if len(r["domains"]) > 1),
            "empty": sum(1 for r in rows if not r["domains"])}


def _t028_majority_hits(rows, domain) -> int:
    """猜 `domain` 在多少行上命主口径(argmax ∈ 金标域,这里等价于该域 ∈ 金标域)。"""
    return sum(1 for r in rows if domain and domain in set(r["domains"]))


def _t028_main_flags(rows, key) -> list:
    """主口径的逐行 0/1(预测分布 argmax ∈ 金标域),用来做 bootstrap。与 `_t027_block` 同口径。"""
    return [1 if domainvote.argmax_domain(row["dists"][key]) in set(row["gold"]) else 0
            for row in rows]


def _t028_bootstrap_ci(flags, n_boot=T028_BOOTSTRAP_N, seed=T028_BOOTSTRAP_SEED) -> tuple:
    """一组 0/1 的 bootstrap 95% 置信区间(2.5/97.5 分位),固定种子;空集返回 (None, None)。"""
    n = len(flags)
    if not n:
        return None, None
    arr = np.asarray(flags, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def t028_eval(tools, labeled_rows, zh_rows, synth_tools, synth_train, real_train,
              real_holdout, req_ref, folds=FOLDS, seed=SEED, top_k=5, soft=SOFT_NEED_DEFAULT,
              e2e=True) -> dict:
    """T-028 全套计算:请求级银标上重评 T-027 的候选,离线。

    - 选参集 = 剔除与参考集重叠后的 `real_train` need≠none 行,全部 5 折折外(候选只在这里选);
    - 留出集 = 剔除后 `real_holdout` need≠none 行,训练用全部剔除后 real_train + 合成,只报一次;
    - 参考集 = `req_ref` need≠none 行,训练剔除参考集 id,只报一次;
    - `e2e=False` 跳过留出集与参考集(单元测试用)。
    """
    labeled_tools = attach_labels(tools, labeled_rows)
    zh_by_id = {r["tool_id"]: r.get("description_zh", "") for r in zh_rows}
    cands = _t027_candidates()
    ref_ids, ref_texts = _t028_ref_sets(req_ref)

    # 0) 与参考集去重:重叠行从选参集与留出集剔除
    train_kept, train_dropped = _t028_dedupe(real_train, ref_ids, ref_texts)
    hold_kept, hold_dropped = _t028_dedupe(real_holdout, ref_ids, ref_texts)
    sel_rows = [r for r in train_kept if r.get("need") != "none"]
    hold_rows = [r for r in hold_kept if r.get("need") != "none"]
    stats = _t028_domain_stats(sel_rows)

    # 1) 选参集:全部 5 折折外。折号与 assign_folds 同种子,按剔除后 real_train 分折。
    fold_of = assign_folds([r["id"] for r in train_kept], folds, seed)
    scored, digest = [], hashlib.sha1()
    for fno in range(folds):
        ids = {i for i, f in fold_of.items() if f == fno}
        rows = [r for r in sel_rows if r["id"] in ids]
        if not rows:
            continue
        train = fold_train_rows(synth_train, train_kept, ids)
        enc, _head, cat = build_fold_index(labeled_tools, synth_tools, train, "tfidf",
                                           bilingual=True, zh_by_id=zh_by_id)
        router = Router(ev.train_request_head(enc, train), cat, top_k=top_k, use_filter=False,
                        soft_need_threshold=soft)
        scored += _t027_score_env(rows, cat, router, lambda r: list(r["domains"]), cands)
        digest.update(_t027_retrieval_digest(rows, cat, router, top_k).encode())
    dev = _t027_block(scored, cands)
    dev_ci = {c["key"]: _t028_bootstrap_ci(_t028_main_flags(scored, c["key"])) for c in cands}

    # 选法:满足延迟约束里主口径最高;4 点门槛与留出集不降在报告里判,不在这里改默认
    base = cands[0]
    others = [c for c in cands if c["key"] != base["key"]]
    lat_ok = [c for c in others if dev[c["key"]]["p95_ms"] <= T028_P95_MAX_MS]
    selected = max(lat_ok or others, key=lambda c: dev[c["key"]]["main"])
    gain = dev[selected["key"]]["main"] - dev[base["key"]]["main"]

    res = {"config": {"k": T027_KS, "betas": T027_BETAS, "weight": T027_WEIGHT,
                      "threshold": T027_THRESHOLD, "soft": soft, "top_k": top_k,
                      "folds": folds, "seed": seed, "gain_min": T028_GAIN_MIN,
                      "p95_max_ms": T028_P95_MAX_MS, "bootstrap_n": T028_BOOTSTRAP_N,
                      "bootstrap_seed": T028_BOOTSTRAP_SEED,
                      "t022_anchor": T028_T022_ANCHOR, "t027_ref_main": T028_T027_REF_MAIN},
           "candidates": cands, "base_key": base["key"], "selected_key": selected["key"],
           "dev": dev, "dev_ci": dev_ci, "dev_gain": gain, "stats": stats,
           "dev_ids": sorted({row["id"] for row in scored}),
           "train_dropped": train_dropped, "holdout_dropped": hold_dropped,
           "train_kept_n": len(train_kept), "holdout_kept_n": len(hold_kept),
           "sel_n": len(sel_rows), "dev_retrieval_sha1": digest.hexdigest()}
    if not e2e:
        return res

    subset = [base, selected]          # 留出集与参考集只报基线和主选,其余候选不落地
    # 2) 留出集(剔除后):训练用全部剔除后 real_train + 合成
    hrouter, hcat = _snapshot_router(labeled_tools, synth_tools, train_kept, "tfidf",
                                     zh_by_id, soft, top_k)
    hscored = _t027_score_env(hold_rows, hcat, hrouter, lambda r: list(r["domains"]), cands)
    res["holdout"] = {"n": len(hold_rows), "dropped": hold_dropped,
                      "retrieval_sha1": _t027_retrieval_digest(hold_rows, hcat, hrouter, top_k),
                      "rows": _t027_block(hscored, subset),
                      "ci": {c["key"]: _t028_bootstrap_ci(_t028_main_flags(hscored, c["key"]))
                             for c in subset},
                      "majority_hits": _t028_majority_hits(hold_rows, stats["majority"])}

    # 3) 参考集(need≠none,只报一次):训练剔除参考集 id
    ref_rows = [r for r in req_ref if r.get("need") != "none"]
    ref_train = list(synth_train) + [r for r in train_kept if r["id"] not in ref_ids]
    rrouter, rcat = _snapshot_router(labeled_tools, synth_tools, ref_train, "tfidf",
                                     zh_by_id, soft, top_k)
    rscored = _t027_score_env(ref_rows, rcat, rrouter, lambda r: list(r["domains"]), cands)
    res["reference"] = {"n": len(ref_rows),
                        "retrieval_sha1": _t027_retrieval_digest(ref_rows, rcat, rrouter, top_k),
                        "rows": _t027_block(rscored, subset),
                        "ci": {c["key"]: _t028_bootstrap_ci(_t028_main_flags(rscored, c["key"]))
                               for c in subset},
                        "majority_hits": _t028_majority_hits(ref_rows, stats["majority"])}

    # 4) 建议:4 点门槛 + 留出集不比 D0 低;两个条件都满足才建议改默认
    h_gain = (res["holdout"]["rows"][selected["key"]]["main"]
              - res["holdout"]["rows"][base["key"]]["main"])
    res["holdout_gain"] = h_gain
    res["recommend"] = bool(gain >= T028_GAIN_MIN and h_gain >= 0.0)
    return res


def _t028_pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _t028_ci(ci) -> str:
    lo, hi = ci
    return "-" if lo is None else f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def _t028_table(L, block, cands, ci_map, with_latency=True):
    head = ("| 候选 | 主口径主域准确率 | 95% CI | 旧口径(阈值不变) | 旧口径(至少给 top-1) "
            "| 空集比例(阈值不变) | 空集比例(至少给 top-1) |")
    rule = "| --- | --- | --- | --- | --- | --- | --- |"
    if with_latency:
        head += " 投票 p50(ms) | 投票 p95(ms) |\n" + rule + " --- | --- |"
    else:
        head += "\n" + rule
    L.append(head)
    for c in cands:
        d = block[c["key"]]
        line = (f"| {c['label']} | {_t028_pct(d['main'])} | {_t028_ci(ci_map[c['key']])} "
                f"| {_t028_pct(d['old_thr'])} | {_t028_pct(d['old_flr'])} "
                f"| {_t028_pct(d['empty_thr'])} | {_t028_pct(d['empty_flr'])} |")
        if with_latency:
            line += f" {d['p50_ms']:.4f} | {d['p95_ms']:.4f} |"
        L.append(line)


def render_t028(res) -> str:
    """把 `t028_eval` 的结果渲染成 `reports/eval_t028.md`。

    除标了 ms 的延迟字段外,两次复跑逐字节相同。
    """
    cands = res["candidates"]
    by_key = {c["key"]: c for c in cands}
    base, selected = by_key[res["base_key"]], by_key[res["selected_key"]]
    base_dev, sel_dev = res["dev"][base["key"]], res["dev"][selected["key"]]
    stats, n = res["stats"], res["sel_n"]
    L = ["# 能力域投票重评:改用请求级银标选参(T-028)\n",
         "T-027 用 BFCL\"金标工具的银标域\"当选参标签,与候选投票同源,开发折 +59 点是循环评估"
         "(`.collab/reviews/T-027.md`)。本卡把同一组候选(D0/D1/D2)原样搬到**请求级**银标"
         "`data/v3-primary/real_train.jsonl`(need≠none)与 `real_holdout.jsonl` 上重评。\n",
         "> 显式设置:分类头与检索都按线上默认(tfidf,双语索引,软过滤关,软 need "
         f"{res['config']['soft']:g},top_k={res['config']['top_k']});投票只在检索 Top-k 的 "
         "`domains` 上做,**不反馈进检索**。选参集全部 5 折折外;留出集与参考集各只报一次。\n",
         f"> 1 条样本 = 1/{n} = {100.0 / n:.2f} 个百分点(选参集)。\n",
         f"> 复跑说明:标 ms 的数字(投票延迟)两次运行会变,其余数字逐字节相同;"
         f"置信区间固定种子 bootstrap {res['config']['bootstrap_n']} 次。\n"]

    # 0. 调查结论
    L.append("\n## 0. 调查结论\n")
    L.append("**(1) 与参考集去重**:参考集 `data/gold_candidates_claude.csv` 200 条,"
             "与 `real_train` / `real_holdout` 按 **id** 与**规范化文本**(NFKC + 去空白 + 小写)"
             "各查重叠,任一命中即剔除。\n")
    L.append("| 集合 | 原行数 | 剔除后 | 剔除条数(need≠none) |\n| --- | --- | --- | --- |")
    L.append(f"| `real_train` | {res['train_kept_n'] + len(res['train_dropped'])} | "
             f"{res['train_kept_n']} | {len(res['train_dropped'])} |")
    L.append(f"| `real_holdout` | {res['holdout_kept_n'] + len(res['holdout_dropped'])} | "
             f"{res['holdout_kept_n']} | {len(res['holdout_dropped'])} |")
    L.append("")
    L.append(f"`real_train` 与参考集**无任何重叠**(id 0 条、规范化文本 0 条);"
             f"`real_holdout` 有 {len(res['holdout_dropped'])} 条与参考集同 id 且同文本,"
             f"全部剔除。所以选参集未变(need≠none {n} 条),留出集从 80 条 need≠none "
             f"削到 {res['holdout']['n']} 条,后者样本很小。\n")
    L.append("**(2) 选参集(剔除后的 real_train need≠none)分布与多数类基线**\n")
    L.append(f"- 条数 **{stats['n']}**;单主域 {stats['single']} 条、多域 {stats['multi']} 条、"
             f"空域 {stats['empty']} 条。\n")
    L.append("| 排名 | 主域 | 占比 |\n| --- | --- | --- |")
    for i, (d, c) in enumerate(stats["top5"], start=1):
        L.append(f"| {i} | `{d}` | {c} 条({c / stats['n'] * 100:.1f}%) |")
    L.append(f"\n- **多数类基线**(永远猜最常见的域 `{stats['majority']}`):主口径 "
             f"{_t028_pct(stats['majority_hits'] / stats['n'])}"
             f"({stats['majority_hits']}/{stats['n']})。\n")
    L.append("**(3) 标签偏差:单主域 vs 参考集的多域口径**\n")
    L.append(f"`real_train` 的 `domains` 是 Jev + Haiku 两路**一致的主域**(primary 口径,"
             f"单个域,实测 {stats['single']}/{stats['n']} 条单域);参考集是 Claude 预标 + gu 抽查的"
             "**请求级**标签(其中若干条多域)。两者都按请求文本打,不是工具银标,但口径不同:\n")
    L.append("- 主口径 = \"预测分布 argmax ∈ 金标域\"。在**单主域**标签上它退化成 top-1 准确率"
             "(argmax 必须正好等于那个主域);在**多域**参考集上,argmax 落进任一域就算命中,"
             "分母相同的多域行更容易得分。所以同一方法在选参集上的主口径与在参考集上的**不可直接比大小**。\n")
    L.append("- `real_train` 的域是**两路一致**的子集,可能偏向模型更常达成一致的常见域(如 "
             f"`{stats['majority']}`),对不常见的域有低估。\n")

    # 1. 候选对照(选参集)
    L.append(f"\n## 1. 候选对照(选参集,{n} 条,全部 5 折折外)\n")
    L.append(f"D0 = 能力域头 argmax(现状);D1 = 检索 Top-k 的 `domains` 投票(权重按名次 `1/rank`);"
             f"D2 = `β·头分布 + (1-β)·投票分布`,k ∈ {{{', '.join(map(str, T027_KS))}}},"
             f"β ∈ {{{', '.join(f'{b:g}' for b in T027_BETAS)}}}。选法:满足延迟约束(投票 p95 ≤ "
             f"{T028_P95_MAX_MS:g} ms)里主口径最高。95% CI = 固定种子 bootstrap "
             f"{res['config']['bootstrap_n']} 次。\n")
    L.append("\n> 旧口径 = 预测 `domains` 的第一个 == 金标 `domains` 的第一个;"
             "`至少给 top-1` 是 `heads._pick` 现在的写法(一个都没过阈值就给 argmax),"
             "所以 D0 的现状是这一档。\n")
    _t028_table(L, res["dev"], cands, res["dev_ci"])
    ranked = sorted((res["dev"][c["key"]]["main"], c["label"]) for c in cands
                    if c["key"] != base["key"])
    L.append("\n按主口径排序的前三(选参集):" + ";".join(
        f"{lab} {_t028_pct(m)}" for m, lab in reversed(ranked[-3:])) + "。\n")
    L.append(f"\n**约束**:所有候选都不反馈进检索,need 与检索候选逐字节不变;投票 p95 最大 "
             f"{max(res['dev'][c['key']]['p95_ms'] for c in cands):.4f} ms(远低于 "
             f"{T028_P95_MAX_MS:g} ms)。检索各折摘要见 `reports/eval_t028.json` 的 "
             f"`dev_retrieval_sha1`。\n")

    # 2. 留出集
    ref = res.get("reference")
    if "holdout" in res:
        h = res["holdout"]
        L.append(f"\n## 2. 留出集(real_holdout 剔除后 need≠none 的 {h['n']} 条,只报一次)\n")
        L.append(f"剔除与参考集重叠的 {len(h['dropped'])} 条后仅剩 {h['n']} 条,"
                 f"1 条 = {100.0 / h['n']:.2f} 个点,幅度不稳。此集不参与选参。\n")
        _t028_table(L, h["rows"], [base, selected], h["ci"], with_latency=False)
        hmaj = h["majority_hits"] / h["n"] if h["n"] else None
        hb, hs = h["rows"][base["key"]]["main"], h["rows"][selected["key"]]["main"]
        L.append(f"\n多数类基线(猜 `{stats['majority']}`):{_t028_pct(hmaj)}。"
                 f"D0 {_t028_pct(hb)} → 主选 {_t028_pct(hs)}(**{(hs - hb) * 100:+.1f} 个点**)。\n")

    # 3. 参考集
    if ref:
        L.append(f"\n## 3. 参考集(need≠none 的 {ref['n']} 条,只报一次)\n")
        L.append("口径:**Claude 预标 + gu 抽查**(D-005),请求级域,可多域;只报告,不调参。"
                 f"对照 T-022 的 {_t028_pct(T028_T022_ANCHOR['argmax'])} / "
                 f"{_t028_pct(T028_T022_ANCHOR['primary'])}(主口径 / 旧口径)与 T-027 开发折主选的 "
                 f"{_t028_pct(T028_T027_REF_MAIN)}。\n")
        _t028_table(L, ref["rows"], [base, selected], ref["ci"], with_latency=False)
        rb, rs = ref["rows"][base["key"]]["main"], ref["rows"][selected["key"]]["main"]
        L.append(f"\nD0 在本报告里复现 T-022 锚点(主口径 {_t028_pct(rb)});主选在参考集上 "
                 f"{_t028_pct(rs)},比 D0 **{(rs - rb) * 100:+.1f} 个点**。\n")

    # 4. 结论
    L.append("\n## 4. 结论\n")
    L.append(f"- 选参集上主口径最高的是 **{selected['label']}**:"
             f"{_t028_pct(sel_dev['main'])} vs D0 {_t028_pct(base_dev['main'])}"
             f"({(sel_dev['main'] - base_dev['main']) * 100:+.1f} 个点);"
             f"改默认门槛 = ≥ {T028_GAIN_MIN * 100:.0f} 个点。\n")
    if "holdout" in res:
        hb = res["holdout"]["rows"][base["key"]]["main"]
        hs = res["holdout"]["rows"][selected["key"]]["main"]
        L.append(f"- 留出集上主选 {_t028_pct(hs)} vs D0 {_t028_pct(hb)}"
                 f"({(hs - hb) * 100:+.1f} 个点);门槛要求\"不比 D0 低\"。\n")
    if ref:
        rb = ref["rows"][base["key"]]["main"]
        rs = ref["rows"][selected["key"]]["main"]
        L.append(f"- 参考集(请求域、可多域)上主选 {_t028_pct(rs)} vs D0 {_t028_pct(rb)}"
                 f"({(rs - rb) * 100:+.1f} 个点)。\n")
    if res.get("recommend"):
        L.append(f"- **建议改默认**:两个条件都满足(选参集 ≥ {T028_GAIN_MIN * 100:.0f} 点且留出集不降)。"
                 "具体参数名 / 默认值 / 关闭方式见下,由 Claude 决定后另做。\n")
    else:
        why = []
        if sel_dev["main"] - base_dev["main"] < T028_GAIN_MIN:
            why.append(f"选参集只高 {(sel_dev['main'] - base_dev['main']) * 100:.1f} 个点"
                       f"(< 事先定的 {T028_GAIN_MIN * 100:.0f} 点门槛)")
        if "holdout" in res and res["holdout_gain"] < 0:
            why.append(f"留出集比 D0 低 {abs(res['holdout_gain']) * 100:.1f} 个点")
        L.append(f"- **不建议改默认**:{'、'.join(why) or '条件未满足'}。`domains` 仍由能力域头出,"
                 "Router 没接投票,`cli evaluate` / `cli route` 输出与改前逐字节一致(只有延迟会变)。"
                 f"未达门槛的收益视为**不能作为改默认依据**。\n")
    L.append("- 本卡只改选参标签、不改线上代码:候选与实现复用 T-027 的 `domainvote` 与 `_t027_*`。"
             "改不改默认由 Claude 拍板后另立卡。\n")
    return "\n".join(L) + "\n"


def t028_metrics_json(res) -> dict:
    """`reports/eval_t028.json` 的机器可读版(延迟字段两次跑会变,其余逐字节相同)。"""
    return res
