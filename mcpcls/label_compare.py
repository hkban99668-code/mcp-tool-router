"""新旧标签口径对照(T-009 / T-010a)。

need 说明从 v2 换成 v3、能力域从 12 个加到 14 个之后,两套标签要能并排放着看:
双路一致率、need 分布、留出集里"需要工具"的条数、同一条请求 need 的转移矩阵,
以及各自用 C 方案(合成训练集 + real_train)重训后在**自己**留出集上的指标。

T-010a 起可以同时比多套口径(`compare_sides`),因为"标签口径"这一维已经从两个值
(v2 / v3)长成两组值的组合:need 说明版本 × 一致判定规则(exact / primary / primary_sym)。
每套口径一个目录(里面四份同名 jsonl),自己带一个名字显示在报告里。

报告的第五节是两个不同的留出集:各口径的标签不是同一批,指标只能看趋势,不能逐点比较。
第六节把 multi 的标注情况单列出来 —— multi 在真实数据里一直很少,值得单独判断它学不学得出来。
只读几套 jsonl,不联网、不打标、不写 data/。
"""

import pathlib
from collections import Counter

from . import evaluate as ev
from .taxonomy import DOMAIN_IDS, NEED_IDS

# 每个口径目录下的文件名(相对目录),缺哪个记在 missing 里
FILES = {"agree": "labeled_agree.jsonl", "review": "review_queue.jsonl",
         "train": "real_train.jsonl", "holdout": "real_holdout.jsonl"}


def _load(path):
    return ev.load_jsonl(path) if pathlib.Path(path).exists() else []


def load_side(directory, label=None) -> dict:
    """读一侧(一个口径目录)的四个文件;不存在的文件算空,名字记在 missing 里。

    label 是这一侧在报告里的名字(给了标签就用它,否则用目录名)。
    """
    d = pathlib.Path(directory)
    name = label or d.name or str(d)
    if not d.exists():
        return {"dir": str(d), "label": name, "missing": list(FILES.values()),
                "agree": [], "review": [], "train": [], "holdout": []}
    out = {"dir": str(d), "label": name, "missing": []}
    for key, name in FILES.items():
        out[key] = []
        if not (d / name).exists():
            out["missing"].append(name)
            continue
        out[key] = _load(d / name)
    return out


def _pairs(side) -> list:
    """两路都给出了答案的那些请求,每条记 {(source), (agreed), (need_same)}。

    有一路没问到答案的两侧不一致,那是"没答上来"不是"判得不一样",所以整条跳过,
    条数在 `skipped` 里单独报出来。
    """
    out = []
    for r in side["agree"]:
        out.append({"source": r.get("source", ""), "agreed": True, "need_same": True})
    for r in side["review"]:
        names = r.get("labelers") or []
        got = [(r.get(n) or {}) for n in names]
        if len(got) < 2 or not all(g.get("ok") for g in got):
            continue
        out.append({"source": r.get("source", ""), "agreed": False,
                    "need_same": len({g.get("need") for g in got}) == 1})
    return out


def _summary(pairs) -> dict:
    n = len(pairs)
    agreed = sum(1 for p in pairs if p["agreed"])
    need_same = sum(1 for p in pairs if p["agreed"] or p["need_same"])
    return {"n": n, "agreed": agreed, "agree_rate": (agreed / n) if n else None,
            "need_agree": need_same, "need_agree_rate": (need_same / n) if n else None}


def agreement(side) -> dict:
    """一侧的一致率:整体 + 按来源;review 里有一路没答的条数记在 skipped。"""
    pairs = _pairs(side)
    out = _summary(pairs)
    out["skipped"] = len(side["agree"]) + len(side["review"]) - len(pairs)
    out["review_n"] = len(side["review"])
    sources = {}
    for src in sorted({p["source"] for p in pairs}):
        sources[src] = _summary([p for p in pairs if p["source"] == src])
    out["by_source"] = sources
    return out


def _side_labelers(side) -> list:
    """这一侧的两路名字:review 行自带 `labelers`,老文件里没有就按 T-006 的两路算。"""
    for r in side["review"]:
        names = r.get("labelers")
        if names:
            return list(names)
    return ["jev", "haiku"]


def multi_stats(side) -> dict:
    """multi 在这一侧的标注情况(T-010a 第 4 节要的那几个数)。

    两路各判了多少条 multi、两路都判 multi 多少条、一路 multi 另一路 single 多少条(按来源拆)。
    合成数据里 multi 是主力之一,真实数据里却极少:这几个数用来区分"标得少"和"两路标得不一样"。
    """
    names = _side_labelers(side)
    n1 = n2 = both = 0
    mixed = []
    for r in side["agree"]:                  # 一致的条目:两路的 need 一样
        if r.get("need") == "multi":
            n1 += 1
            n2 += 1
            both += 1
    for r in side["review"]:
        a, b = (r.get(names[0]) or {}), (r.get(names[1]) or {})
        if not a.get("ok") or not b.get("ok"):
            continue                          # 有一路没答上来,不是"判成了别的"
        na, nb = a.get("need"), b.get("need")
        n1 += int(na == "multi")
        n2 += int(nb == "multi")
        both += int(na == "multi" and nb == "multi")
        if {na, nb} == {"multi", "single"}:
            mixed.append(r.get("source", ""))
    return {"labelers": names, f"{names[0]}_multi": n1, f"{names[1]}_multi": n2,
            "both_multi": both,
            "multi_vs_single": {"total": len(mixed), "by_source": dict(Counter(mixed))}}


def need_distribution(agreed) -> dict:
    """一致集里的 need 分布(按 NEED_IDS 顺序,缺的补 0)。"""
    counts = Counter(r["need"] for r in agreed)
    return {k: counts.get(k, 0) for k in NEED_IDS}


def domain_counts(agreed) -> dict:
    """一致集里每个能力域被标了多少条(多值算多次),按 DOMAIN_IDS 顺序。"""
    counts = Counter(d for r in agreed for d in r.get("domains") or [])
    return {k: counts.get(k, 0) for k in DOMAIN_IDS}


def holdout_needs_tool(holdout) -> dict:
    """留出集里"需要工具"(single / multi)的条数。"""
    counts = Counter(r["need"] for r in holdout)
    needs = sum(counts.get(k, 0) for k in NEED_IDS if k != "none")
    return {"n": len(holdout), "needs_tool": needs, "none": counts.get("none", 0),
            "needs_tool_rate": (needs / len(holdout)) if holdout else None}


def _need_by_id(rows) -> dict:
    return {r["id"]: r["need"] for r in rows if r.get("id")}


def _matrix(pairs) -> dict:
    """{(旧 need, 新 need): 条数}"""
    out = Counter((a, b) for a, b in pairs)
    return {f"{a}->{b}": c for (a, b), c in sorted(out.items())}


def transitions(old, new) -> dict:
    """同一条请求在两版里 need 的变化:整体 + 按来源 + 按 math_like。

    只比两侧都有双路一致标签的请求;只在一侧一致的不算转移。
    """
    old_map, new_map = _need_by_id(old["agree"]), _need_by_id(new["agree"])
    ids = sorted(set(old_map) & set(new_map))
    rows = old["agree"] + new["agree"]
    meta = {}
    for r in rows:
        if r["id"] in ids:
            meta[r["id"]] = {"source": r.get("source", ""),
                             "math_like": bool((r.get("hint") or {}).get("math_like"))}
    groups = {"全部": ids}
    for src in sorted({m["source"] for m in meta.values()}):
        groups[f"source={src}"] = [i for i in ids if meta[i]["source"] == src]
    for want in (True, False):
        groups[f"math_like={want}"] = [i for i in ids if meta[i]["math_like"] is want]
    out = {"n_compared": len(ids),
           "only_old": len(set(old_map) - set(new_map)),
           "only_new": len(set(new_map) - set(old_map))}
    out["groups"] = {name: _matrix([(old_map[i], new_map[i]) for i in group])
                     for name, group in groups.items()}
    return out


def eval_version(encoder, tools, synth_train, side) -> dict:
    """C 方案(合成训练集 + real_train)在这一侧留出集上的指标。

    留出集里一行样本就可能让 need 准确率动好几个百分点,且两版的留出集标签不是同一批,
    所以这里的数字只能看趋势,不能把两版的数字当成同一个评估集上的差别。
    """
    train, holdout = side["train"], side["holdout"]
    if not train or not holdout:
        return None
    rows = [dict(r, tool_ids=r.get("tool_ids") or []) for r in list(synth_train) + list(train)]
    router = ev.build_system(encoder, tools, rows)
    outs = router.head.predict([r["text"] for r in holdout])
    m = ev.request_metrics(holdout, [(o.need, o.domains, o.actions) for o in outs])
    return {"train_n": len(rows), "n": len(holdout), "need_accuracy": m["need_accuracy"],
            "need_macro_f1": m["need_macro_f1"], "recall_needs_mcp": m["recall_needs_mcp"],
            "none_recall": (m["none_correct"] / m["none_total"]) if m["none_total"] else None,
            "none_correct": m["none_correct"], "none_total": m["none_total"],
            "domain": m["domain"], "action": m["action"]}


def side_keys(sides) -> list:
    """每套口径在结果里的键:前两套沿用 `old` / `new`,再往后用它的名字(重复就加下标)。"""
    keys = []
    for i, (label, _) in enumerate(sides):
        key = ("old", "new")[i] if i < 2 else str(label)
        while key in keys:
            key = f"{key}{i}"
        keys.append(key)
    return keys


def compare_sides(sides, tools, requests, encoder="tfidf") -> dict:
    """算任意多套口径的对照数字;sides 是 [(名字, 目录), ...]。

    每套口径缺文件的一侧算空,名字留在 missing 里。转移矩阵只在**前两套**之间算
    (它要的就是"同一条请求从旧口径变到新口径"这件事)。
    """
    keys = side_keys(sides)
    loaded = [(k, label, load_side(directory, label)) for k, (label, directory) in zip(keys, sides)]
    out = {"encoder": encoder, "keys": keys,
           "labels": {k: label for k, label, _ in loaded},
           "dirs": {k: s["dir"] for k, _, s in loaded},
           "missing": {k: s["missing"] for k, _, s in loaded},
           "n": {k: {"agree": len(s["agree"]), "review": len(s["review"]),
                     "train": len(s["train"]), "holdout": len(s["holdout"])}
                 for k, _, s in loaded},
           "agreement": {k: agreement(s) for k, _, s in loaded},
           "need_distribution": {k: need_distribution(s["agree"]) for k, _, s in loaded},
           "domain_counts": {k: domain_counts(s["agree"]) for k, _, s in loaded},
           "holdout": {k: holdout_needs_tool(s["holdout"]) for k, _, s in loaded},
           "multi": {k: multi_stats(s) for k, _, s in loaded},
           "eval": {k: eval_version(encoder, tools, requests, s) for k, _, s in loaded}}
    if len(loaded) >= 2 and loaded[0][2]["agree"] and loaded[1][2]["agree"]:
        out["transitions"] = transitions(loaded[0][2], loaded[1][2])
    else:
        out["transitions"] = {"n_compared": 0, "only_old": 0, "only_new": 0, "groups": {}}
    return out


def compare_labels(old_dir, new_dir, tools, requests, encoder="tfidf") -> dict:
    """两套口径的对照(T-009 的老接口);`compare_sides` 的两套情形。"""
    out = compare_sides([("v2-exact", old_dir), ("v3-exact", new_dir)], tools, requests, encoder)
    out["old_dir"] = str(old_dir)
    out["new_dir"] = str(new_dir)
    return out


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _needs_tool(need_counts) -> int:
    """一致集里"需要工具"的条数(single + multi)。"""
    return sum(need_counts.get(k, 0) for k in NEED_IDS if k != "none")


def _summary_table(result) -> list:
    """口径总表(T-010a):一行一套口径,列是挑取舍要一起看的那些数。"""
    keys = result["keys"]
    L = ["## 口径总表(need 说明版本 × 一致判定规则)\n",
         "| 口径 | 目录 | 有效条数 | 一致率 | need 一致率 | 需要工具的样本数 | weather_geo | "
         "entertainment | 留出集 need 准确率 | need macro-F1 | 需要 MCP 召回 | none 召回 | "
         "能力域 micro-F1 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
         "--- | --- | --- | --- |"]
    for k in keys:
        a, nd, dc, ev_ = result["agreement"][k], result["need_distribution"][k], \
            result["domain_counts"][k], result["eval"][k]
        L.append(f"| {result['labels'][k]} | `{result['dirs'][k]}` | {a['n']} "
                 f"| {_pct(a['agree_rate'])} | {_pct(a['need_agree_rate'])} "
                 f"| {_needs_tool(nd)} | {dc.get('weather_geo', 0)} | {dc.get('entertainment', 0)} "
                 f"| {_pct(ev_ and ev_['need_accuracy'])} | {_pct(ev_ and ev_['need_macro_f1'])} "
                 f"| {_pct(ev_ and ev_['recall_needs_mcp'])} | {_pct(ev_ and ev_['none_recall'])} "
                 f"| {_pct(ev_ and ev_['domain']['micro_f1'])} |")
    L.append("\n“有效条数”的分母是两路都答上来的条目;留出集那五列是 C 方案(合成 + real_train)"
             "在**各自**留出集上的数字,不同口径的留出集不是同一批标签,只能看趋势。")
    return L


def render(result) -> str:
    keys = result["keys"]
    labels = {k: result["labels"][k] for k in keys}
    head = "、".join(f"`{result['dirs'][k]}`({labels[k]})" for k in keys)
    L = ["# 标签口径对照(T-009 / T-010a)\n",
         f"比的是 {len(keys)} 套口径:{head};编码器 `{result['encoder']}`。\n",
         "> **各口径的留出集不是同一批标签**:标签是重新判定出来的,"
         "所以第五节的指标只能看趋势,不能逐点比较。\n"]
    L += _summary_table(result)

    miss = result["missing"]
    if any(miss.values()):
        L.append("\n> 缺文件(按 0 条算):"
                 + "；".join(f"`{labels[k]}` 目录没有 {', '.join(v)}"
                             for k, v in miss.items() if v) + "。\n")

    n = result["n"]
    L.append("\n## 1. 数据规模与双路一致率\n")
    L.append("| 口径 | 一致集 | 待人工 | real_train | 留出集 | 有效条数 | 双路一致 | 一致率 | "
             "need 一致率 | 有一路没答 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in keys:
        a, cnt = result["agreement"][k], n[k]
        L.append(f"| {labels[k]} | {cnt['agree']} | {cnt['review']} | {cnt['train']} "
                 f"| {cnt['holdout']} "
                 f"| {a['n']} | {a['agreed']} | {_pct(a['agree_rate'])} | {_pct(a['need_agree_rate'])} "
                 f"| {a['skipped']} |")
    L.append("\n“有一路没答”的条目既不算一致也不算不一致，已从上表的有效条数里排除。\n")
    L.append("\n**按来源**(有效条数、一致率)\n")
    srcs = sorted({s for k in keys for s in result["agreement"][k]["by_source"]})
    L.append("| source | " + " | ".join(f"{labels[k]} 有效 | {labels[k]} 一致率" for k in keys)
             + " |\n| --- |" + " --- |" * (2 * len(keys)))
    for s in srcs:
        cells = []
        for k in keys:
            g = result["agreement"][k]["by_source"].get(s)
            cells += [g["n"] if g else 0, _pct(g["agree_rate"]) if g else "-"]
        L.append(f"| {s} | " + " | ".join(str(c) for c in cells) + " |")

    L.append("\n## 2. 一致集里的 need 分布与能力域样本数\n")
    nd = result["need_distribution"]
    L.append("| need | " + " | ".join(f"{labels[k]}(条)" for k in keys)
             + " |\n| --- |" + " --- |" * len(keys))
    for need in NEED_IDS:
        L.append(f"| {need} | " + " | ".join(str(nd[k][need]) for k in keys) + " |")
    dc = result["domain_counts"]
    L.append("\n| 能力域 | " + " | ".join(f"{labels[k]}(条)" for k in keys)
             + " |\n| --- |" + " --- |" * len(keys))
    for d in DOMAIN_IDS:
        L.append(f"| {d} | " + " | ".join(str(dc[k][d]) for k in keys) + " |")
    L.append("\nweather_geo / entertainment 是 v3 起的新域:v2 口径的答案里没有这两题,"
             "v2 那一列一定是 0,只有 v3 重新打标后才可能不为 0。")

    L.append("\n## 3. 留出集里“需要工具”的样本数\n")
    h = result["holdout"]
    L.append("| 口径 | 留出集 | 需要工具(single/multi) | none | 占比 |\n| --- | --- | --- | --- | --- |")
    for k in keys:
        r = h[k]
        L.append(f"| {labels[k]} | {r['n']} | {r['needs_tool']} | {r['none']} | "
                 f"{_pct(r['needs_tool_rate'])} |")

    tr = result["transitions"]
    if len(keys) >= 2 and tr.get("groups"):
        L.append(f"\n## 4. need 转移矩阵({labels[keys[0]]} → {labels[keys[1]]};"
                 f"共 {tr['n_compared']} 条两套口径都有一致标签;"
                 f"只在前一套 {tr['only_old']} 条,只在后一套 {tr['only_new']} 条)\n")
        L.append("| 分组 | none→single | single→none | none→multi | multi→none | single→multi | "
                 "multi→single | 不变 |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, matrix in tr["groups"].items():
            changed = sum(c for kk, c in matrix.items() if kk.split("->")[0] != kk.split("->")[1])
            stable = sum(matrix.values()) - changed

            def cell(a, b):
                return matrix.get(f"{a}->{b}", 0)

            L.append(f"| {name} | {cell('none', 'single')} | {cell('single', 'none')} "
                     f"| {cell('none', 'multi')} | {cell('multi', 'none')} "
                     f"| {cell('single', 'multi')} | {cell('multi', 'single')} | {stable} |")
    elif len(keys) >= 2:
        L.append(f"\n## 4. need 转移矩阵\n\n两套口径都缺一致集,这一节没跑。")

    L.append("\n## 5. C 方案(合成 + real_train)在各自留出集上的指标\n")
    e = result["eval"]
    if not any(e.values()):
        L.append("所有口径都缺 real_train 或留出集,这一节没跑。")
    else:
        L.append("| 口径 | 训练样本 | 留出集 | need 准确率 | need macro-F1 | 需要 MCP 召回 | "
                 "none 召回 | 能力域 micro/macro | 动作 micro/macro |\n"
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for k in keys:
            r = e[k]
            if not r:
                L.append(f"| {labels[k]} | - | - | - | - | - | - | - | - |")
                continue
            L.append(f"| {labels[k]} | {r['train_n']} | {r['n']} | {_pct(r['need_accuracy'])} "
                     f"| {_pct(r['need_macro_f1'])} | {_pct(r['recall_needs_mcp'])} "
                     f"| {r['none_correct']}/{r['none_total']} "
                     f"| {_pct(r['domain']['micro_f1'])} / {_pct(r['domain']['macro_f1'])} "
                     f"| {_pct(r['action']['micro_f1'])} / {_pct(r['action']['macro_f1'])} |")
        L.append("\n各口径留出集的标签来自各自口径下的双路一致,**不是同一批标签**:"
                 "这里的差别里既有标签口径的变化,也有模型本身的变化,只能当作趋势看。")

    L.append("\n## 6. multi 的标注情况(真实数据里 multi 有多少)\n")
    multi = result["multi"]
    names = multi[keys[0]]["labelers"]
    L.append("| 口径 | " + " | ".join(f"{nm} 判 multi" for nm in names)
             + " | 两路都判 multi | 一路 multi 一路 single | 按来源拆 |\n"
             + "| --- |" + " --- |" * (3 + len(names)))
    for k in keys:
        m = multi[k]
        src = "、".join(f"{s} {c}" for s, c in sorted(m["multi_vs_single"]["by_source"].items())) or "无"
        L.append(f"| {labels[k]} | " + " | ".join(str(m[f"{nm}_multi"]) for nm in names)
                 + f" | {m['both_multi']} | {m['multi_vs_single']['total']} | {src} |")
    L.append("\n这一节只有真实数据自己的答案,没有掺合成数据:multi 少就是少。"
             "合成训练集里 multi 是主力之一,所以第五节里能用上 multi 的数字,"
             "并不能证明真实请求上也判得出来。")
    return "\n".join(L) + "\n"
