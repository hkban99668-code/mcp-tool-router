"""从 data/raw/bfcl/ 抽真实工具目录,并给池子里的 BFCL 请求配上"该调哪个工具"(T-010 A 部分)。

全程离线,不调任何模型。产出三样东西:
- `data/tools_real.jsonl`:从 BFCL 每条请求自带的 `function` 定义里抽出的真实工具目录(去重后)。
- `data/requests_real_tool_ids.jsonl`:池子里每条 BFCL 请求 → 应调用的 tool_id(来自 possible_answer)
  和它自带的候选集 tool_id(以后算"候选集内 Recall@k"用)。
- `reports/real_tools_stats.md`:统计报告。

请求 id(`bfcl-<n>`)是 `scripts/build_pool.py` 里"清洗去重后按文件顺序编号"得到的,
所以这里复用 build_pool 的清洗规则把同一张表重跑一遍,才能按 n 对回原始 BFCL 记录。

python scripts/build_real_tools.py                 # 用默认路径全量生成
python scripts/build_real_tools.py --raw X --pool Y --out-dir Z   # 测试用,可指向临时目录
"""

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_pool                      # 复用建池子的清洗规则与文件顺序
import fetch_public                    # 复用 sha256_of 与来源(许可/署名)元数据

RAW = ROOT / "data" / "raw" / "bfcl"
POOL = ROOT / "data" / "pool.jsonl"
OUT_DIR = ROOT / "data"
STATS = ROOT / "reports" / "real_tools_stats.md"
SOURCES = ROOT / "data" / "SOURCES.md"

# 函数名里的 `_数字_` 分段:BFCL 的 Weather_1_GetWeather 这类,前面一段是命名空间
_UNDERSCORE_NS = re.compile(r"_\d+_")
# 只是粗看风险的计数,当统计用,不当标签(真正的风险分级在 B 部分)
_RISKY = re.compile(
    r"delete|remove|drop|send|post|pay|transfer|execute|run|create|update|write",
    re.IGNORECASE)

SOURCES_MARKER = "<!-- tool-catalog -->"


# ---------------------------------------------------------------- 规范化与推断

def norm_desc(text) -> str:
    """描述的规范化形式:压掉多余空白并小写。只用于去重,不写进输出。"""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def canon_params(params) -> str:
    """参数的规范化 JSON 串(键有序、紧凑分隔符)。只用于去重。"""
    return json.dumps(params or {}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def dedup_key(func) -> tuple:
    """工具的去重键:(函数名, 规范化描述, 规范化参数)。"""
    return (func.get("name") or "", norm_desc(func.get("description")),
            canon_params(func.get("parameters")))


def infer_server(name: str) -> str:
    """从函数名推断命名空间(server)。

    三条规则,按序:
    1. 名字里有 `.`:取最后一个 `.` 之前的部分(`requests.get` → `requests`,
       `acl_api.AclApi.retrieve_projects` → `acl_api.AclApi`);
    2. 否则有 `_数字_`:取第一个 `_数字_` 之前的部分(`Weather_1_GetWeather` → `Weather`);
    3. 都没有:`misc`。
    取出来是空串时按规则 3 处理,避免出现空 server。
    """
    if "." in name:
        server = name.rsplit(".", 1)[0]
        if server:
            return server
    m = _UNDERSCORE_NS.search(name)
    if m:
        server = name[:m.start()]
        if server:
            return server
    return "misc"


# ---------------------------------------------------------------- 读原始数据

def iter_records(raw):
    """按 build_pool 的文件顺序逐条产出 (来源文件名, 文件内序号, 原始记录)。"""
    for name, _split, _category in build_pool.BFCL_FILES:
        path = raw / name
        if not path.exists():
            continue                        # 缺哪个文件在 main 里一次性提示
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(l for l in f if l.strip()):
                yield name, i, json.loads(line)


def load_answers(raw, source_file):
    """读 possible_answer/<同名文件>,返回 (按 id 索引的 ground_truth, 按行序的名字列表)。

    BFCL 的 possible_answer 每行是 `{"id": ..., "ground_truth": [{函数名: 参数}, ...]}`;
    ground_truth 里每个元素只有一个键,那个键就是被调用的函数名。
    文件不存在时返回 (None, None),调用方据此跳过。
    """
    path = raw / "possible_answer" / source_file
    if not path.exists():
        return None, None
    by_id, ordered = {}, []
    with open(path, encoding="utf-8") as f:
        for line in (l for l in f if l.strip()):
            rec = json.loads(line)
            names = [next(iter(d)) for d in (rec.get("ground_truth") or [])
                     if isinstance(d, dict) and d]
            by_id[rec.get("id")] = names
            ordered.append(names)
    return by_id, ordered


# ---------------------------------------------------------------- 建目录

def build_catalog(raw):
    """扫全部问题文件,按 (名字, 描述, 参数) 去重,返回 (目录列表, 去重键→tool_id)。

    同名但描述或参数不同的当成不同工具,tool_id 依次加 `#2`、`#3`;
    后缀序号按 (描述, 参数) 排序决定,和文件遍历顺序无关,保证可复现。
    """
    variants = {}
    for source_file, _i, rec in iter_records(raw):
        for func in (rec.get("function") or []):
            key = dedup_key(func)
            got = variants.get(key)
            if got is None:
                variants[key] = {"name": func.get("name") or "",
                                 "description": func.get("description") or "",
                                 "parameters": func.get("parameters"),
                                 "source_files": set(), "n_requests": 0,
                                 "_key": key}
                got = variants[key]
            got["source_files"].add(source_file)

    by_name = collections.defaultdict(list)
    for key, v in variants.items():
        by_name[v["name"]].append(key)
    for name, keys in by_name.items():
        for i, key in enumerate(sorted(keys, key=lambda k: (k[1], k[2]))):
            v = variants[key]
            v["server"] = infer_server(name)
            v["tool_id"] = f"{v['server']}/{name}" + ("" if i == 0 else f"#{i + 1}")

    catalog = sorted(variants.values(), key=lambda v: v["tool_id"])
    key_to_id = {v["_key"]: v["tool_id"] for v in catalog}
    return catalog, key_to_id


def count_requests(raw, catalog, key_to_id):
    """给每条工具数一下:它作为候选出现在多少条 BFCL 记录里(写回 n_requests)。

    数的是 8 个问题文件里的全部记录,不是池子取样后的 600 条。
    """
    by_tool_id = {v["tool_id"]: v for v in catalog}
    for _source_file, _i, rec in iter_records(raw):
        seen = {key_to_id[dedup_key(f)] for f in (rec.get("function") or [])
                if dedup_key(f) in key_to_id}
        for tool_id in seen:
            by_tool_id[tool_id]["n_requests"] += 1


# ---------------------------------------------------------------- 请求→工具

def resolve_tool_ids(answer_names, request_funcs, key_to_id, name_to_ids):
    """把 possible_answer 里的函数名映射到 tool_id。

    优先用"这条请求自带的函数定义"精确匹配(同名不同描述的冲突由此消解);
    答案里的名字不在候选集里时,退而求其次:该名字在目录里只对应一个工具才采信。
    返回 (tool_id 列表, 映射失败的名字列表)。
    """
    cand_by_name = collections.defaultdict(list)
    for func in request_funcs:
        tid = key_to_id.get(dedup_key(func))
        if tid is not None:
            cand_by_name[func.get("name") or ""].append(tid)
    tool_ids, failed = [], []
    for name in answer_names:
        hits = cand_by_name.get(name)
        if not hits:
            # 答案里的名字不在本请求候选集里:目录里唯一同名才敢用
            hits = name_to_ids.get(name) or []
            if len(hits) != 1:
                failed.append(name)
                continue
        for tid in hits:
            if tid not in tool_ids:
                tool_ids.append(tid)
    return tool_ids, failed


def build_request_map(raw, pool_path, key_to_id, catalog):
    """池子里每条 BFCL 请求 → (应调 tool_ids, 候选 tool_ids, 答案来源)。"""
    name_to_ids = collections.defaultdict(list)
    for v in catalog:
        name_to_ids[v["name"]].append(v["tool_id"])

    # 和 build_pool 同一套顺序与清洗规则,才能按 bfcl-<n> 对回原记录
    rows = []
    for source_file, idx_in_file, rec in iter_records(raw):
        rows.append({"text_orig": build_pool.first_user_content(rec.get("question")),
                     "source_file": source_file, "idx_in_file": idx_in_file,
                     "orig_id": rec.get("id"),
                     "functions": rec.get("function") or []})
    kept = build_pool.clean(rows)
    file_sizes = collections.Counter(r["source_file"] for r in rows)

    pool = [json.loads(l) for l in open(pool_path, encoding="utf-8") if l.strip()]
    bfcl_rows = sorted((r for r in pool if r.get("source") == "bfcl"),
                       key=lambda r: int(str(r["id"]).split("-")[1]))

    answers_cache, out, skipped, failures = {}, [], [], []
    for r in bfcl_rows:
        rec = kept[int(str(r["id"]).split("-")[1])]
        source_file = rec["source_file"]
        if source_file not in answers_cache:
            answers_cache[source_file] = load_answers(raw, source_file)
        by_id, ordered = answers_cache[source_file]
        if by_id is None:
            skipped.append({"id": r["id"], "source_file": source_file})
            continue
        names = by_id.get(rec["orig_id"])
        if names is None and len(ordered) == file_sizes[source_file]:
            # 极少数 id 在问题文件与答案文件里不一致(实测 live_multiple 有 1 条),
            # 其余全部对得上时才敢按行号兜底
            names = ordered[rec["idx_in_file"]]
        if names is None:
            skipped.append({"id": r["id"], "source_file": source_file})
            continue
        tool_ids, failed = resolve_tool_ids(names, rec["functions"], key_to_id,
                                            name_to_ids)
        if failed:
            failures.append({"id": r["id"], "names": failed})
        candidate = []
        for func in rec["functions"]:
            tid = key_to_id.get(dedup_key(func))
            if tid is not None and tid not in candidate:
                candidate.append(tid)
        out.append({"id": r["id"], "tool_ids": tool_ids,
                    "candidate_tool_ids": candidate,
                    "answer_source": f"possible_answer/{source_file}"})
    return out, skipped, failures, len(bfcl_rows)


# ---------------------------------------------------------------- 统计

def percentile(sorted_vals, q):
    """线性插值的分位数;q 取 0..1。"""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def render_stats(catalog, requests, n_raw_functions, skipped, failures, n_bfcl):
    """按任务卡第 3 节的要求出统计报告。"""
    lens = sorted(len(v["description"]) for v in catalog)
    no_desc = sum(1 for v in catalog if not v["description"].strip())
    risky = [v for v in catalog if _RISKY.search(v["name"] or "")
             or _RISKY.search(v["description"] or "")]
    by_name = collections.defaultdict(list)
    for v in catalog:
        by_name[v["name"]].append(v)
    conflicts = {k: v for k, v in by_name.items() if len(v) > 1}

    by_file = collections.Counter()
    for v in catalog:
        for f in v["source_files"]:
            by_file[f] += 1
    by_server = collections.Counter(v["server"] for v in catalog)

    n_tools = len(requests)
    n_with = sum(1 for r in requests if r["tool_ids"])
    all_ids = [t for r in requests for t in r["tool_ids"]]
    cand_sizes = sorted(len(r["candidate_tool_ids"]) for r in requests)
    id_sizes = sorted(len(r["tool_ids"]) for r in requests)
    cand_hist = collections.Counter(cand_sizes)
    id_hist = collections.Counter(id_sizes)

    L = ["# 真实工具目录统计(T-010 A 部分)\n",
         "由 `scripts/build_real_tools.py` 生成,数据来自 `data/raw/bfcl/` 的 8 个问题文件"
         "与对应的 possible_answer。全部离线,没有调任何模型。\n",
         "\n## 1. 工具目录\n",
         f"- 函数出现总数(去重前):{n_raw_functions}",
         f"- 去重后工具数:**{len(catalog)}**",
         f"- 唯一函数名:{len(by_name)}",
         f"- 同名但描述或参数不同的名字数:**{len(conflicts)}**"
         f"(涉及 {sum(len(v) for v in conflicts.values())} 条工具记录)",
         f"- 没有描述的工具数:{no_desc}",
         "\n按来源文件(一个工具可能出现在多个文件里,所以这里相加大于总数):\n",
         "\n| 来源文件 | 工具数 |\n| --- | --- |"]
    for f, n in sorted(by_file.items(), key=lambda kv: (-kv[1], kv[0])):
        L.append(f"| `{f}` | {n} |")
    worst = sorted(conflicts.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
    if worst:
        L.append("\n同名冲突最多的 5 个名字:\n")
        L.append("\n| 函数名 | 变体数 | 其中一条描述 |\n| --- | --- | --- |")
        for k, v in worst:
            desc = sorted(x["description"] for x in v)[0].replace("|", "/")[:60]
            L.append(f"| `{k}` | {len(v)} | {desc} |")
    L.append("\n按推断的 server(前 20):\n")
    L.append("\n| server | 工具数 |\n| --- | --- |")
    for s, n in sorted(by_server.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        L.append(f"| `{s}` | {n} |")
    L.append(f"\n共 {len(by_server)} 个不同的 server。\n")

    L.append("\n## 2. 请求 → 工具对应关系\n")
    L.append(f"- 池子里 BFCL 请求:{n_bfcl} 条")
    L.append(f"- 有 tool_ids 的:{n_with} 条"
             + (f"(占 {n_with / n_bfcl * 100:.1f}%)" if n_bfcl else ""))
    L.append(f"- 跳过(没有 possible_answer 或 id 对不上):{len(skipped)} 条")
    L.append(f"- 答案函数名映射失败的:{len(failures)} 条")
    avg_ids = f"{len(all_ids) / n_tools:.2f}" if n_tools else "—"
    L.append(f"- tool_ids 总数:{len(all_ids)};每条请求平均 {avg_ids} 个")
    L.append("")
    L.append("| 每条请求的 tool_ids 个数 | 请求数 |\n| --- | --- |")
    for k in sorted(id_hist):
        L.append(f"| {k} | {id_hist[k]} |")
    L.append("")
    L.append("| 候选集大小 | 请求数 |\n| --- | --- |")
    for k in sorted(cand_hist):
        L.append(f"| {k} | {cand_hist[k]} |")
    if cand_sizes:
        L.append(f"\n候选集大小:最小 {cand_sizes[0]},中位数 "
                 f"{percentile(cand_sizes, 0.5):.0f},最大 {cand_sizes[-1]},"
                 f"平均 {sum(cand_sizes) / len(cand_sizes):.2f}。\n")
    if failures:
        L.append("\n映射失败的例子:")
        for f in failures[:5]:
            L.append(f"- `{f['id']}`:{', '.join(f['names'])}")

    L.append("\n## 3. 描述长度(字符数)\n")
    L.append("| 最小 | 25% | 中位数 | 75% | 最大 | 平均 |\n| --- | --- | --- | --- | --- | --- |")
    L.append(f"| {lens[0]} | {percentile(lens, 0.25):.0f} | {percentile(lens, 0.5):.0f} "
             f"| {percentile(lens, 0.75):.0f} | {lens[-1]} "
             f"| {sum(lens) / len(lens):.1f} |")
    buckets = [(0, 40), (40, 80), (80, 160), (160, 320), (320, 10 ** 9)]
    L.append("\n| 长度区间 | 工具数 |\n| --- | --- |")
    for lo, hi in buckets:
        n = sum(1 for x in lens if lo <= x < hi)
        label = f"{lo}–{hi - 1}" if hi < 10 ** 9 else f"≥{lo}"
        L.append(f"| {label} | {n} |")
    L.append(f"\n没有描述的工具:{no_desc} 个。\n")

    L.append("\n## 4. 粗看风险(只计数,不是标签)\n")
    rate = f"(占 {len(risky) / len(catalog) * 100:.1f}%)" if catalog else ""
    L.append("函数名或描述里出现 "
             "`delete|remove|drop|send|post|pay|transfer|execute|run|create|update|write` "
             f"的工具:**{len(risky)}** 个{rate}。")
    L.append("\n这个正则很宽(`run`/`post` 之类会误命中),只用来给 B 部分打标时对照,"
             "不要当风险标签用。\n")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- SOURCES

def tool_catalog_section(raw, out_paths):
    """给 data/SOURCES.md 补的"工具目录"一节:文件、许可、署名、sha256。"""
    L = [SOURCES_MARKER, "", "## 工具目录(T-010)\n",
         "本小节由 `scripts/build_real_tools.py` 维护(重跑 `scripts/fetch_public.py` "
         "会重写上面的表格,之后需要再跑一次本脚本把这一节补回来)。\n",
         "\n### 用到的 BFCL 文件\n",
         "\n| 文件 | 许可 | 行数 | 大小(字节) | sha256 |\n| --- | --- | --- | --- | --- |"]
    meta = {(src, rel): lic for src, rel, _u, lic, _attr in fetch_public.FILES}
    for name, _split, _cat in build_pool.BFCL_FILES:
        for rel, path in ((f"bfcl/{name}", raw / name),
                          (f"bfcl/possible_answer/{name}",
                           raw / "possible_answer" / name)):
            lic = meta.get(("bfcl", rel), "Apache-2.0")
            if not path.exists():
                L.append(f"| `{rel}` | {lic} | — | — | (未下载) |")
                continue
            lines = sum(1 for l in open(path, encoding="utf-8") if l.strip())
            L.append(f"| `{rel}` | {lic} | {lines} | {path.stat().st_size} "
                     f"| `{fetch_public.sha256_of(path)[:16]}…` |")
    L.append("\n### 派生文件\n")
    L.append("\n| 文件 | 条数 | 大小(字节) | sha256 |\n| --- | --- | --- | --- |")
    for path in out_paths:
        if not path.exists():
            continue
        lines = sum(1 for l in open(path, encoding="utf-8") if l.strip())
        L.append(f"| `{path.name}` | {lines} | {path.stat().st_size} "
                 f"| `{fetch_public.sha256_of(path)[:16]}…` |")
    L.append("\n### 署名与许可\n")
    L.append("- Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm — Apache-2.0"
             "(问题文件与 possible_answer 同一许可)")
    L.append("- 工具目录是 BFCL 每条请求自带函数定义的去重汇总,`inputSchema` 原样保留,"
             "没有改写内容。")
    return "\n".join(L) + "\n"


def write_sources_section(section, sources_path=SOURCES):
    """把"工具目录"一节追加到 SOURCES.md;已有就替换,保证幂等。"""
    text = sources_path.read_text(encoding="utf-8") if sources_path.exists() else ""
    head = text.split(SOURCES_MARKER)[0].rstrip() + "\n"
    sources_path.write_text(head + "\n" + section, encoding="utf-8")


# ---------------------------------------------------------------- 主流程

def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="从 BFCL 抽真实工具目录与请求→工具对应")
    p.add_argument("--raw", default=str(RAW))
    p.add_argument("--pool", default=str(POOL))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--stats", default=str(STATS))
    p.add_argument("--no-sources", action="store_true", help="不改 data/SOURCES.md")
    a = p.parse_args(argv)

    raw = pathlib.Path(a.raw)
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n, _s, _c in build_pool.BFCL_FILES if not (raw / n).exists()]
    if missing:
        print(f"缺少 {len(missing)} 个问题文件,跳过:{', '.join(missing)}", file=sys.stderr)

    catalog, key_to_id = build_catalog(raw)
    count_requests(raw, catalog, key_to_id)
    requests, skipped, failures, n_bfcl = build_request_map(
        raw, a.pool, key_to_id, catalog)

    n_raw_functions = sum(len(r.get("function") or [])
                          for _s, _i, r in iter_records(raw))

    tools_path = out_dir / "tools_real.jsonl"
    req_path = out_dir / "requests_real_tool_ids.jsonl"
    dump_jsonl(tools_path, [{"server": v["server"], "name": v["name"],
                             "tool_id": v["tool_id"], "description": v["description"],
                             "inputSchema": v["parameters"], "source": "bfcl",
                             "source_files": sorted(v["source_files"]),
                             "n_requests": v["n_requests"]} for v in catalog])
    dump_jsonl(req_path, requests)

    stats_path = pathlib.Path(a.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(render_stats(catalog, requests, n_raw_functions, skipped,
                                      failures, n_bfcl), encoding="utf-8")
    if not a.no_sources:
        write_sources_section(tool_catalog_section(raw, [tools_path, req_path]))

    print(f"工具目录 {len(catalog)} 条(去重前 {n_raw_functions} 次出现)-> {tools_path}")
    print(f"请求对应 {len(requests)} 条(池子里 BFCL 共 {n_bfcl} 条,"
          f"跳过 {len(skipped)},映射失败 {len(failures)}) -> {req_path}")
    print(f"统计 -> {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
