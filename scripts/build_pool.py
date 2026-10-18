"""从 data/raw/ 里取样、清洗、去重,生成 data/pool.jsonl(固定种子,可复现)。

三个来源各按配额取:
- BFCL 600(live 与非 live 各半,四类尽量均匀)
- API-Bank 300(lv1 / lv2 各半)
- alpaca-zh 500

每行字段:`id / text_orig / lang / source / source_file / hint`。
`hint` 只用于报告分析,不当标签用;BFCL 的 hint 里多一个 `math_like`,用来把"算学/计算类"
请求单独分出来报数字(见 reviews/T-006.md 开放问题 4)。

python scripts/build_pool.py
"""

import argparse
import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
POOL = ROOT / "data" / "pool.jsonl"

SEED = 0
QUOTA = {"bfcl": 600, "apibank": 300, "alpaca_zh": 500}
MIN_LEN, MAX_LEN = 4, 300
# 占位句:数据集里偶尔出现的"用户其实没说话",不该进池子
PLACEHOLDERS = ("the user did not provide", "no query provided", "the user did not specify")

_USER_LINE = re.compile(r"(?m)^User:[ \t]*(.*)$")
_API_REQ_LINE = re.compile(r"(?m)^API-Request:")

# 算学 / 计算类:函数名或描述里出现这些词就打上 hint.math_like,只用于分开报数字
_MATH_LIKE = re.compile(
    r"area|calculate|calc|math|equation|solve|convert|derivative|integral|factorial"
    r"|prime|gcd|lcm|sqrt", re.IGNORECASE)

BFCL_FILES = (
    ("BFCL_v3_live_simple.json", "live", "simple"),
    ("BFCL_v3_live_multiple.json", "live", "multiple"),
    ("BFCL_v3_live_parallel.json", "live", "parallel"),
    ("BFCL_v3_live_parallel_multiple.json", "live", "parallel_multiple"),
    ("BFCL_v3_simple.json", "base", "simple"),
    ("BFCL_v3_multiple.json", "base", "multiple"),
    ("BFCL_v3_parallel.json", "base", "parallel"),
    ("BFCL_v3_parallel_multiple.json", "base", "parallel_multiple"),
)


def norm_key(text: str) -> str:
    """去重用的规范化文本:去掉所有空白并小写。"""
    return re.sub(r"\s+", "", text).lower()


def is_english(text: str) -> bool:
    """ASCII 字母占非空白字符超过 60% 就算英文(中文里的地名不影响整句判断)。"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    ascii_letters = sum(1 for c in chars if c.isascii() and c.isalpha())
    return ascii_letters / len(chars) > 0.6


def math_like(funcs) -> bool:
    """算学 / 计算类:函数名或描述里出现 _MATH_LIKE 里的词就算。"""
    text = " ".join(f"{f.get('name') or ''} {f.get('description') or ''}"
                    for f in (funcs or []) if isinstance(f, dict))
    return bool(_MATH_LIKE.search(text))


def first_user_content(question) -> str:
    """BFCL 的 question 是多轮:取第一条 role=user 的 content。"""
    for turn in (question or []):
        for msg in turn:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content") or "")
    return ""


def first_user_line(input_text: str) -> str:
    """API-Bank:取对话里第一条 `User:` 后面的话。

    调查结论(见 reports/T-006.md):20 条抽样里第一条 User 全部能看出要做什么(20/20),
    而"紧挨第一个 API-Request 之前那条"有 2 条取不到、1 条丢了上下文(17/20),所以取第一条。
    """
    m = _USER_LINE.search(input_text or "")
    return m.group(1).strip() if m else ""


def iter_bfcl(raw):
    """BFCL:每个文件一个 JSONL,取 question 里第一条 user 消息。"""
    for name, split, category in BFCL_FILES:
        path = raw / "bfcl" / name
        if not path.exists():
            print(f"缺少文件,跳过:{path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for row in (json.loads(l) for l in f if l.strip()):
                text = first_user_content(row.get("question"))
                funcs = row.get("function") or []
                yield {"text_orig": text, "source": "bfcl", "source_file": name,
                       "split": split, "category": category,
                       "hint": {"n_functions": len(funcs), "category": category,
                                "math_like": math_like(funcs)}}


def iter_apibank(raw):
    for name, level in (("lv1-train.json", 1), ("lv2-train.json", 2)):
        path = raw / "apibank" / name
        if not path.exists():
            print(f"缺少文件,跳过:{path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            text = first_user_line(row.get("input") or "")
            yield {"text_orig": text, "source": "apibank", "source_file": name,
                   "level": level, "hint": {"level": level}}


def iter_alpaca(raw):
    path = raw / "alpaca_zh" / "alpaca_gpt4_data_zh.json"
    if not path.exists():
        print(f"缺少文件,跳过:{path}", file=sys.stderr)
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        text = (row.get("instruction") or "").strip()
        extra = (row.get("input") or "").strip()
        if extra:
            text = f"{text}\n{extra}"
        yield {"text_orig": text, "source": "alpaca_zh", "source_file": "alpaca_gpt4_data_zh.json",
               "hint": {}}


def clean(rows):
    """去掉空串、太短太长、占位句;按规范化文本去重(保留第一次出现的)。"""
    out, seen = [], set()
    for r in rows:
        text = (r["text_orig"] or "").strip()
        if not text or len(text) < MIN_LEN or len(text) > MAX_LEN:
            continue
        low = text.lower()
        if any(p in low for p in PLACEHOLDERS):
            continue
        key = norm_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(r, text_orig=text))
    return out


def take(rnd, rows, k):
    """随机取 k 条;不足就全给。"""
    pool = list(rows)
    rnd.shuffle(pool)
    return pool[:k]


def take_by_group(rnd, rows, k, group_of):
    """按组尽量均匀地取 k 条:各组先各自打乱,再轮转着一条一条取。"""
    groups = {}
    for r in rows:
        groups.setdefault(group_of(r), []).append(r)
    for g in groups.values():
        rnd.shuffle(g)
    order = sorted(groups)
    out, i = [], 0
    while len(out) < k:
        added = False
        for name in order:
            if i < len(groups[name]) and len(out) < k:
                out.append(groups[name][i])
                added = True
        if not added:
            break                      # 所有组都取完了
        i += 1
    return out


def sample(rnd, rows):
    """按配额取样:BFCL live/非 live 各半且四类均匀,API-Bank lv1/lv2 各半,alpaca 直接取。"""
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    out = []
    bfcl = by_source.get("bfcl", [])
    for split in ("live", "base"):
        part = [r for r in bfcl if r["split"] == split]
        out += take_by_group(rnd, part, QUOTA["bfcl"] // 2, lambda r: r["category"])
    out += take_by_group(rnd, by_source.get("apibank", []), QUOTA["apibank"],
                         lambda r: r["level"])
    out += take(rnd, by_source.get("alpaca_zh", []), QUOTA["alpaca_zh"])
    return out


def build(raw):
    """取样并返回 (池子, 各来源清洗前后计数)。"""
    rnd = random.Random(SEED)
    rows, stats = [], {}
    for name, it in (("bfcl", iter_bfcl), ("apibank", iter_apibank), ("alpaca_zh", iter_alpaca)):
        raw_rows = list(it(raw))
        kept = clean(raw_rows)
        stats[name] = {"raw": len(raw_rows), "kept": len(kept),
                       "dropped": len(raw_rows) - len(kept)}
        # id 用"来源内序号":文件顺序固定,所以同一份 raw 跑出来的 id 永远一样
        for i, r in enumerate(kept):
            r["id"] = f"{name}-{i}"
            r["lang"] = "en" if is_english(r["text_orig"]) else "zh"
        rows += kept
    picked = sample(rnd, rows)
    out = []
    for r in picked:
        out.append({"id": r["id"], "text_orig": r["text_orig"], "lang": r["lang"],
                    "source": r["source"], "source_file": r["source_file"],
                    "hint": r["hint"]})
    return out, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="从 data/raw/ 生成 data/pool.jsonl")
    p.add_argument("--raw", default=str(RAW))
    p.add_argument("--out", default=str(POOL))
    a = p.parse_args(argv)

    pool, stats = build(pathlib.Path(a.raw))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, s in stats.items():
        n = sum(1 for r in pool if r["source"] == name)
        print(f"{name}: 原始 {s['raw']} 条 -> 清洗去重后 {s['kept']} 条"
              f"(去掉 {s['dropped']} 条) -> 取样 {n} 条")
    print(f"共 {len(pool)} 条 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
