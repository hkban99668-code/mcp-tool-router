"""下载三个公开数据集到 data/raw/,并写 data/SOURCES.md。

只下载公开数据集,不需要任何 key。每个文件记 URL、许可、sha256、下载日期、署名,
这样以后换机器或别人复现时能核对拿到的到底是不是同一份文件。

python scripts/fetch_public.py            # 缺哪个下哪个,已存在的跳过
python scripts/fetch_public.py --force    # 全部重下
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SOURCES = ROOT / "data" / "SOURCES.md"

# 每个条目:(来源, 保存的相对路径, 下载 URL, 许可, 署名)
# 三个数据集都是 Claude 在 2026-10-09 确认过许可与可下载性的,不要随意换成别的来源。
FILES = (
    ("bfcl", "bfcl/BFCL_v3_live_simple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_live_simple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/BFCL_v3_live_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_live_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/BFCL_v3_live_parallel.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_live_parallel.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/BFCL_v3_live_parallel_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_live_parallel_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/BFCL_v3_simple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_simple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/BFCL_v3_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/BFCL_v3_parallel.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_parallel.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/BFCL_v3_parallel_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/BFCL_v3_parallel_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    # 上面每个问题文件的标准答案(BFCL 叫 possible_answer),T-010 用它做"请求→工具"的金标。
    # 目录与命名跟上游一致:bfcl/possible_answer/<与问题文件同名>。
    ("bfcl", "bfcl/possible_answer/BFCL_v3_live_simple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_live_simple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/possible_answer/BFCL_v3_live_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_live_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/possible_answer/BFCL_v3_live_parallel.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_live_parallel.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/possible_answer/BFCL_v3_live_parallel_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_live_parallel_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/possible_answer/BFCL_v3_simple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_simple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/possible_answer/BFCL_v3_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/possible_answer/BFCL_v3_parallel.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_parallel.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("bfcl", "bfcl/possible_answer/BFCL_v3_parallel_multiple.json",
     "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/"
     "resolve/main/possible_answer/BFCL_v3_parallel_multiple.json",
     "Apache-2.0",
     "Berkeley Function Calling Leaderboard (BFCL) v3, gorilla-llm"),
    ("apibank", "apibank/lv1-train.json",
     "https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/main/training-data/lv1-train.json",
     "MIT", "API-Bank, liminghao1630 (原作者 Minghao Li 等)"),
    ("apibank", "apibank/lv2-train.json",
     "https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/main/training-data/lv2-train.json",
     "MIT", "API-Bank, liminghao1630 (原作者 Minghao Li 等)"),
    ("alpaca_zh", "alpaca_zh/alpaca_gpt4_data_zh.json",
     "https://huggingface.co/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json",
     "CC-BY-4.0", "alpaca-zh, shibing624(由 GPT-4 生成的中文指令数据)"),
)

TIMEOUT = 120.0
RETRIES = 2


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: pathlib.Path) -> None:
    """下载一个文件;失败重试 RETRIES 次,最后一次的异常抛上去。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                data = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as e:      # 超时、连接错误、HTTP 非 200
            last = e
            if attempt < RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}:{type(last).__name__}: {last}")


def count_records(path) -> int:
    """这个文件的样本条数:BFCL 是 JSONL(数行),API-Bank 与 alpaca 是 JSON 数组(数元素)。"""
    with open(path, encoding="utf-8") as f:
        if f.read(1) == "[":
            f.seek(0)
            return len(json.load(f))
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def render_sources(rows) -> str:
    L = ["# 公开数据来源\n",
         "由 `scripts/fetch_public.py` 下载到 `data/raw/`(该目录不进版本库)。"
         "每行给出下载 URL、许可、文件 sha256、下载日期与署名,"
         "换机器重跑后 sha256 应当完全一致。\n",
         "\n| 来源 | 文件 | 许可 | 行数 | 大小(字节) | sha256 | 下载日期 |\n"
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['source']} | `{r['path']}` | {r['license']} | {r['lines']} "
                 f"| {r['size']} | `{r['sha256'][:16]}…` | {r['date']} |")
    L.append("\n## 署名\n")
    seen = []
    for r in rows:
        if r["attribution"] not in seen:
            seen.append(r["attribution"])
    for a in seen:
        L.append(f"- {a}")
    L.append("\n## URL 清单\n")
    for r in rows:
        L.append(f"- `{r['path']}` ← {r['url']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="下载公开数据集到 data/raw/")
    p.add_argument("--force", action="store_true", help="已存在的文件也重新下载")
    p.add_argument("--raw", default=str(RAW), help="下载目录")
    a = p.parse_args(argv)

    raw = pathlib.Path(a.raw)
    rows, failed = [], []
    for source, rel, url, license_, attribution in FILES:
        dest = raw / rel
        if dest.exists() and not a.force:
            print(f"已存在,跳过:{rel}")
        else:
            print(f"下载中:{url}")
            try:
                download(url, dest)
            except Exception as e:
                print(f"失败:{e}", file=sys.stderr)
                failed.append(rel)
                continue
        rows.append({"source": source, "path": rel, "url": url, "license": license_,
                     "attribution": attribution, "sha256": sha256_of(dest),
                     "size": dest.stat().st_size, "lines": count_records(dest),
                     "date": time.strftime("%Y-%m-%d")})
    if failed:
        print(f"有 {len(failed)} 个文件没下下来:{', '.join(failed)}", file=sys.stderr)
        return 1
    SOURCES.write_text(render_sources(rows), encoding="utf-8")
    print(f"共 {len(rows)} 个文件 -> {raw};来源清单 {SOURCES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
