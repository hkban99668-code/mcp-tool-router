"""离线对比:把"本地结果 + 云端答案 + 组合策略"的所有组合算成一张表。

流程上刻意分成两步,因为联网又慢又花钱:
1. `fetch_answers` 每个(客户端, 说明版本, 数据集)只问一次,落盘到 `.cache/answers/`,按 text 去重;
2. 所有策略、触发方式都从这份缓存里算,不再联网。

gold 只用于报告:策略与说明版本的选择看 train(`--dataset train`,本地用 5 折折外预测)。
"""

import json
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import evaluate as ev
from .jev import (PaymentRequired, RateLimited, answers_to_labels, make_client, redact,
                 request_questions)
from .strategy import STRATEGIES, TRIGGERS, apply_strategy

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "answers"    # 已在 .gitignore 里

# 需要 MCP 召回率不低于"本地基线 − 2 个百分点"的行会被标出来
RECALL_TOLERANCE = 0.02

# 并发取答案时写缓存要加锁:多线程同时追加会把行写坏或写重
_CACHE_LOCK = threading.Lock()


def cache_path(client_name, need_version, dataset) -> pathlib.Path:
    # 指定模型的 codebuddy 缓存名是 codebuddy@<模型>;文件名里的 @ 换成 _at_
    return CACHE_DIR / f"{client_name.replace('@', '_at_')}-{need_version}-{dataset}.jsonl"


def _read_cache(path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["text"]] = rec
    return out


def _append_cache(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class FetchStats:
    """一次联网取答案的统计:问了多少条、失败多少条、第一条错误、花了多少钱。

    T-005 里 Haiku 返回 HTTP 402 被整批吞掉、报告只显示"调用 0%",
    所以失败次数和第一条(脱敏后的)错误必须单独记下来并在终端和报告里都出现。
    """

    def __init__(self):
        self.asked = 0
        self.failed = 0
        self.first_error = ""
        self.cost = 0.0
        self.stopped = ""

    def add(self, rec) -> "FetchStats":
        self.asked += 1
        self.cost += float(rec.get("cost") or 0.0)
        if not rec.get("ok"):
            self.failed += 1
            if not self.first_error:
                self.first_error = rec.get("error") or "未知错误"
        return self

    def line(self) -> str:
        """终端与报告共用的那一行;没失败就是空串。"""
        return f"失败 {self.failed} 条,第一条错误:{self.first_error}" if self.failed else ""

    def as_dict(self) -> dict:
        return {"asked": self.asked, "failed": self.failed, "first_error": self.first_error,
                "cost_usd": float(self.cost), "stopped": self.stopped, "line": self.line()}


def _ask(client, text: str, need_version: str, questions=None) -> dict:
    """问一次,返回一条记录。失败不抛,记 ok=False 与脱敏后的错误。

    questions 给 None 时用 `request_questions(need_version)`;
    翻译那一步传自己的问题集,走的是同一套 HTTP / 重试 / 脱敏 / 计费。
    """
    t0 = time.perf_counter()
    try:
        answers = client.decide(text, questions or request_questions(need_version))
    except (RateLimited, PaymentRequired):
        raise                      # 限流 / 欠费不算普通失败,让 _ask_many 立刻停
    except Exception as e:
        return {"text": text, "answers": None, "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                "ok": False,
                "error": redact(f"{type(e).__name__}: {e}", getattr(client, "api_key", ""))}
    call = client.calls[-1] if client.calls else None
    return {"text": text, "answers": answers,
            "latency_ms": call.latency_ms if call else (time.perf_counter() - t0) * 1000.0,
            "cost": call.cost if call else 0.0,
            # T-017:token 用量一并写进缓存,给国产模型的费用估算用(缺失记 0)
            "prompt_tokens": getattr(call, "prompt_tokens", 0) if call else 0,
            "completion_tokens": getattr(call, "completion_tokens", 0) if call else 0,
            "reasoning_tokens": getattr(call, "reasoning_tokens", 0) if call else 0,
            "ok": True, "error": ""}


def _ask_batch(client, texts, need_version, questions) -> list:
    """批量客户端(有 `decide_batch`)一次问一块,返回与 texts 等长同序的记录。

    整批失败时每条都记失败;块内某条解析不出来时只有那一条 ok=False。
    延迟按块的总耗时除以条数,和客户端里 `calls` 的记法一致。
    """
    t0 = time.perf_counter()
    try:
        got = client.decide_batch(texts, questions or request_questions(need_version))
    except (RateLimited, PaymentRequired):
        raise                      # 限流 / 欠费不算普通失败,让 _ask_many 立刻停
    except Exception as e:      # 整批超时 / 退出码非 0 / 输出不是 JSON
        err = redact(f"{type(e).__name__}: {e}", getattr(client, "api_key", ""))
        per = (time.perf_counter() - t0) * 1000.0 / max(1, len(texts))
        return [{"text": t, "answers": None, "latency_ms": per, "cost": 0.0, "ok": False,
                 "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                 "error": err} for t in texts]
    per = (time.perf_counter() - t0) * 1000.0 / max(1, len(texts))
    return [{"text": t, "answers": ans, "latency_ms": per, "cost": 0.0,
             "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
             "ok": ans is not None, "error": "" if ans is not None else "这条没解析出答案"}
            for t, ans in zip(texts, got)]


def _chunks(texts, size) -> list:
    """按 batch_size 切成块;块就是一次调用要问的那些文本。"""
    size = max(1, int(size))
    return [texts[s:s + size] for s in range(0, len(texts), size)]


def _ask_job(client, texts, need_version, questions) -> list:
    """一个线程池任务:批量客户端问一块,其他客户端问一条;统一返回记录列表。"""
    if hasattr(client, "decide_batch"):
        return _ask_batch(client, texts, need_version, questions)
    return [_ask(client, texts[0], need_version, questions)]


def _ask_many(client, texts, need_version, questions, workers, max_consecutive_failures,
              stats, path, cache):
    """并发地问,边问边把成功的记录写进缓存(加锁);连续失败到上限就停。

    停下来时已经拿到的答案都留在缓存里,下次重跑不会重复花这份钱。

    连续失败的计数单位按客户端分两种(T-008):
    - 批量客户端(有 `decide_batch`,见 `codebuddy_client.py`):整批都失败才算一次,
      批内部分条目失败不累加;默认连续 3 批停。
    - 其他客户端(jev / haiku)一条一次调用:按条计,默认连续 5 条停,行为不变。
    max_consecutive_failures 传 None 时用这两组默认。

    同一波里按**提交顺序**处理各批的结果(不是完成顺序),所以同一份失败序列
    两次跑会停在同一个批,计数不随线程调度漂移。
    限流 / 欠费(RateLimited / PaymentRequired)不进连续计数,看到就立刻停。
    """
    workers = max(1, int(workers))
    ex = ThreadPoolExecutor(max_workers=workers)
    batched = hasattr(client, "decide_batch")
    if max_consecutive_failures is None:
        max_consecutive_failures = 3 if batched else 5
    unit = "批" if batched else "条"
    consecutive = 0
    try:
        if batched:
            jobs = _chunks(texts, getattr(client, "batch_size", 20))
        else:
            jobs = [[t] for t in texts]
        # 一批一批地提交,而不是一次把全部压进队列:连续失败要停下时,
        # 最多多问一批(workers 个任务),剩下的根本不会发出去。
        for start in range(0, len(jobs), workers):
            futures = [ex.submit(_ask_job, client, job, need_version, questions)
                       for job in jobs[start:start + workers]]
            for fu in futures:              # 按提交顺序取结果:停在哪一批是确定的
                try:
                    recs = fu.result()
                except (RateLimited, PaymentRequired) as e:
                    stats.stopped = _fatal_stop_line(e)
                    return stats
                ok_any = False
                for rec in recs:
                    stats.add(rec)
                    if rec["ok"]:
                        ok_any = True
                        with _CACHE_LOCK:
                            cache[rec["text"]] = rec
                            _append_cache(path, [rec])
                if ok_any or not recs:
                    consecutive = 0         # 批内有一条成功就不算整批失败
                elif max_consecutive_failures:
                    consecutive += 1
                    if consecutive >= max_consecutive_failures:
                        stats.stopped = (f"连续 {consecutive} {unit}调用失败,提前停下;"
                                         f"第一条错误:{stats.first_error}")
                        return stats
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return stats


def _fatal_stop_line(e) -> str:
    """限流 / 欠费时 stats.stopped 的内容;错误原文里带着重置时间或状态码。"""
    if isinstance(e, PaymentRequired):
        return f"OpenRouter 余额不足(402),立刻停下;{e}"
    return f"限流,立刻停下;{e}"


def _missing_record(text: str) -> dict:
    """没问(离线)或中途停下时占位用的记录。"""
    return {"text": text, "answers": None, "latency_ms": 0.0, "cost": 0.0, "ok": False,
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "error": ""}


def fetch_answers(client, client_name, items, need_version, dataset, live=False, questions=None,
                  workers=1, max_consecutive_failures=None, stats=None) -> list:
    """每个(客户端, 说明版本, 数据集)一份缓存,按 text 去重。

    live=False 时只读缓存:缓存里没有的条目记 answers=None, ok=False,不联网。
    只把成功的答案写进缓存,失败的下次再问。
    返回与 items 等长、同序的记录列表;统计写进 stats(见 `FetchStats`)。
    """
    path = cache_path(client_name, need_version, dataset)
    cache = _read_cache(path)
    stats = stats if stats is not None else FetchStats()
    todo = []
    for item in items:
        text = item["text"]
        if text not in cache:
            cache[text] = None                  # 占位:同一批里重复的文本不再问第二次
            if live:
                todo.append(text)
    if todo:
        _ask_many(client, todo, need_version, questions, workers, max_consecutive_failures,
                  stats, path, cache)
        if stats.line():
            print(stats.line())
    return [cache.get(item["text"]) or _missing_record(item["text"]) for item in items]


# 与 `evaluate` 的 gold 主口径一致:默认去掉 13 个"训练集 0 次"的词(见 rules.GOLD_ONLY_KEYWORDS),
# 这样基线行的数字可以和 reports/eval_report.md 的 gold 主数字对齐(82.6% 而不是含词的 92.8%)。
MAIN_NEED_FN = ev.request_need_features_ablated


def local_with_latency(dataset, tools, train, gold, encoder="tfidf", folds=5, random_state=0,
                       need_feature_fn=MAIN_NEED_FN):
    """返回 (逐条 HeadOutput, 逐条本地延迟 ms),顺序与数据集一致。

    gold:用全部训练集训练后预测;train:5 折分层交叉验证的折外预测。
    逐条预测是为了拿到单条延迟,端到端延迟要按条加客户端延迟。
    """
    items = gold if dataset == "gold" else train
    texts = [r["text"] for r in items]
    outs = [None] * len(texts)
    lats = [0.0] * len(texts)

    def run(head, idx, text):
        t0 = time.perf_counter()
        outs[idx] = head.predict([text])[0]
        lats[idx] = (time.perf_counter() - t0) * 1000.0

    if dataset == "gold":
        head = ev.train_request_head(ev.fit_encoder(encoder, tools, train), train,
                                     need_feature_fn=need_feature_fn)
        for i, text in enumerate(texts):
            run(head, i, text)
    else:
        for tr_idx, te_idx in ev.cv_splits(train, folds, random_state):
            fold_train = [train[i] for i in tr_idx]
            head = ev.train_request_head(ev.fit_encoder(encoder, tools, fold_train), fold_train,
                                         need_feature_fn=need_feature_fn)
            for i in te_idx:
                run(head, i, train[i]["text"])
    return outs, lats


def local_outputs(dataset, tools, train, gold, encoder="tfidf", need_feature_fn=MAIN_NEED_FN) -> list:
    return local_with_latency(dataset, tools, train, gold, encoder,
                              need_feature_fn=need_feature_fn)[0]


def _metrics(items, preds) -> dict:
    """preds: [(need, domains, actions)];口径就是 `evaluate.request_metrics`。"""
    return ev.request_metrics(items, preds)


def _row(items, outs, lats, fetched, client, need_version, strategy, trigger, threshold,
         with_preds=False) -> dict:
    """算一个组合:逐条套用 apply_strategy,再汇总指标、调用比例、延迟与费用。"""
    preds, called, used, cost, e2e = [], 0, 0, 0.0, []
    for i, out in enumerate(outs):
        rec = fetched[i]
        triggered = trigger == "always" or out.confidence < threshold
        # strategy=local 恒用本地结果,不需要问
        want = triggered and strategy != "local"
        labels = None
        if want and rec["ok"] and rec["answers"] is not None:
            labels = answers_to_labels(rec["answers"])
            called += 1
            cost += rec["cost"]
        merged = apply_strategy(out, labels, strategy)
        preds.append((merged["need"], merged["domains"], merged["actions"]))
        used += int(merged["used"])
        e2e.append(lats[i] + (rec["latency_ms"] if want and rec["ok"] else 0.0))
    n = len(items) or 1
    row = {"client": client, "need_version": need_version, "strategy": strategy, "trigger": trigger,
           "called_rate": called / n, "used_rate": used / n, "cost_usd": float(cost),
           "latency_ms": {"p50": float(np.percentile(e2e, 50)), "p95": float(np.percentile(e2e, 95))}}
    row.update(_metrics(items, preds))
    if with_preds:
        row["preds"] = [{"need": p[0], "domains": p[1], "actions": p[2]} for p in preds]
    return row


def compare(dataset, clients, need_versions, live=False, threshold=0.55,
            tools=None, train=None, gold=None, encoder="tfidf", make=make_client,
            with_preds=False, need_feature_fn=MAIN_NEED_FN) -> dict:
    """跑全组合对比。data 缺省时用 `data/` 下的种子数据。

    clients 是客户端名字(jev / haiku);live=False 时只读缓存、不联网,缺答案的组合退回本地结果。
    """
    tools = ev.load_jsonl(DATA / "tools.jsonl") if tools is None else tools
    train = ev.load_jsonl(DATA / "requests_train.jsonl") if train is None else train
    gold = ev.load_jsonl(DATA / "requests_gold.jsonl") if gold is None else gold
    items = gold if dataset == "gold" else train
    if not items:
        raise ValueError(f"数据集 {dataset!r} 是空的")
    outs, lats = local_with_latency(dataset, tools, train, gold, encoder,
                                    need_feature_fn=need_feature_fn)
    none_fetched = [{"answers": None, "latency_ms": 0.0, "cost": 0.0, "ok": False}] * len(items)

    baseline = _row(items, outs, lats, none_fetched, "-", "-", "local", "low_conf", threshold,
                    with_preds)
    rows = [baseline]
    fetch_stats = FetchStats()
    for name in clients:
        client = make(name) if live else None
        # 指定了模型的 codebuddy 缓存名是 codebuddy@<模型>,别和默认模型的混在一个文件里
        cache_name = getattr(client, "cache_name", name)
        for need_version in need_versions:
            fetched = fetch_answers(client, cache_name, items, need_version, dataset, live,
                                    stats=fetch_stats)
            for strategy in STRATEGIES:
                for trigger in TRIGGERS:
                    rows.append(_row(items, outs, lats, fetched, name, need_version, strategy,
                                     trigger, threshold, with_preds))
    return {"dataset": dataset, "n": len(items), "threshold": threshold, "live": bool(live),
            "clients": list(clients), "need_versions": list(need_versions),
            "baseline": baseline, "rows": rows, "fetch": fetch_stats.as_dict()}


def _pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def render_compare_report(result) -> str:
    """按 need 准确率降序排一张全组合表;召回没掉过头的行打 *。"""
    baseline = result["baseline"]
    floor = baseline["recall_needs_mcp"] - RECALL_TOLERANCE
    rows = sorted(result["rows"], key=lambda r: -r["need_accuracy"])
    L = ["# 兜底组合策略对比\n",
         f"数据集:`{result['dataset']}`({result['n']} 条);置信度阈值 {result['threshold']};"
         f"客户端 {', '.join(result['clients'])};need 说明版本 {', '.join(result['need_versions'])};"
         f"{'联网取答案' if result['live'] else '只读缓存(缺答案的条目退回本地结果)'}。\n"]
    if result.get("fetch", {}).get("failed"):
        L.append("> 取答案时" + result["fetch"]["line"] + "。这些条目按没问到处理,退回本地结果。\n")
    L += ["> gold 只用于报告;挑策略和说明版本看 `--dataset train`(本地用 5 折折外预测)。\n",
         "> `strategy=local` 的行不调客户端,数字与纯本地基线相同,重复出现是为了让"
         "(客户端 × 说明版本 × 策略 × 触发)的网格完整。\n",
         "\n| 客户端 | 说明 | 策略 | 触发 | need 准确率 | need macro-F1 | 需要 MCP 召回 | "
         "none 判对 | 能力域 micro/macro | 动作 micro/macro | 调用比例 | 改写比例 | "
         "延迟 p50/p95 (ms) | 费用(美元) |\n| --- | --- | --- | --- | --- | --- | --- | --- | "
         "--- | --- | --- | --- | --- | --- |"]
    for r in rows:
        mark = " *" if r["recall_needs_mcp"] >= floor else ""
        tag = " (基线)" if r is baseline else ""
        L.append(
            f"| {r['client']}{tag} | {r['need_version']} | {r['strategy']} | {r['trigger']} "
            f"| {_pct(r['need_accuracy'])}{mark} | {_pct(r['need_macro_f1'])} "
            f"| {_pct(r['recall_needs_mcp'])} | {r['none_correct']}/{r['none_total']} "
            f"| {_pct(r['domain']['micro_f1'])} / {_pct(r['domain']['macro_f1'])} "
            f"| {_pct(r['action']['micro_f1'])} / {_pct(r['action']['macro_f1'])} "
            f"| {_pct(r['called_rate'])} | {_pct(r['used_rate'])} "
            f"| {r['latency_ms']['p50']:.1f} / {r['latency_ms']['p95']:.1f} "
            f"| {r['cost_usd']:.6f} |")
    L.append(f"\n`*` = 需要 MCP 召回率不低于本地基线 − {RECALL_TOLERANCE * 100:.0f} 个百分点"
             f"(基线 {_pct(baseline['recall_needs_mcp'])} → 门槛 {_pct(floor)})。"
             f"延迟是本地延迟加被触发条的客户端延迟,费用只算被触发的条目。")
    return "\n".join(L) + "\n"
