"""真实数据(T-006):翻译、双路打标、切分、金标候选表、重训评估。

四步串成一条流水线,每步都落盘、都走 `.cache/answers/` 的缓存,中断后重跑只补没做完的:

    translate   data/pool.jsonl          -> data/pool_zh.jsonl
    label       data/pool_zh.jsonl       -> data/labeled_agree.jsonl + data/review_queue.jsonl
    label --single                     -> data/labeled_single.jsonl(单路,不进双路数据)
    split       data/labeled_agree.jsonl -> data/real_train.jsonl + data/real_holdout.jsonl
    candidates  上面两份                 -> data/gold_candidates.csv

上面每个箭头右边的路径都是 CLI 参数(`--pool-zh` / `--agree` / `--review` / `--real-train` /
`--holdout` / `--candidates` / `--single-out`),所以换口径重新跑时把四份输出一起指到
`data/v3/` 就行,`data/` 下的旧标签一行都不会被覆盖(T-009 的 v3 口径就是这么跑的)。

钱和失败是这一卡的重点(见 T-005 审查):所有联网步骤共用一个累计费用
(`data/realdata_cost.json`,按步骤和客户端分列),下一次调用前发现累计已超预算就停(退出码 3);
某一路构造失败或连续失败(单条客户端连续 5 条、批量客户端连续 3 整批,见 `compare._ask_many`)
也立刻停(退出码 4)。限流 / 欠费(RateLimited / PaymentRequired)不等连续计数,直接停。
无论怎么停,已经问到的答案都留在缓存里,不会白花第二份钱。
"""

import csv
import json
import pathlib
import random
from collections import Counter

from sklearn.model_selection import train_test_split

from . import compare, evaluate as ev
from .jev import answers_to_labels, make_client, request_questions
from .taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS, category_cards

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
POOL = DATA / "pool.jsonl"
POOL_ZH = DATA / "pool_zh.jsonl"
AGREE = DATA / "labeled_agree.jsonl"
SINGLE = DATA / "labeled_single.jsonl"
REVIEW = DATA / "review_queue.jsonl"
REAL_TRAIN = DATA / "real_train.jsonl"
HOLDOUT = DATA / "real_holdout.jsonl"
CANDIDATES = DATA / "gold_candidates.csv"
CANDIDATES_README = DATA / "gold_candidates_README.md"
COST_FILE = DATA / "realdata_cost.json"

# 默认值:T-005 在 train 上选定 need 说明 v2;预算是所有联网步骤的总额。
# 连续失败的上限交给 compare._ask_many:单条客户端 5 条,批量客户端 3 整批。
NEED_VERSION_DEFAULT = "v2"
# T-010a:一致的判定有三种口径。`exact` 是 T-006 起的旧行为,留着做对照;
# `primary` 按第一路概率最高的那个域(主域)落没落在第二路的域集合里判;
# `primary_sym` 再放宽一层:第二路的主域落在第一路的域里也算一致。
AGREEMENT_RULES = ("exact", "primary", "primary_sym")
AGREEMENT_DEFAULT = "primary"
AGREEMENT_EXACT = "exact"
BUDGET_DEFAULT = 2.0
WORKERS_DEFAULT = 4
TRANSLATE_TAG = "zh"          # 翻译的缓存文件名中段(没有 need 版本这一说)
HOLDOUT_SIZE = 0.2
N_CANDIDATES = 200

EXIT_BUDGET = 3               # 累计费用已超预算
EXIT_CLIENT = 4               # 客户端构造失败 / 连续失败 / 限流 / 欠费

# 候选表列名:T-008 起两路不再写死 jev/haiku,叫 labeler1 / labeler2(第一路是基准路)
CAND_COLUMNS = ("id", "text", "text_orig", "source", "labeler1_name", "labeler1_need",
                "labeler1_domains", "labeler2_name", "labeler2_need", "labeler2_domains",
                "local_need", "local_domains", "need", "domains", "actions", "note")
MULTI_SEP = ";"


class Stopped(Exception):
    """联网步骤被叫停。exit_code 决定进程退出码(3 预算 / 4 客户端)。"""

    def __init__(self, reason: str, exit_code: int):
        super().__init__(reason)
        self.reason = reason
        self.exit_code = exit_code


# ---------------- 读写 ----------------

def write_jsonl(path, rows) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    return ev.load_jsonl(path)


def is_english(text: str) -> bool:
    """ASCII 字母占非空白字符超过 60% 算英文(与 `scripts/build_pool.py` 同一条规则)。"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    return sum(1 for c in chars if c.isascii() and c.isalpha()) / len(chars) > 0.6


# ---------------- 费用 ----------------

class CostBook:
    """所有联网步骤共用的累计费用,按步骤和客户端分列,落盘在 data/realdata_cost.json。"""

    def __init__(self, path=COST_FILE):
        self.path = pathlib.Path(path)
        self.steps = {}
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.steps = json.load(f).get("steps") or {}
            except (ValueError, OSError):
                self.steps = {}      # 文件坏了就当没花过钱,不要因此卡住整个流程

    def total(self) -> float:
        return sum(sum(v.values()) for v in self.steps.values())

    def add(self, step: str, client: str, usd: float) -> None:
        self.steps.setdefault(step, {})
        self.steps[step][client] = self.steps[step].get(client, 0.0) + float(usd)

    def check(self, budget) -> None:
        """下一次联网调用前查一次:累计已超预算就停,已完成的缓存保留。"""
        if budget is None:
            return
        if self.total() >= budget:
            raise Stopped(f"累计费用 {self.total():.4f} 美元已达到预算 {budget:.2f} 美元,"
                          f"停止联网步骤;已完成的答案都留在缓存里。", EXIT_BUDGET)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"total_usd": round(self.total(), 6), "steps": self.steps},
                      f, ensure_ascii=False, indent=2)


def open_client(name, timeout=30.0, model=None):
    """按名字造客户端;构造失败(没有 key、账号没钱、命令不在 PATH)立刻停,退出码 4。

    codebuddy 一次调用一批要几十秒,沿用 `--jev-timeout` 的 30 秒会把每批都打成超时,
    所以带 `min_timeout` 的客户端(批量)把超时抬到它自己的下限。
    model 只对 codebuddy 生效(`--codebuddy-model`),None 时用 CodeBuddy 默认模型。
    """
    try:
        kw = {"timeout": timeout}
        if name == "codebuddy":
            from .codebuddy_client import CodeBuddyClient
            kw["timeout"] = max(timeout, CodeBuddyClient.min_timeout)
            if model:
                kw["model"] = model
        return make_client(name, **kw)
    except Exception as e:
        raise Stopped(f"客户端 {name} 不可用:{type(e).__name__}: {e}", EXIT_CLIENT) from None


class _CacheOnlyClient:
    """只读缓存时的占位客户端:只带一个 cache_name 供 `fetch_answers` 定位缓存。

    `compare.fetch_answers(live=False)` 只用 `client.cache_name`,不碰连接;
    没有 key / 命令不在 PATH 时构造不出真客户端,拿这个占位照样能读缓存。
    """

    def __init__(self, cache_name):
        self.cache_name = cache_name


def open_client_or_cache(name, timeout=30.0, model=None, live=True):
    """联网时照常构造客户端;只读缓存(live=False)时构造失败就退回占位客户端。

    T-011: `tools-real translate` 不加 `--live` 时,环境里没有 key 也应当能只读缓存,
    不该因为造不出客户端就整个退出。
    """
    try:
        return open_client(name, timeout=timeout, model=model)
    except Stopped:
        if live:
            raise
        cache_name = name
        if name == "codebuddy" and model:
            cache_name = f"codebuddy@{model}"
        return _CacheOnlyClient(cache_name)


# ---------------- 1. 翻译 ----------------

def translation_questions() -> dict:
    """翻译那一步的问题集:一题自由文本。

    走的是 `OpenRouterChatClient` 同一套 HTTP / 重试 / 脱敏 / 计费,
    所以翻译和打标共用一份缓存机制和一份费用记录。
    """
    return {"translation": {
        "type": "text",
        "instructions": "把这条用户请求翻译成自然、口语化的中文请求。"
                        "保留专有名词、数字、URL、代码标识符原样;不要回答这条请求,只输出译文。",
        "criteria": {"answer": "整句中文译文,不要加解释"},
    }}


def tool_translation_questions() -> dict:
    """工具描述翻译(T-011):一题自由文本,保留函数名、参数名、代码标识符。"""
    return {"translation": {
        "type": "text",
        "instructions": "把这段工具说明翻译成自然、准确的中文。"
                        "保留函数名、参数名、代码标识符、专有名词、数字、URL 原样;"
                        "只输出译文,不要解释。",
        "criteria": {"answer": "整句中文译文,不要加解释"},
    }}


def _needs_translation(row) -> bool:
    lang = row.get("lang")
    if lang is None:
        lang = "en" if is_english(row["text_orig"]) else "zh"
    return lang == "en"


def translate(pool, client, client_name, workers=WORKERS_DEFAULT, budget=BUDGET_DEFAULT,
              cost=None, live=True, stats=None) -> dict:
    """英文条目翻成中文口语请求,中文条目直接沿用。返回 {"rows", "stats", "untranslated"}。

    没翻成的条目保留原文继续走打标(不静默丢数据),条数记在 untranslated 里,报告要写出来。
    """
    stats = stats if stats is not None else compare.FetchStats()
    if live and cost is not None:
        cost.check(budget)
    items = [{"text": r["text_orig"]} for r in pool if _needs_translation(r)]
    got = {}
    if items:
        # 客户端的缓存名(指定模型的 codebuddy 是 codebuddy@<模型>)与展示名分开
        recs = compare.fetch_answers(client, getattr(client, "cache_name", client_name), items,
                                     TRANSLATE_TAG, "translate", live=live,
                                     questions=translation_questions(), workers=workers,
                                     stats=stats)
        got = {r["text"]: r for r in recs}
    rows, untranslated = [], 0
    for r in pool:
        text = r["text_orig"]
        if not _needs_translation(r):
            rows.append(dict(r, text=text))
            continue
        rec = got.get(text)
        zh = ""
        if rec and rec.get("ok"):
            zh = ((rec.get("answers") or {}).get("translation") or {}).get("text") or ""
        if zh.strip():
            rows.append(dict(r, text=zh.strip()))
        else:
            untranslated += 1
            rows.append(dict(r, text=text))
    if live and cost is not None:
        cost.add("translate", client_name, stats.cost)
        cost.save()
    if stats.stopped:
        raise Stopped(stats.stopped, EXIT_CLIENT)
    return {"rows": rows, "stats": stats, "untranslated": untranslated}


def translate_tools(tools, client, client_name, live=False, workers=WORKERS_DEFAULT,
                    budget=None, cost=None) -> dict:
    """把真实工具描述的英文条目翻成中文,返回 {"rows", "stats", "untranslated"}。

    缓存键是 (client_name, "zh", "tools_real_desc"),按 description 文本去重。
    没翻成或原本没描述的写空串,条数记在 untranslated 里。
    """
    stats = compare.FetchStats()
    if live and cost is not None:
        cost.check(budget)
    items = [{"text": t.get("description", "")} for t in tools
             if t.get("description", "").strip()]
    got = {}
    if items:
        recs = compare.fetch_answers(client, getattr(client, "cache_name", client_name), items,
                                     TRANSLATE_TAG, "tools_real_desc", live=live,
                                     questions=tool_translation_questions(), workers=workers,
                                     stats=stats)
        got = {r["text"]: r for r in recs}
    rows, untranslated = [], 0
    for t in tools:
        text = t.get("description", "")
        if not text.strip():
            rows.append({"tool_id": t["tool_id"], "description_zh": ""})
            continue
        rec = got.get(text)
        zh = ""
        if rec and rec.get("ok"):
            zh = ((rec.get("answers") or {}).get("translation") or {}).get("text") or ""
        if not zh.strip():
            untranslated += 1
        rows.append({"tool_id": t["tool_id"], "description_zh": zh.strip()})
    if live and cost is not None:
        cost.add("translate-tools", client_name, stats.cost)
        cost.save()
    if stats.stopped:
        raise Stopped(stats.stopped, EXIT_CLIENT)
    return {"rows": rows, "stats": stats, "untranslated": untranslated}


# ---------------- 2. 双路打标 ----------------

def _labels(rec):
    """一条缓存记录 -> {"need","domains","actions","ok","error"};没问到返回 None。"""
    if not rec or not rec.get("ok") or not rec.get("answers"):
        return None
    labels = answers_to_labels(rec["answers"])
    return {"need": labels["need"], "domains": list(labels["domains"]),
            "actions": list(labels["actions"]), "ok": True, "error": ""}


def _failed_labels(rec):
    return {"need": "", "domains": [], "actions": [], "ok": False,
            "error": (rec or {}).get("error") or "没问到"}


def agree(a: dict, b: dict) -> bool:
    """一致的定义(exact 口径):need 相同;need 非 none 时 domains 集合相同。"""
    if a["need"] != b["need"]:
        return False
    return a["need"] == "none" or set(a["domains"]) == set(b["domains"])


def domain_noul_probs(answers) -> dict:
    """原始 answers 里每个域的 noul 概率;那一题没问到的域不在结果里。

    v2 的缓存是 12 域时期问的,里面没有 weather_geo / entertainment 那两题,
    所以主域天然只会从当时的 12 个域里挑。
    """
    out = {}
    for d in DOMAIN_IDS:
        noul = (answers.get(f"domain_{d}") or {}).get("noul")
        if noul is not None:
            out[d] = float(noul)
    return out


def primary_domain(answers, domains) -> str:
    """第一路的主域:各域 noul 概率最高的那个(并列时取 DOMAIN_IDS 里靠前的)。

    答案里没有概率可看时(比如假客户端给的答案),退回第一路 domains 的第一个。
    """
    probs = domain_noul_probs(answers or {})
    if probs:
        return max(probs, key=probs.get)
    return next(iter(domains or []), "")


def agreement_outcome(a, b, answers_a=None, answers_b=None, rule=AGREEMENT_DEFAULT) -> dict:
    """三种口径的一致判定,统一返回 {"agreed", "domains", "primary_domain"};domains 是这一致的标签。

    exact:need 相同,need 非 none 时域集合相同 —— 标签沿用第一路的整份 domains。
    primary:need 相同,且第一路的主域在第二路的 domains 里 —— 标签 domains = [主域]
             (actions 仍按 `merge_actions`,这一点三种口径都一样)。
    primary_sym:上面那条或者"第二路的主域在第一路的 domains 里"都算一致 ——
             domains 取命中的那个主域,两个都命中时取第一路的。

    Jev 一路多选偏多(v3 平均 2.86 个域),要求两个集合完全相同会把大量"need 其实一致、
    只是一个多选了一个或少选了一个"的条目打进待人工,主域口径就是为此加的。
    """
    pa = primary_domain(answers_a, a.get("domains"))
    pb = primary_domain(answers_b, b.get("domains"))
    if a["need"] != b["need"]:
        return {"agreed": False, "domains": [], "primary_domain": pa}
    if a["need"] == "none":
        return {"agreed": True, "domains": [], "primary_domain": ""}
    if rule == AGREEMENT_EXACT:
        same = set(a["domains"]) == set(b["domains"])
        return {"agreed": same, "domains": list(a["domains"]), "primary_domain": pa}
    hit_a = pa in b["domains"]
    if rule == "primary_sym" and not hit_a and pb in a["domains"]:
        return {"agreed": True, "domains": [pb], "primary_domain": pa}
    return {"agreed": hit_a, "domains": [pa] if hit_a else [], "primary_domain": pa}


def merge_actions(a: dict, b: dict) -> list:
    """actions 取两路的交集(按第一路的顺序);交集为空时取第一路的。"""
    inter = [x for x in a["actions"] if x in b["actions"]]
    return inter or list(a["actions"])


def is_math_like(row) -> bool:
    """`scripts/build_pool.py` 给 BFCL 打的 `hint.math_like`:函数名/描述像算学或计算的。

    只用来分开报数字,不当标签用;非 BFCL 来源没有这个 hint,一律算 False。
    """
    return bool((row.get("hint") or {}).get("math_like"))


def _math_like_split(agreed, review, pool_zh) -> dict:
    """按 math_like 分开报:每组的一致率与 need 分布(一致集里的)。

    提出这一组数字是为了看"算学题"这一类是不是特别容易让两路打不一致。
    """
    flag = {r["text"]: is_math_like(r) for r in pool_zh}
    out = {}
    for key, rows in (("math_like", [r for r in pool_zh if flag.get(r["text"])]),
                      ("other", [r for r in pool_zh if not flag.get(r["text"])])):
        texts = {r["text"] for r in rows}
        ag = [r for r in agreed if r["text"] in texts]
        rv = [r for r in review if r["text"] in texts]
        n = len(ag) + len(rv)
        out[key] = {"n": n, "agreed": len(ag), "review": len(rv),
                    "agree_rate": (len(ag) / n) if n else None,
                    "need": dict(Counter(r["need"] for r in ag))}
    return out


def dual_label(pool_zh, labelers, need_version=NEED_VERSION_DEFAULT, workers=WORKERS_DEFAULT,
               budget=BUDGET_DEFAULT, cost=None, live=True,
               agreement=AGREEMENT_DEFAULT) -> dict:
    """两路各问同一组题(T-009 起是 1 + 14 个域 + 6 个动作 = 21 题):一致的进 agree,不一致或任一路失败进 review。

    labelers: [(名字, 客户端), ...],第一个是基准路(actions 交集为空时取它的,
    主域也从它的原始 answers 里取,见 `primary_domain`)。名字同时决定缓存文件名
    (指定模型的 codebuddy 除外,见 `CodeBuddyClient.cache_name`)。
    agreement 决定一致的口径(exact / primary / primary_sym,见 `agreement_outcome`),
    默认 primary;待审队列每行带着 `agreement_rule` 与 `primary_domain`,换口径重算后
    一眼能看出这行是按哪条规则判的、第一路的主域是什么。
    结果里 primary_missing 单独报"第一路缺答案"的条数 ——这批只能靠单路打标补,
    不能当成"两路不一致"。
    """
    names = [n for n, _ in labelers]
    got, stats = {}, []
    for name, client in labelers:
        s = compare.FetchStats()
        if live and cost is not None:
            cost.check(budget)
        recs = compare.fetch_answers(client, getattr(client, "cache_name", name), pool_zh,
                                     need_version, "pool", live=live, workers=workers, stats=s)
        got[name] = {r["text"]: r for r in recs}
        stats.append((name, s))
        if live and cost is not None:
            cost.add("label", name, s.cost)
            cost.save()
        if s.stopped:
            raise Stopped(s.stopped, EXIT_CLIENT)

    primary, second = names[0], names[1]
    agreed, review = [], []
    primary_missing = 0
    for r in pool_zh:
        rec_a = got[primary].get(r["text"])
        rec_b = got[second].get(r["text"])
        a = _labels(rec_a)
        b = _labels(rec_b)
        if a is None:
            primary_missing += 1          # 第一路没答案:单路打标的事,不是"两路不一致"
        out = (agreement_outcome(a, b, (rec_a or {}).get("answers"),
                                 (rec_b or {}).get("answers"), agreement)
               if a is not None and b is not None
               else {"agreed": False, "domains": [], "primary_domain": ""})
        if out["agreed"]:
            agreed.append({"id": r["id"], "text": r["text"], "text_orig": r["text_orig"],
                           "source": r["source"], "need": a["need"], "domains": out["domains"],
                           "actions": merge_actions(a, b), "label_source": "+".join(names),
                           "hint": r.get("hint") or {}})
            continue
        review.append({"id": r["id"], "text": r["text"], "text_orig": r["text_orig"],
                       "source": r["source"], "labelers": names, "agreement_rule": agreement,
                       "primary_domain": out["primary_domain"], "hint": r.get("hint") or {},
                       primary: a if a is not None else _failed_labels(got[primary].get(r["text"])),
                       second: b if b is not None else _failed_labels(got[second].get(r["text"]))})
    return {"agreed": agreed, "review": review, "stats": stats, "labelers": names,
            "agreement": agreement, "primary_missing": primary_missing,
            "by_math_like": _math_like_split(agreed, review, pool_zh)}


def single_label(pool_zh, labeler, need_version=NEED_VERSION_DEFAULT, workers=WORKERS_DEFAULT,
                 budget=BUDGET_DEFAULT, cost=None, live=True) -> dict:
    """单路打标(`realdata label --single`):只用一路,结果单独落盘,不进双路数据。

    用在双路里第一路缺答案的那批(比如 Jev 402 之后的 261 条 alpaca 负例):
    每行带 `label_source`(这一路的名字)和 `single_source: true`,split 不读它,
    evaluate-real 把它并进 D 方案,只在银标留出集上看有没有帮助。
    这一路也没问到答案的条目跳过,条数记在 unlabeled 里。
    """
    name, client = labeler
    stats = compare.FetchStats()
    if live and cost is not None:
        cost.check(budget)
    recs = compare.fetch_answers(client, getattr(client, "cache_name", name), pool_zh,
                                 need_version, "pool", live=live, workers=workers, stats=stats)
    got = {r["text"]: r for r in recs}
    if live and cost is not None:
        cost.add("label", name, stats.cost)
        cost.save()
    if stats.stopped:
        raise Stopped(stats.stopped, EXIT_CLIENT)
    rows, unlabeled = [], 0
    for r in pool_zh:
        lab = _labels(got.get(r["text"]))
        if lab is None:
            unlabeled += 1
            continue
        rows.append({"id": r["id"], "text": r["text"], "text_orig": r["text_orig"],
                     "source": r["source"], "need": lab["need"], "domains": lab["domains"],
                     "actions": lab["actions"], "label_source": name, "single_source": True,
                     "hint": r.get("hint") or {}})
    return {"rows": rows, "stats": stats, "unlabeled": unlabeled, "labeler": name}


# ---------------- 3. 切分 ----------------

def split(agreed, test_size=HOLDOUT_SIZE, random_state=0):
    """一致集按 need 分层:`test_size` 进银标留出集,其余进 real_train。

    留出集的标签来自两路一致,不是人工金标,报告里必须写明它偏向这两个模型。
    """
    if len(agreed) < 2:
        return list(agreed), []
    labels = [r["need"] for r in agreed]
    counts = Counter(labels)
    # 折数不够分层就退化成不分层的随机切,免得直接报错
    stratify = labels if len(agreed) >= 10 and min(counts.values()) >= 2 else None
    idx = list(range(len(agreed)))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=random_state,
                                      stratify=stratify)
    return [agreed[i] for i in tr_idx], [agreed[i] for i in te_idx]


# ---------------- 4. 金标候选表 ----------------

def _join(values) -> str:
    return MULTI_SEP.join(values or [])


def _split_ids(value: str) -> list:
    return [x.strip() for x in (value or "").split(MULTI_SEP) if x.strip()]


def _check_ids(line_no, field, values, allowed):
    for v in values:
        if v not in allowed:
            raise ValueError(f"第 {line_no} 行:{field} 里的 {v!r} 不是合法 id,"
                             f"可选 {list(allowed)}")


def _review_note(r, a, b) -> str:
    """不一致的原因写进 note:exact 口径沿用旧的"两路不一致",主域口径分出 need / 主域两种。"""
    if not a.get("ok") or not b.get("ok"):
        return "有一路没问到答案"
    if (r.get("agreement_rule") or AGREEMENT_EXACT) == AGREEMENT_EXACT:
        return "两路不一致"
    return "need 不一致" if a.get("need") != b.get("need") else "主域不一致"


def _review_domains(r, a, b) -> str:
    """review 行预填的 domains:主域口径下 need 相同时预填第一路的主域,其余沿用第一路的整份。"""
    primary = r.get("primary_domain")
    need_same = a.get("need") == b.get("need")
    if primary and need_same and (r.get("agreement_rule") or AGREEMENT_EXACT) != AGREEMENT_EXACT:
        return primary
    return _join(a.get("domains"))


def _candidate_row(r, names):
    """候选表的一行;review 行与留出行的差别只有预填值不同。

    两路的判断在 labeler1_* / labeler2_* 列里,各自的名字在 labeler1_name / labeler2_name
    (第一路是 `--labelers` 里第一个,基准路)。review 行里有一路没问到答案时
    note 写"有一路没问到答案",和真正的"两路不一致"分开;按主域判定的行进一步写明
    是 need 不一致还是主域不一致,domains 也只预填第一路的那个主域。
    """
    base = {"id": r["id"], "text": r["text"], "text_orig": r["text_orig"], "source": r["source"],
            "labeler1_name": names[0], "labeler2_name": names[1],
            "local_need": "", "local_domains": "", "note": ""}
    if "labelers" in r:                      # review_queue:两路不一致 / 有一路失败
        a, b = r[names[0]], r[names[1]]
        # need 由本地模型预填(见 candidates);domains / actions 都预填基准路(第一路)的,
        # gu 只改错的。两路的原始判断都在 labeler1_* / labeler2_* 那几列里,改的时候可以对照。
        return dict(base,
                    labeler1_need=a.get("need", ""), labeler1_domains=_join(a.get("domains")),
                    labeler2_need=b.get("need", ""), labeler2_domains=_join(b.get("domains")),
                    need="", domains=_review_domains(r, a, b), actions=_join(a.get("actions")),
                    note=_review_note(r, a, b))
    return dict(base,                        # 留出集:两路一致
                labeler1_need=r["need"], labeler1_domains=_join(r["domains"]),
                labeler2_need=r["need"], labeler2_domains=_join(r["domains"]),
                need=r["need"], domains=_join(r["domains"]), actions=_join(r["actions"]))


def _stratified_take(rows, k, seed, key):
    """按 key 分层轮转取 k 条:层内用固定种子打乱,层间一条一条轮着取。"""
    if k <= 0 or not rows:
        return []
    rnd = random.Random(seed)
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    for g in groups.values():
        rnd.shuffle(g)
    out, i = [], 0
    while len(out) < k:
        added = False
        for name in sorted(groups):
            if i < len(groups[name]) and len(out) < k:
                out.append(groups[name][i])
                added = True
        if not added:
            break
        i += 1
    return out


def candidates(review, holdout, predict, n=N_CANDIDATES, seed=0) -> list:
    """给 gu 标注的候选表:review_queue 与留出集各抽一半,都按 need 分层;不够就从另一边补。

    predict: f(texts) -> [(need, domains, actions) 本地模型的预测,只用来预填 local_* 两列
    和 review 行的建议 need;gu 只改错的。
    """
    names = review[0]["labelers"] if review else ["jev", "haiku"]
    rev_rows = [_candidate_row(r, names) for r in review]
    hold_rows = [_candidate_row(r, names) for r in holdout]
    if predict is not None:
        texts = [r["text"] for r in rev_rows + hold_rows]
        for row, p in zip(rev_rows + hold_rows, predict(texts)):
            row["local_need"], row["local_domains"] = p[0], _join(p[1])
        for row in rev_rows:
            # 两路 need 相同、只是能力域不同时,need 用两路的共同判断;need 本身不一致才按本地模型预填。
            # domains 已经在 _candidate_row 里填了第一路的。
            same = row["labeler1_need"] and row["labeler1_need"] == row["labeler2_need"]
            row["need"] = row["labeler1_need"] if same else row["local_need"]
    half = n // 2
    picked = _stratified_take(rev_rows, half, seed, lambda r: r["need"] or r["local_need"])
    picked += _stratified_take(hold_rows, half, seed, lambda r: r["need"] or r["local_need"])
    if len(picked) < n:                      # 一边不够就从另一边补
        have = {id(r) for r in picked}
        rest = [r for r in rev_rows + hold_rows if id(r) not in have]
        picked += rest[:n - len(picked)]
    return picked[:n]


def write_gold_candidates(path, rows) -> None:
    """UTF-8 with BOM(utf-8-sig),Excel 直接打开不乱码。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CAND_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CAND_COLUMNS})


def render_candidates_readme() -> str:
    """候选表的说明:每列含义、怎么改、类别定义(类别卡从 taxonomy 生成)。"""
    L = ["# 金标候选表说明(data/gold_candidates.csv)\n",
         "200 条真实请求,请逐条核对 `need / domains / actions` 三列:预填值大多数是对的,"
         "**只改错的**,改完把文件发回来即可(用 Excel 直接打开就行,文件带 BOM)。\n",
         "\n## 每列是什么\n",
         "| 列 | 含义 | 要不要改 |\n| --- | --- | --- |",
         "| id | 样本编号(`来源-序号`),用来回查原始数据集 | 不要改 |",
         "| text | 翻译成中文的请求正文,模型看到的就是这一列 | 不要改 |",
         "| text_orig | 数据集里的原文 | 不要改 |",
         "| source | 来自哪个数据集(bfcl / apibank / alpaca_zh) | 不要改 |",
         "| labeler1_name / labeler2_name | 两路各自的客户端名(第一路是 `--labelers` 里第一个,默认 Jev) | 不要改 |",
         "| labeler1_need / labeler1_domains | 第一路(基准路)的判断 | 参考 |",
         "| labeler2_need / labeler2_domains | 第二路的判断 | 参考 |",
         "| local_need / local_domains | 本地模型的判断 | 参考 |",
         "| need | **要核对的**:none / single / multi | 改错的 |",
         "| domains | **要核对的**:能力域,多值用 `;` 分隔 | 改错的 |",
         "| actions | **要核对的**:动作类型,多值用 `;` 分隔 | 改错的 |",
         "| note | `need 不一致` / `主域不一致` 两个模型判断不同;`有一路没问到答案` 表示有一路当时没拿到结果 | 不要改 |",
         "\n两路不一致的行怎么预填的:`need` 按本地模型的判断,`domains` 和 `actions` 按第一路"
         "(`--labelers` 里第一个,默认 jev)的判断,两路的原始判断都留在 `labeler1_*` / `labeler2_*` "
         "那几列里可以对照。觉得都不对就直接改成你认为对的。\n",
         "\n## 类别定义(取自 `taxonomy.py`,改表时以这里为准)\n",
         category_cards()]
    return "\n".join(L) + "\n"


def load_gold_candidates(path):
    """读回 gu 改过的 CSV 并校验,转成 gold 格式的记录列表。

    非法类别 id 直接报出行号(表头算第 1 行,所以第一条数据从第 2 行开始数)。
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows, start=2):
        need = (r.get("need") or "").strip()
        if need not in NEED_IDS:
            raise ValueError(f"第 {i} 行:need 值 {need!r} 不在 {list(NEED_IDS)} 里")
        domains, actions = _split_ids(r.get("domains")), _split_ids(r.get("actions"))
        _check_ids(i, "domains", domains, DOMAIN_IDS)
        _check_ids(i, "actions", actions, ACTION_IDS)
        out.append({"id": (r.get("id") or "").strip(), "text": (r.get("text") or "").strip(),
                    "source": (r.get("source") or "").strip(), "need": need,
                    "domains": domains, "actions": actions})
    return out


# ---------------- 4b. 参考集(D-005 定稿) ----------------

def load_request_reference(path, pool_zh=POOL_ZH):
    """读请求参考集 CSV(Claude 预标 + gu 抽查定稿,D-005)。

    只取 `id / need / domains / actions` 四列当标签;`text / text_orig / source` 不读 CSV,
    而是拿 `id` 对回请求池(`pool_zh.jsonl` 里有中文 `text`),保证喂给模型的是池里那份原文。
    两路 need 也一并带回:两路一致时给 `silver_need`,供"银标 vs 参考集"的一致率用。
    """
    pool = {r["id"]: r for r in load_jsonl(pool_zh)}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows, start=2):
        rid = (r.get("id") or "").strip()
        if rid not in pool:
            raise ValueError(f"第 {i} 行:id {rid!r} 不在请求池 {pool_zh}")
        need = (r.get("need") or "").strip()
        if need not in NEED_IDS:
            raise ValueError(f"第 {i} 行:need 值 {need!r} 不在 {list(NEED_IDS)} 里")
        domains, actions = _split_ids(r.get("domains")), _split_ids(r.get("actions"))
        _check_ids(i, "domains", domains, DOMAIN_IDS)
        _check_ids(i, "actions", actions, ACTION_IDS)
        n1 = (r.get("labeler1_need") or "").strip()
        n2 = (r.get("labeler2_need") or "").strip()
        out.append({"id": rid, "text": pool[rid]["text"],
                    "text_orig": pool[rid].get("text_orig", ""),
                    "source": pool[rid].get("source", ""), "need": need,
                    "domains": domains, "actions": actions,
                    "silver_need": n1 if (n1 and n1 == n2) else None})
    return out


# ---------------- 5. 重训与评估 ----------------

def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _as_train_rows(rows):
    """真实数据没有 tool_ids(那是 gold 用来算 Recall@k 的),建索引前补成空列表。

    补空列表不会给检索加任何示例查询,所以不影响 Recall@k 的口径。
    """
    return [dict(r, tool_ids=r.get("tool_ids") or []) for r in rows]


def _gold_row(router, gold) -> dict:
    """合成 gold 上的一整套指标(含 Recall@k:只有 gold 有 tool_ids)。"""
    res = ev.eval_requests(router, gold)
    conf = res["need"]["confusion"]
    row = {"eval_set": "合成 gold", "n": res["n"], "need_accuracy": res["need"]["accuracy_3class"],
           "need_macro_f1": res["need"]["macro_f1_3class"],
           "recall_needs_mcp": res["need"]["recall_needs_mcp"],
           "none_correct": conf["none"]["none"],
           "none_total": sum(conf["none"].values()),
           "domain": res["domain"], "action": res["action"],
           "need_missing": list(getattr(router.head, "need_missing", ()) or ()),
           "recall": {k: res["routing"][f"filtered@{k}"] for k in (1, 3, 5)}}
    return row


def _holdout_row(router, holdout) -> dict:
    """银标留出集的一套指标,外加按 source 与 math_like 拆分的 need 准确率。"""
    outs = router.head.predict([r["text"] for r in holdout])
    m = ev.request_metrics(holdout, [(o.need, o.domains, o.actions) for o in outs])
    by_source = {}
    for src in sorted({r["source"] for r in holdout}):
        sub = [(r, o) for r, o in zip(holdout, outs) if r["source"] == src]
        by_source[src] = _mean([float(r["need"] == o.need) for r, o in sub])
    by_math_like = {}
    for key, want in (("math_like", True), ("other", False)):
        sub = [(r, o) for r, o in zip(holdout, outs) if is_math_like(r) is want]
        if not sub:
            continue
        by_math_like[key] = {"n": len(sub),
                             "accuracy": _mean([float(r["need"] == o.need) for r, o in sub]),
                             "need": dict(Counter(r["need"] for r, _ in sub))}
    return {"eval_set": "银标留出集", "n": len(holdout), "need_accuracy": m["need_accuracy"],
            "need_macro_f1": m["need_macro_f1"], "recall_needs_mcp": m["recall_needs_mcp"],
            "none_correct": m["none_correct"], "none_total": m["none_total"],
            "domain": m["domain"], "action": m["action"], "recall": None,
            "need_missing": list(getattr(router.head, "need_missing", ()) or ()),
            "by_source": by_source, "by_math_like": by_math_like}


def cv_on_real(encoder, tools, real_train, folds=5, random_state=0):
    """C 方案在 real_train 上的 k 折交叉验证;某一类样本不够 k 条时自动降折数。"""
    if not real_train:
        return None
    counts = Counter(r["need"] for r in real_train)
    k = min(folds, min(counts.values()))
    if k < 2:
        return None
    per_fold = []
    for tr_idx, te_idx in ev.cv_splits(real_train, k, random_state):
        tr = [real_train[i] for i in tr_idx]
        te = [real_train[i] for i in te_idx]
        head = ev.train_request_head(ev.fit_encoder(encoder, tools, tr), tr)
        outs = head.predict([r["text"] for r in te])
        per_fold.append(ev.need_metrics([r["need"] for r in te], [o.need for o in outs]))
    out = {"folds": k, "random_state": random_state, "n": len(real_train), "per_fold": per_fold}
    for key in ("accuracy_3class", "macro_f1_3class", "recall_needs_mcp"):
        vals = [f[key] for f in per_fold]
        out[key] = {"mean": _mean(vals), "std": (sum((v - _mean(vals)) ** 2 for v in vals)
                                                 / len(vals)) ** 0.5 if vals else 0.0}
    return out


def evaluate_real(tools, synth_train, gold, real_train, holdout, labeled_single=(),
                  encoder="tfidf", folds=5, random_state=0) -> dict:
    """三种训练集 × 两个评估集;有单路数据(labeled_single)时再加一组 D。

    选 A/B/C 看银标留出集和交叉验证,不看合成 gold。D = 合成 + real_train + 单路数据,
    只在银标留出集上报(和 C 对比 need 准确率、none 召回、能力域 F1,看单路数据有没有帮助);
    单路数据不进 A/B/C,也不进 real_train / holdout。
    """
    variants = [("A 只用合成训练集", synth_train), ("B 只用 real_train", real_train),
                ("C 合成 + real_train", synth_train + real_train)]
    rows = []
    for name, train in variants:
        if not train:
            continue
        router = ev.build_system(encoder, tools, _as_train_rows(train))
        rows.append(dict({"variant": name, "train_n": len(train)},
                         **(_gold_row(router, gold) if gold else {})))
        if holdout:
            rows.append(dict({"variant": name, "train_n": len(train)},
                             **_holdout_row(router, holdout)))
    if labeled_single and holdout:
        train = synth_train + real_train + labeled_single
        if train:
            router = ev.build_system(encoder, tools, _as_train_rows(train))
            rows.append(dict({"variant": "D 合成 + real_train + 单路", "train_n": len(train)},
                             **_holdout_row(router, holdout)))
    return {"encoder": encoder, "rows": rows,
            "cv_on_real": cv_on_real(encoder, tools, real_train, folds, random_state),
            "n": {"synth_train": len(synth_train), "gold": len(gold),
                  "real_train": len(real_train), "holdout": len(holdout),
                  "labeled_single": len(labeled_single)}}


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _recall_cell(recall) -> str:
    if not recall:
        return "-"
    return " / ".join(_pct(recall[k]) for k in (1, 3, 5))


def render_real_eval(result) -> str:
    """三种训练集在两个评估集上的对照表;有单路数据时加一行 D 对比 C。"""
    L = ["# 真实数据重训评估(T-006)\n",
         f"编码器:`{result['encoder']}`;"
         f"合成训练集 {result['n']['synth_train']} 条 / 合成 gold {result['n']['gold']} 条 / "
         f"real_train {result['n']['real_train']} 条 / 银标留出集 {result['n']['holdout']} 条 / "
         f"单路数据 {result['n'].get('labeled_single', 0)} 条。\n",
         "> 留出集的标签来自 Jev 与 Haiku 两路一致,不是人工金标,"
         "所以下面的数字偏向这两个模型,不能当成真实水平。\n",
         "> 选 A/B/C 看银标留出集和下面的交叉验证,不看合成 gold。\n",
         "\n| 训练集 | 评估集 | n | need 准确率 | need macro-F1 | 需要 MCP 召回 | none 判对 "
         "| 能力域 micro/macro | 动作 micro/macro | Recall@1/3/5 |\n"
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        L.append(f"| {r['variant']} | {r['eval_set']} | {r['n']} | {_pct(r['need_accuracy'])} "
                 f"| {_pct(r['need_macro_f1'])} | {_pct(r['recall_needs_mcp'])} "
                 f"| {r['none_correct']}/{r['none_total']} "
                 f"| {_pct(r['domain']['micro_f1'])} / {_pct(r['domain']['macro_f1'])} "
                 f"| {_pct(r['action']['micro_f1'])} / {_pct(r['action']['macro_f1'])} "
                 f"| {_recall_cell(r.get('recall'))} |")
    seen = set()
    for r in result["rows"]:
        missing = r.get("need_missing") or []
        if not missing or r["variant"] in seen:    # 同一个训练集两个评估集,提示只写一次
            continue
        seen.add(r["variant"])
        L.append(f"\n**{r['variant']}**:训练集缺 {'、'.join(sorted(missing))},"
                 f"{'、'.join(sorted(missing))} 无法预测(两段式 need 头的第二段"
                 f"恒定输出剩下的那一类,概率 1.0)。")
    for r in result["rows"]:
        if r.get("by_source"):
            cells = "、".join(f"{k} {_pct(v)}" for k, v in r["by_source"].items())
            L.append(f"\n**{r['variant']} 在留出集上按来源拆分的 need 准确率**:{cells}")
        if r.get("by_math_like"):
            for key, g in r["by_math_like"].items():
                need_dist = "、".join(f"{k} {v}" for k, v in sorted(g["need"].items())) or "空"
                L.append(f"\n**{r['variant']} 在留出集上按 math_like={key} 拆分的 need 准确率**"
                         f":{_pct(g['accuracy'])}({g['n']} 条);need 分布 {need_dist}")
    cv = result.get("cv_on_real")
    if cv:
        acc, f1 = cv["accuracy_3class"], cv["macro_f1_3class"]
        L.append(f"\n**C 方案在 real_train 上的交叉验证**({cv['folds']} 折,"
                 f"`random_state={cv['random_state']}`,{cv['n']} 条):"
                 f"need 准确率 {_pct(acc['mean'])} ± {acc['std'] * 100:.1f}%,"
                 f"need macro-F1 {_pct(f1['mean'])} ± {f1['std'] * 100:.1f}%。")
    else:
        L.append("\nreal_train 为空或某一类样本不足 2 条,交叉验证没跑。")
    d_vs_c = _d_vs_c_line(result["rows"])
    if d_vs_c:
        L.append(d_vs_c)
    return "\n".join(L) + "\n"


def _d_vs_c_line(rows) -> str:
    """D(加单路数据)对比 C,只在银标留出集上;没有 D 行时返回空串。"""
    hold = [r for r in rows if r["eval_set"] == "银标留出集"]
    c = next((r for r in hold if r["variant"].startswith("C ")), None)
    d = next((r for r in hold if r["variant"].startswith("D ")), None)
    if not (c and d):
        return ""
    return (f"\n**D(合成 + real_train + 单路)对比 C(合成 + real_train),银标留出集**:"
            f"need 准确率 {_pct(c['need_accuracy'])} → {_pct(d['need_accuracy'])},"
            f"需要 MCP 召回 {_pct(c['recall_needs_mcp'])} → {_pct(d['recall_needs_mcp'])},"
            f"能力域 micro-F1 {_pct(c['domain']['micro_f1'])} → {_pct(d['domain']['micro_f1'])}。"
            f"单路数据只有一路模型的判断,这行数字偏那一个模型,只能看趋势。")
