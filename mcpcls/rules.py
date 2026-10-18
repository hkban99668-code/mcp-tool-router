"""工具侧的规则信号:工具名动词、参数名、MCP annotations。

两个用途:
1. 作为离散特征喂给工具侧分类头(tool_features)。
2. 推导动作类型和风险等级的规则下限(rule_actions / rule_risk)。最终风险取规则与模型中更高的一档。

annotations 由 server 作者自报,可能缺失或低报,所以规则以工具名和 schema 为主。
"""

import re
from pathlib import Path

from .taxonomy import L3_ACTIONS, OPEN_WORLD_DOMAINS, WRITE_ACTIONS, max_risk

VERB_ACTIONS = {
    "read": {"get", "list", "search", "find", "query", "read", "fetch", "lookup", "describe",
             "show", "view", "download", "retrieve", "count", "check", "status", "logs", "stat"},
    "create": {"create", "add", "new", "insert", "upload", "write", "save", "make", "generate",
               "schedule", "open", "draft", "invite", "book"},
    "update": {"update", "edit", "modify", "set", "rename", "move", "patch", "merge", "assign",
               "scale", "comment", "label", "complete", "close", "archive", "restart"},
    "delete": {"delete", "remove", "drop", "destroy", "purge", "truncate", "cancel", "revoke",
               "uninstall", "terminate", "overwrite"},
    "execute": {"run", "rerun", "retry", "exec", "execute", "eval", "invoke", "deploy", "apply", "shell",
                "command", "script", "trigger", "click", "navigate", "type", "fill", "evaluate",
                "play"},
    "send": {"send", "post", "publish", "reply", "email", "message", "notify", "tweet",
             "pay", "charge", "refund", "transfer", "broadcast", "forward", "share",
             "buy", "order", "reserve", "book", "booking"},
}

# 参数名里的强信号
PARAM_ACTIONS = {
    "execute": {"command", "cmd", "script", "code", "shell"},
    "send": {"recipient", "recipients", "to", "cc", "bcc", "channel", "amount", "payee"},
    "delete": {"force", "cascade"},
}
PARAM_DOMAIN_HINTS = {
    "file": {"path", "filepath", "filename", "file_path", "directory"},
    "database": {"sql", "query_sql", "table", "schema_name"},
    "search_web": {"url", "query", "q"},
    "code": {"repo", "owner", "branch", "pull_number", "issue_number"},
    "comms": {"recipient", "recipients", "channel", "subject"},
    "calendar_task": {"start_time", "end_time", "due_date", "attendees"},
    "cloud_ops": {"namespace", "pod", "cluster", "replicas", "region"},
    "business": {"amount", "currency", "customer_id", "order_id"},
    "media": {"prompt", "image_url", "audio_url", "width", "height"},
}

# 能力域的英文关键词先验。MCP 工具描述多为英文,新 server 的词表训练集里可能没见过,
# 这组关键词让工具侧分类头在没见过的 server 上也有基本信号。
DOMAIN_KEYWORDS = {
    "file": {"file", "files", "directory", "folder", "path", "drive", "pdf", "document", "docx", "disk"},
    "code": {"github", "gitlab", "git", "repository", "repo", "pull", "commit", "branch", "issue", "issues",
             "ci", "workflow", "build", "pipeline", "package", "npm"},
    "database": {"sql", "database", "db", "table", "tables", "postgres", "mysql", "query", "spreadsheet",
                 "sheet", "sheets", "cells", "warehouse", "bigquery"},
    "search_web": {"web", "search", "url", "fetch", "internet", "arxiv", "papers", "scholar", "crawl", "markdown"},
    "browser": {"browser", "page", "click", "element", "screenshot", "navigate", "playwright", "puppeteer", "form"},
    "comms": {"slack", "email", "emails", "mail", "gmail", "message", "messages", "channel", "chat",
              "meeting", "zoom", "teams", "inbox", "recipients"},
    "calendar_task": {"calendar", "event", "events", "schedule", "task", "tasks", "todo", "issue", "tracker",
                      "linear", "jira", "assignee", "due"},
    "knowledge": {"notion", "wiki", "page", "pages", "knowledge", "memory", "note", "notes", "confluence", "graph"},
    "cloud_ops": {"kubernetes", "k8s", "pod", "pods", "deployment", "cluster", "aws", "ec2", "instances", "cloud",
                  "stack", "cloudformation", "replicas", "docker", "container", "logs"},
    "business": {"stripe", "payment", "payments", "refund", "charge", "customer", "crm", "deal", "ticket",
                 "support", "order", "invoice", "zendesk"},
    "media": {"image", "images", "audio", "video", "transcribe", "generate", "prompt", "speech", "music", "trim"},
    "system_misc": {"shell", "command", "terminal", "python", "code", "sandbox", "weather", "forecast", "maps",
                    "directions", "location", "time"},
}


def split_identifier(name: str) -> list:
    """get_file_contents / getFileContents / get-file -> ['get', 'file', 'contents']"""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name or "")
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


def param_names(tool: dict) -> list:
    props = ((tool.get("inputSchema") or {}).get("properties") or {})
    return [p.lower() for p in props]


def tool_text(tool: dict) -> str:
    """工具侧文本模板:server 名 + 工具名 + 描述 + 参数名列表。"""
    params = " ".join(param_names(tool))
    name = " ".join(split_identifier(tool.get("name", "")))
    return f"[server] {tool.get('server', '')} [tool] {name} [desc] {tool.get('description', '')} [params] {params}"


def rule_actions(tool: dict) -> set:
    tokens = set(split_identifier(tool.get("name", "")))
    params = set(param_names(tool))
    acts = {a for a, verbs in VERB_ACTIONS.items() if tokens & verbs}
    acts |= {a for a, ps in PARAM_ACTIONS.items() if params & ps}
    ann = tool.get("annotations") or {}
    if ann.get("destructiveHint") is True and ann.get("readOnlyHint") is not True:
        acts.add("delete")
    if ann.get("readOnlyHint") is True and not (acts - {"read"}):
        acts = {"read"}
    return acts


def tool_action_features(tool: dict) -> dict:
    """动作头用的离散特征:去掉能力域关键词,避免 search/generate 这类词干扰动作判断。"""
    return {k: v for k, v in tool_features(tool).items() if not k.startswith("kw_domain=")}


def tool_features(tool: dict) -> dict:
    """离散特征,给 DictVectorizer 用。"""
    feats = {}
    for tok in split_identifier(tool.get("name", "")):
        feats[f"verb={tok}"] = 1.0
    for p in param_names(tool):
        feats[f"param={p}"] = 1.0
    for a in rule_actions(tool):
        feats[f"rule_action={a}"] = 1.0
    for d, ps in PARAM_DOMAIN_HINTS.items():
        if set(param_names(tool)) & ps:
            feats[f"param_domain={d}"] = 1.0
    words = set(split_identifier(tool.get("server", ""))) | set(split_identifier(tool.get("name", "")))
    words |= set(re.findall(r"[a-z0-9]+", (tool.get("description") or "").lower()))
    for d, kws in DOMAIN_KEYWORDS.items():
        hits = len(words & kws)
        if hits:
            feats[f"kw_domain={d}"] = float(hits)
    ann = tool.get("annotations") or {}
    for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
        val = ann.get(key)
        feats[f"ann:{key}={'missing' if val is None else bool(val)}"] = 1.0
    return feats


def risk_from(actions, domains=(), open_world=None) -> str:
    actions, domains = set(actions), set(domains)
    if actions & L3_ACTIONS:
        return "L3"
    if actions & WRITE_ACTIONS:
        return "L2"
    if open_world is None:
        open_world = bool(domains & OPEN_WORLD_DOMAINS)
    return "L1" if open_world else "L0"


def rule_risk_base(tool: dict, domains=()) -> str:
    """T-011 及以前的规则风险(只看动作 / 能力域 / openWorldHint)。

    不含 T-023 的两类信号,留给报告里的"改前"对照;线上判定用 `rule_risk`。
    """
    ann = tool.get("annotations") or {}
    open_world = ann.get("openWorldHint")
    if open_world is None:
        open_world = bool(set(domains) & OPEN_WORLD_DOMAINS)
    return risk_from(rule_actions(tool), domains, open_world)


# ---------------- 两类"风险至少 L3"的信号(T-023) ----------------
#
# 背景:`Crawler.scan`(主动扫描外部主机)和 `misc/oven_preheat`(操作实体设备)
# 在参考集里真值是 L3,但旧规则判成 L0/L1,漏设 `requires_confirmation`。
# 这两类动作作用于外部主机 / 真实世界,影响不可撤销(扫描会在对端留痕、可能触发风控;
# 设备动作改变物理状态),所以命中即把风险抬到 L3。
#
# 词表只按开发份(按函数名 6:4 切分)选,规则 = "开发份银标 L3 命中 >= 2 且
# 非 L3 误伤 = 0"(逐词统计见 reports/eval_t023.md 第 1 节);测试份和参考集只报一次。
# 纯只读的设备状态查询(`get_temperature` / `list_devices`)不升 L3:设备信号要求
# "设备名词 + 控制动词"同时出现,只读动词不会命中。
SCAN_HOST_TOKENS = frozenset({
    "scan",        # 开发份银标 L3 命中 4(XSS_Scanner.scan / network_scanner.scan / nuclei_http 等)
    "scanner",     # 开发份 L3 命中 2
    "penetration", # 开发份 L3 命中 2(misc/Attack、hacking.query 的描述)
})

DEVICE_NOUNS = frozenset({
    "oven", "appliance", "device", "light", "garage",            # 真实工具目录里出现过
    "thermostat", "lamp", "bulb", "heater", "lock",              # 通用设备名词(prior)
})
DEVICE_CONTROL_VERBS = frozenset({
    "preheat", "heat", "control", "power", "activate", "set",     # 真实工具目录里出现过
    "unlock", "dim", "toggle", "deactivate", "lock",             # 通用控制动词(prior)
})
# 分词器会把 "turn on" 拆开,所以这两个短语在归一化名字 / 原始描述上按子串匹配。
DEVICE_CONTROL_PHRASES = ("turn on", "turn off", "turn_on", "turn_off",
                          "switch on", "switch off", "power on", "power off")

# 新词的来源标注,沿用下面 KEYWORD_SOURCES 的三值口径,但工具侧没有"请求训练集",
# 对应关系改成:
#   train     真实工具目录 `data/tools_real.jsonl`(工具侧银标数据)的名字 / 描述里出现过;
#   taxonomy  taxonomy.py 的类别定义卡里出现过;
#   prior     train / taxonomy 里都没有,凭通用语言常识写的词。
# tests/test_t023_risk_signals.py::test_t023_keyword_sources_match_the_catalog 会重核一遍。
RISK_KEYWORD_SOURCES = {
    "scan": "train", "scanner": "train", "penetration": "train",
    "oven": "train", "appliance": "train", "device": "train", "light": "train",
    "garage": "train", "preheat": "train", "heat": "train", "control": "train",
    "power": "train", "activate": "train", "set": "train",
    "thermostat": "prior", "lamp": "prior", "bulb": "prior", "heater": "prior",
    "lock": "prior", "unlock": "prior", "dim": "prior", "toggle": "prior",
    "deactivate": "prior",
}


def _text_words(tool: dict) -> tuple:
    """工具名 token + 描述词 + 原始描述;名字走 split_identifier(拆 camelCase)。"""
    name_tokens = set(split_identifier(tool.get("name", "")))
    desc = (tool.get("description") or "").lower()
    return name_tokens, set(re.findall(r"[a-z0-9]+", desc)), desc


def risk_signals(tool: dict) -> set:
    """命中的 T-023 风险信号名(scan_host / device_control),空集表示没命中。"""
    name_tokens, desc_words, desc = _text_words(tool)
    words = name_tokens | desc_words
    signals = set()
    if words & SCAN_HOST_TOKENS:
        signals.add("scan_host")
    # 设备动作要求"名词 + 控制动词"同时出现,避免把只读状态查询当成控制。
    if (words & DEVICE_NOUNS):
        joined = "_".join(split_identifier(tool.get("name", "")))
        if (words & DEVICE_CONTROL_VERBS) or any(p in joined or p in desc
                                                 for p in DEVICE_CONTROL_PHRASES):
            signals.add("device_control")
    return signals


def rule_risk(tool: dict, domains=()) -> str:
    """规则风险:在 `rule_risk_base` 之上叠加 T-023 的两类信号,命中即 L3。"""
    if risk_signals(tool):
        return "L3"
    return rule_risk_base(tool, domains)


def final_risk(tool: dict, model_actions, domains) -> str:
    """规则与模型不一致时取更高的一档。"""
    ann = tool.get("annotations") or {}
    model = risk_from(model_actions, domains, ann.get("openWorldHint"))
    return max_risk(model, rule_risk(tool, domains))


# ---------------- 请求侧离散特征 ----------------

MULTI_STEP_CUES = ("然后", "再", "之后", "后在", "后把", "并且", "同时", "接着", "如果", "就", "有问题", "，", ",")
ENTITY_PATTERNS = {
    "url": r"https?://|www\.|网址|链接",
    "file_ext": r"\.(txt|md|csv|pdf|docx?|xlsx?|py|ts|js|json|ya?ml)\b",
    "email_addr": r"[\w.]+@[\w.]+",
    "channel": r"#[\w-]+|频道|群",
    "issue_ref": r"#\d+|PR|pull request|issue",
    "sql_like": r"\b(select|update|delete|insert)\b|表\b|字段",
    "time_ref": r"明天|今天|下周|周[一二三四五六日末]|[上下]午|点",
    "money": r"退款|付款|支付|\d+ ?(块|元)",
}


def request_features(req) -> dict:
    """请求侧离散特征:多步骤连接词 + 常见实体模式。req 可以是 dict 或字符串。"""
    text = req["text"] if isinstance(req, dict) else req
    feats = {}
    cues = sum(text.count(c) for c in MULTI_STEP_CUES)
    if cues:
        feats["multi_step_cues"] = float(min(cues, 3))
    for name, pat in ENTITY_PATTERNS.items():
        if re.search(pat, text, re.I):
            feats[f"entity={name}"] = 1.0
    return feats


# ---------------- 请求侧"无外部对象"信号 ----------------

# none 请求的共性:在问知识,或者让你产出文本。
# 词表来源不统一,逐个词的来源见下面的 KEYWORD_SOURCES:
# 42 个词在训练集里出现过,1 个来自 taxonomy.py,60 个是凭通用语言常识写的(prior);
# 其中有 13 个 prior 词只在 gold 出现、训练集里 0 次(GOLD_ONLY_KEYWORDS),
# 交叉验证衡量不到它们,拿结论时必须看消融。
NONE_INTENT_KEYWORDS = {
    "explain": ("解释", "介绍一下", "简介", "什么是", "是什么", "什么意思", "怎么理解", "为什么",
                "原理", "概念", "区别", "优缺点", "对比", "比较", "几种", "区别在哪", "如何",
                "explain", "what is", "what are", "what's", "difference", "how do", "how does",
                "why", "compare", "pros and cons"),
    "authoring": ("写一首", "写一段", "写个", "写一封", "帮我写", "帮我拟", "拟一", "起个", "想个",
                  "取个", "润色", "改写", "扩写", "缩写", "翻成", "翻译", "翻一下", "总结一下这",
                  "压缩成", "推荐", "建议", "列个", "讲个", "自我介绍", "算一下", "等于几",
                  "write", "draft", "polish", "rewrite", "translate", "summarize", "recommend"),
    "chitchat": ("你好", "谢谢", "早上好", "晚上好", "晚安", "辛苦了", "你觉得", "说说你", "看法",
                 "hello", "thanks"),
}

# 指向"用户自己的对象"的词。命中任何一个,就不再算"无外部对象":
# 「解释一下我仓库里的 setup.py」和「解释一下什么是闭包」句式像,但前者要读数据。
OBJECT_HINTS = ("我的", "我仓库", "我这", "我邮箱", "我网盘", "我们", "咱们", "公司", "线上",
                "prod", "刚才", "昨天", "前天", "上次", "那个", "这个", "群里", "频道", "邮箱",
                "日程", "待办", "工单", "客户", "数据库", "服务器", "集群", "录音", "附件",
                "my ", "our ", "recent", "last ")


def request_need_features(req, drop_gold_only: bool = False) -> dict:
    """只在 need 头上叠加的"无外部对象"特征。req 可以是 dict 或字符串。

    分成两步判断而不是单纯加词表:先看有没有知识/写作意图,再看有没有指向
    用户自有对象的词。意图命中且没有对象词时给 no_object_intent,才是强 none 信号。

    drop_gold_only=True 时跳过 GOLD_ONLY_KEYWORDS 里的词,用于消融
    (这些词在训练集里一次都没出现,交叉验证衡量不到它们)。
    """
    text = req["text"] if isinstance(req, dict) else req
    low = text.lower()

    def usable(words):
        return [w for w in words if not (drop_gold_only and w in GOLD_ONLY_KEYWORDS)]

    feats = {}
    intents = []
    for name, kws in NONE_INTENT_KEYWORDS.items():
        hits = sum(1 for kw in usable(kws) if kw in low)
        if hits:
            feats[f"none_intent={name}"] = float(min(hits, 2))
            intents.append(name)
    has_object = any(hint in low for hint in usable(OBJECT_HINTS))
    feats["has_object_hint"] = 1.0 if has_object else 0.0
    if intents and not has_object:
        feats["no_object_intent"] = 1.0
    return feats


def request_need_features_ablated(req) -> dict:
    """消融版本:去掉只在 gold 出现、训练集 0 次的词。"""
    return request_need_features(req, drop_gold_only=True)


# ---------------- 词来源标注与泄漏检查 ----------------

# 词来源只有三种,每个词必须恰好属于一组。tests/test_mcpcls.py 的
# test_keyword_sources_match_the_data 会按下面的规则重新核对一遍:
#   train     请求训练集里出现过(调参集合,合法来源)
#   taxonomy  taxonomy.py 的类别定义卡里出现过
#   prior     train 和 taxonomy 里都没有,凭通用语言常识写的词
KEYWORD_SOURCES = {
    "train": (
        "解释", "什么是", "是什么", "怎么理解", "为什么", "区别", "优缺点", "比较", "几种", "如何", "写个", "起个", "扩写", "翻译",
        "推荐", "列个", "讲个", "算一下", "等于几", "draft", "谢谢", "早上好", "hello", "我的", "我邮箱", "我们", "公司",
        "线上", "刚才", "昨天", "上次", "那个", "这个", "群里", "频道", "邮箱", "待办", "工单", "客户", "数据库", "服务器",
        "录音",
    ),
    "taxonomy": (
        "日程",
    ),
    "prior": (
        "介绍一下", "简介", "什么意思", "原理", "概念", "对比", "区别在哪", "explain", "what is", "what are",
        "what's", "difference", "how do", "how does", "why", "compare", "pros and cons", "写一首",
        "写一段", "写一封", "帮我写", "帮我拟", "拟一", "想个", "取个", "润色", "改写", "缩写", "翻成", "翻一下", "总结一下这",
        "压缩成", "建议", "自我介绍", "write", "polish", "rewrite", "translate", "summarize", "recommend",
        "你好", "晚上好", "晚安", "辛苦了", "你觉得", "说说你", "看法", "thanks", "我仓库", "我这", "我网盘", "咱们", "prod",
        "前天", "集群", "附件", "my ", "our ", "recent", "last ",
    ),
}

# 训练集 0 次、gold 至少 1 次的词:交叉验证的任何一折都不会触发它们,
# 所以 gold 与探针上的增益里混着它们,不能当作独立证据。
# test_gold_only_keywords_are_pinned 会重算这个集合,新增这类词必须显式改这里。
GOLD_ONLY_KEYWORDS = frozenset({
    "介绍一下", "概念", "写一首", "写一段", "写一封", "帮我写", "润色", "自我介绍",
    "说说你", "看法", "我这", "prod", "日程",
})


def all_keywords() -> list:
    """两个词表的全部词,返回 [(组名, 词)]。"""
    out = [(group, w) for group, kws in NONE_INTENT_KEYWORDS.items() for w in kws]
    out += [("OBJECT_HINTS", w) for w in OBJECT_HINTS]
    return out


def source_of(word: str):
    """词的来源;没标注过返回 None。"""
    for src, words in KEYWORD_SOURCES.items():
        if word in words:
            return src
    return None


def taxonomy_vocabulary() -> str:
    """taxonomy.py 里字符串字面量拼成的文本,用于核对 taxonomy 来源。"""
    from . import taxonomy

    src = Path(taxonomy.__file__).read_text(encoding="utf-8")
    return " ".join(re.findall(r'"([^"]*)"', src)).lower()


def keyword_source_problems(train_text: str, taxonomy_text: str) -> list:
    """核对来源标注与数据是否自洽,返回问题列表(空表示没问题)。"""
    problems = []
    declared = {w: src for src, words in KEYWORD_SOURCES.items() for w in words}
    for group, word in all_keywords():
        src = declared.get(word)
        if src is None:
            problems.append(f"{group}/{word}: 未标注来源")
            continue
        in_train = word.lower() in train_text
        in_taxonomy = word.lower() in taxonomy_text
        if src == "train" and not in_train:
            problems.append(f"{word}: 标成 train,但训练集里没有")
        elif src == "taxonomy" and not in_taxonomy:
            problems.append(f"{word}: 标成 taxonomy,但 taxonomy.py 里没有")
        elif src == "prior" and in_train:
            problems.append(f"{word}: 标成 prior,但训练集里出现过(应标 train)")
    return problems
