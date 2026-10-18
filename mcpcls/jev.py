"""Jev(TypeSafe AI 的决策模型)客户端,用作低置信度兜底分类器。

Jev 不输出向量,不能当编码器用:它回答有类型的问题——
`noul`(是非题,返回为真的概率)、`choice`(选择题,返回 choice / 各选项概率 / confidence)、
`score`(打分题)。所以这里把它当"能问出结构化答案的分类器",一次请求问一组问题,
再把答案解析成 need / domains / actions,挂在 `Router.llm_fallback` 这个位置上。

调用经 OpenRouter 的 `/api/alpha/decisions`,用标准库 urllib,不加新依赖。
延迟实测 0.6s ~ 32s(全量 p95 约 16s),所以默认超时 30 秒、重试 1 次;
失败时由 Router 保留本地结果,不向上抛。

除了 Jev 自己,通用聊天模型(见 `llm_or.py`)也回答同一组问题,
所以问题构造(`request_questions`)与答案解析(`answers_to_labels`)是两者共用的。
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .taxonomy import ACTIONS, DOMAINS, NEED, ACTION_IDS, DOMAIN_IDS, NEED_IDS

# 默认走 commandcode 网关(gu 2026-10-10 指定;OpenRouter 余额为 0,Jev 返回 402)。
# 换服务商:JEV_BASE_URL 指向一个 TypeSafe 原生接口的根路径(请求发到 <根>/systemone),
# 例如 OpenRouter 是 https://openrouter.ai/api/v1;JEV_MODEL 覆盖模型名;
# key 依次取 JEV_API_KEY、OPENROUTER_API_KEY。
JEV_URL = "https://api.commandcode.ai/provider/v1/systemone"
DEFAULT_JEV_MODEL = "typesafe/jev"
# T-004~T-006 用的是 OpenRouter 上的 jev-1.13,它的缓存文件名是 "jev";换了模型就换缓存名,免得新旧答案混在一起
LEGACY_JEV_MODEL = "typesafe/jev-1.13"
USER_AGENT = "mcpcls/0.1"
MAX_QUESTIONS_PER_CALL = 20   # Jev 一次最多 20 题(超了返回 400 "at most 20 questions per call")   # 有的网关(Cloudflare)会拦下 urllib 默认的 Python-urllib UA


def jev_endpoint(base_url=None) -> str:
    base = base_url or os.environ.get("JEV_BASE_URL")
    return f"{base.rstrip('/')}/systemone" if base else JEV_URL

# need 说明的三个版本:v1 是 T-004 的原文,v2 是 T-005 改写版,v3 是 T-009 改写版。
# v1 / v2 的文字一个字不改;三个版本的字面量见 _NEED_V2 / _NEED_V3。
NEED_VERSIONS = ("v1", "v2", "v3")
# 客户端名字:jev 走决策模型,haiku 走聊天接口(见 llm_or.py),
# codebuddy 走本机命令行(见 codebuddy_client.py,不联网、不花钱)
CLIENT_NAMES = ("jev", "haiku", "codebuddy")


class JevUnavailable(Exception):
    """没法用 Jev 时抛出(构造时就报,不等到第一次调用)。"""


class JevError(Exception):
    """一次调用失败(超时、HTTP 错误、响应里没有 answers / 输出不是 JSON)。"""


class RateLimited(JevError):
    """被限流(HTTP 429 / codebuddy 命令行 429):重试只会浪费时间,调用方应立刻停。"""


class PaymentRequired(JevError):
    """欠费(HTTP 402,OpenRouter 余额不足):同上,立刻停,重试没有意义。"""


def fatal_http_error(e) -> "JevError | None":
    """把限流(429) / 欠费(402)的 HTTPError 换成不重试的错误;其他情况返回 None。

    Jev 与聊天客户端共用这一份判断,两类错误都不该走 `retries` 重试。
    """
    if isinstance(e, urllib.error.HTTPError) and e.code in (402, 429):
        cls = RateLimited if e.code == 429 else PaymentRequired
        return cls(f"HTTP {e.code}:{str(e)[:200]}")
    return None


@dataclass
class JevCall:
    """每次调用的记录。error 是简短信息,绝不能包含 key。

    token 用量(T-017):推理型聊天模型的 usage 里有 prompt_tokens /
    completion_tokens / reasoning_tokens,记下来估算费用。三个字段都排在最后并给默认值,
    所以老的按位置构造(比如 `FakeJevClient`、`codebuddy_client`)不用改。
    `input_tokens` 是 Jev 自己的用法字段,保留不动。
    """
    latency_ms: float
    cost: float             # usage.cost,缺失记 0.0
    input_tokens: int
    ok: bool
    error: str = ""
    prompt_tokens: int = 0       # usage.prompt_tokens(与 input_tokens 语义相同,聊天模型单独记)
    completion_tokens: int = 0   # usage.completion_tokens(推理模型的推理 token 也算在里面)
    reasoning_tokens: int = 0    # usage.completion_tokens_details.reasoning_tokens


def redact(msg, api_key) -> str:
    """把 key 从错误信息里抹掉。Jev 与聊天客户端共用这一份。"""
    text = str(msg)[:300]
    return text.replace(api_key, "***") if api_key else text


def post_json(url: str, body: bytes, api_key: str, timeout: float, extra_headers=None) -> dict:
    """POST 一个 JSON 请求体,返回解析后的响应。key 只出现在请求头里。"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "User-Agent": USER_AGENT}
    headers.update(extra_headers or {})
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


class JevClient:
    source = "jev_fallback"        # 结果来源标记,给 Router 用
    error_source = "jev_error"     # 调用失败时的来源标记

    def __init__(self, model=None, timeout=30.0, retries=1, api_key=None, base_url=None):
        """key 依次取 api_key 参数、JEV_API_KEY、OPENROUTER_API_KEY;都没有就直接抛 JevUnavailable。

        model / base_url 为 None 时读 JEV_MODEL / JEV_BASE_URL,再没有就用 OpenRouter 的默认值。
        """
        self.model = model or os.environ.get("JEV_MODEL") or DEFAULT_JEV_MODEL
        self.url = jev_endpoint(base_url)
        self.timeout = timeout
        self.retries = retries
        self.api_key = api_key or os.environ.get("JEV_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.calls: list = []
        if not self.api_key:
            raise JevUnavailable("没有 Jev 的 key:传 api_key 参数,或设置 JEV_API_KEY / OPENROUTER_API_KEY")

    def decide(self, state: str, questions: dict) -> dict:
        """问 Jev 一组问题,返回响应里的 answers;失败抛 JevError。

        retries=1 表示最多试 2 次。无论成败,`calls` 里只追加一条(重试后的最终结果);
        `latency_ms` 从第一次尝试算起,重试的等待时间也算在内,否则端到端延迟会被低估。
        """
        body = json.dumps({"model": self.model, "state": state, "questions": questions},
                          ensure_ascii=False).encode("utf-8")
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                answers, usage = self._post(body)
            except Exception as e:      # 超时、连接错误、HTTP 非 200、响应缺 answers
                fatal = fatal_http_error(e)
                if fatal is not None:    # 限流 / 欠费不重试,记一次失败再抛
                    self.calls.append(JevCall((time.perf_counter() - t0) * 1000.0, 0.0, 0, False,
                                              self._safe(fatal)))
                    raise fatal from None
                if attempt < self.retries:
                    continue
                self.calls.append(JevCall((time.perf_counter() - t0) * 1000.0, 0.0, 0, False,
                                          self._safe(e)))
                raise JevError(self._safe(e)) from None
            self.calls.append(JevCall((time.perf_counter() - t0) * 1000.0,
                                      float(usage.get("cost") or 0.0),
                                      int(usage.get("input_tokens") or 0), True))
            return answers

    @property
    def cache_name(self) -> str:
        """缓存用的客户端名:旧的 jev-1.13 仍是 "jev",其他模型是 "jev@<模型>"(斜杠换成横线)。"""
        if self.model == LEGACY_JEV_MODEL:
            return "jev"
        return f"jev@{self.model.replace('/', '-')}"

    def _post(self, body: bytes):
        """题目超过 MAX_QUESTIONS_PER_CALL 时拆成几次调用,答案合并;usage 的 cost / token 相加。"""
        payload = json.loads(body.decode("utf-8"))
        names = list(payload["questions"])
        if len(names) <= MAX_QUESTIONS_PER_CALL:
            return self._post_one(body)
        answers, usage = {}, {"cost": 0.0, "input_tokens": 0}
        for i in range(0, len(names), MAX_QUESTIONS_PER_CALL):
            part = dict(payload, questions={n: payload["questions"][n]
                                            for n in names[i:i + MAX_QUESTIONS_PER_CALL]})
            a, u = self._post_one(json.dumps(part, ensure_ascii=False).encode("utf-8"))
            answers.update(a)
            usage["cost"] += float(u.get("cost") or 0.0)
            usage["input_tokens"] += int(u.get("input_tokens") or 0)
        return answers, usage

    def _post_one(self, body: bytes):
        data = post_json(self.url, body, self.api_key, self.timeout)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise JevError(f"响应里没有 answers 字段:{str(data)[:200]}")
        return answers, data.get("usage") or {}

    def _safe(self, e) -> str:
        return redact(f"{type(e).__name__}: {e}", self.api_key)


class FakeJevClient:
    """测试用的假客户端:同样的 decide / calls 接口,不联网。"""
    def __init__(self, answers_fn, latency_ms=1.0, fail=False, cost=1.84e-5, input_tokens=438):
        self.answers_fn = answers_fn        # f(state, questions) -> answers
        self.latency_ms = latency_ms
        self.fail = fail                    # True 时 decide 抛 JevError
        self.cost = cost
        self.input_tokens = input_tokens
        self.calls: list = []
        self.states: list = []              # 每次 decide 的入参,用来断言"不该问的没问"

    def decide(self, state: str, questions: dict) -> dict:
        self.states.append(state)
        if self.fail:
            self.calls.append(JevCall(self.latency_ms, 0.0, 0, False, "fake: 调用失败"))
            raise JevError("fake: 调用失败")
        answers = self.answers_fn(state, questions)
        self.calls.append(JevCall(self.latency_ms, self.cost, self.input_tokens, True))
        return answers


def _noul_question(cat, subject: str, extra: str = "") -> dict:
    """是非题:这条请求是否涉及某个能力域 / 包含某个动作。"""
    return {
        "type": "noul",
        "instructions": f"判断这条用户请求是否{subject}「{cat.name}」。判定标准:{cat.definition}"
                        + (f"边界:{cat.boundary}" if cat.boundary else "") + extra,
        "criteria": {"true": cat.definition, "false": f"与{cat.name}无关"},
    }


# v2 的 need 说明:文字由 Claude 在没看 gold 逐条错例的前提下写定,逐字照抄任务卡,不要改写
_NEED_V2 = {
    "instructions": (
        "判断这条用户请求能否只靠语言模型自身的知识完成。只要需要读取、查询或改动用户的外部数据或系统"
        "（邮件、消息、日程、任务、文件、数据库、网页、浏览器、代码仓库、云服务、订单或客户记录、音视频等），"
        "就需要工具。只按请求本身判断，不要替用户设想没说出来的需求。"),
    "criteria": {
        "none": "不需要任何外部工具：解释概念、知识问答、写作、翻译、润色、出主意、闲聊或表达观点，"
                "模型凭自身知识就能直接回答，不涉及读取或改动用户的任何外部数据。",
        "single": "需要一个外部工具完成一件事：请求里有明确的操作对象，例如发一封邮件或消息、查或改日程、"
                  "建一个任务、读写某个文件、查数据库、搜网页、操作网页、查日志或部署、改订单或客户记录。"
                  "即使请求很短、语气很随意，只要要对外部数据或系统做一件事，就是 single。",
        "multi": "需要先后或组合使用两个及以上不同的工具或系统，例如先查日程再发邮件、先转写录音再存成文件、"
                 "先查 issue 再建待办。",
    },
}


# v3 的 need 说明(T-009):和 v2 的差别是明确写清"实时/最新/会变化的外部信息"需要工具,
# 并点出数学计算、换算、解释概念、写作翻译不需要工具。文字逐字照抄任务卡,不要改写。
_NEED_V3 = {
    "instructions": (
        "判断这条用户请求能否只靠语言模型自身的知识完成。只要需要读取、查询或改动用户的外部数据或系统，"
        "或者需要实时、最新、会变化的外部信息（价格、汇率、天气、新闻、榜单、库存、营业时间、航班车次、"
        "比分等），就需要工具。数学计算、换算、解释概念、写作翻译不需要工具。只按请求本身判断，"
        "不要替用户设想没说出来的需求。"),
    "criteria": {
        "none": "不需要任何外部工具：解释概念、常识或知识问答、数学计算与单位换算、写作、翻译、润色、"
                "出主意、闲聊或表达观点，模型凭自身知识就能直接回答，不依赖实时信息，"
                "也不读取或改动用户的任何外部数据。",
        "single": "需要一个外部工具完成一件事：要对外部数据或系统做一件事（发消息、查改日程、读写文件、"
                  "查数据库、操作网页、查日志或部署、改订单或客户记录），或者需要查一次实时、最新的外部信息"
                  "（价格、天气、新闻、榜单、库存、航班等）。即使请求很短、语气很随意，也是 single。",
        "multi": "需要先后或组合使用两个及以上不同的工具或系统，例如先查日程再发邮件、先查天气再订车、"
                 "先转写录音再存成文件。",
    },
}


def need_question(need_version: str = "v1") -> dict:
    """只含 need 这一题,供组合策略与对照实验单独复用。"""
    if need_version == "v1":
        return {
            "type": "choice",
            "instructions": "判断这条用户请求是否需要调用外部工具(MCP 工具)才能完成。"
                            "只按请求本身判断,不要替用户设想没说出来的需求。",
            "criteria": {c.id: f"{c.name}。{c.definition}" for c in NEED},
        }
    if need_version == "v2":
        return {"type": "choice", "instructions": _NEED_V2["instructions"],
                "criteria": dict(_NEED_V2["criteria"])}
    if need_version == "v3":
        return {"type": "choice", "instructions": _NEED_V3["instructions"],
                "criteria": dict(_NEED_V3["criteria"])}
    raise ValueError(f"need_version 只能是 {NEED_VERSIONS} 之一,收到 {need_version!r}")


def request_questions(need_version: str = "v1") -> dict:
    """一条请求要问的全部问题:1 个 need 选择题 + 每个能力域一题 + 6 个动作是非题。

    题数跟着 taxonomy 走(T-009 起是 14 个域,共 21 题),所以改 taxonomy 不用改这里。
    说明文字全部从 taxonomy 的 name / definition / boundary 生成,不另写一份类别定义,
    免得两边说法不一致时不知道以哪个为准。need 那一题有 v1 / v2 / v3 三个版本(见 `need_question`)。
    """
    questions = {"need": need_question(need_version)}
    for c in DOMAINS:
        questions[f"domain_{c.id}"] = _noul_question(c, "需要操作或查询")
    for c in ACTIONS:
        questions[f"action_{c.id}"] = _noul_question(c, "包含", "一个请求可以同时命中多个动作。")
    return questions


def tool_questions() -> dict:
    """一个工具要问的全部问题:每个能力域一题 + 6 个动作是非题,共 20 题。

    问法和 `request_questions` 一样从 taxonomy 生成(不另写类别定义),只是把主语从
    "这条用户请求"换成"这个工具",问"是否提供 / 操作这一项"。题数正好是
    `MAX_QUESTIONS_PER_CALL`,所以每个工具一次调用就问完,不用像请求侧那样拆两次。
    """
    questions = {}
    for c in DOMAINS:
        questions[f"domain_{c.id}"] = {
            "type": "noul",
            "instructions": f"判断这个工具是否提供或操作「{c.name}」。判定标准:{c.definition}"
                            + (f"边界:{c.boundary}" if c.boundary else ""),
            "criteria": {"true": c.definition, "false": f"与{c.name}无关"},
        }
    for c in ACTIONS:
        questions[f"action_{c.id}"] = {
            "type": "noul",
            "instructions": f"判断这个工具是否会{c.name}。判定标准:{c.definition}"
                            + "一个工具可以同时命中多个动作。",
            "criteria": {"true": c.definition, "false": f"不会{c.name}"},
        }
    return questions


def _picked(answers: dict, prefix: str, ids, threshold: float) -> list:
    """过阈值的全部选中;一个都没过时取概率最高的那一个。"""
    probs = [(i, float((answers.get(f"{prefix}_{i}") or {}).get("noul") or 0.0)) for i in ids]
    hit = [i for i, p in probs if p > threshold]
    if hit:
        return hit
    return [max(probs, key=lambda x: x[1])[0]] if probs else []


def answers_to_labels(answers: dict, noul_threshold: float = 0.5) -> dict:
    """把 Jev 的 answers 解析成 {"need", "domains", "actions", "confidence"}。

    need 判成 none 时 domains / actions 一律置空;need 非 none 但没有任何域过阈值时,
    取概率最高的一个域(动作同理),否则路由会拿到空域、检索退化成不过滤。
    """
    need_answer = answers.get("need") or {}
    need = need_answer.get("choice") or "single"
    if need not in NEED_IDS:
        need = "single"
    confidence = float(need_answer.get("confidence") or 0.0)
    if need == "none":
        return {"need": need, "domains": [], "actions": [], "confidence": confidence}
    return {"need": need,
            "domains": _picked(answers, "domain", DOMAIN_IDS, noul_threshold),
            "actions": _picked(answers, "action", ACTION_IDS, noul_threshold),
            "confidence": confidence}


def jev_fallback(client, noul_threshold: float = 0.5, need_version: str = "v1"):
    """给 Router 用的兜底函数 f(text) -> dict(need, domains, actions)。

    带 `source` 属性(结果来源,默认 jev_fallback,区别于 llm_fallback)与 `error_source`
    (调用失败时的来源,默认 jev_error);两个客户端(Haiku)各自的类属性会覆盖它们。
    """
    def fallback(text: str) -> dict:
        labels = answers_to_labels(client.decide(text, request_questions(need_version)), noul_threshold)
        return {"need": labels["need"], "domains": labels["domains"], "actions": labels["actions"]}

    fallback.source = getattr(client, "source", "jev_fallback")
    fallback.error_source = getattr(client, "error_source", "jev_error")
    fallback.client = client
    return fallback


def make_client(name, **kw):
    """工厂:按名字造客户端。`jev` 走决策模型,`haiku` 走聊天接口(见 `llm_or.py`)。

    返回的客户端带 `source` / `error_source`,交给 `jev_fallback` 就能得到对应来源标记的兜底函数;
    离线对比要按 need 版本换问题集,所以这里返回客户端本身而不是兜底函数。
    """
    if name == "jev":
        return JevClient(**kw)
    if name == "haiku":
        from .llm_or import OpenRouterChatClient
        return OpenRouterChatClient(**kw)
    if name == "codebuddy":
        from .codebuddy_client import CodeBuddyClient
        return CodeBuddyClient(**kw)
    raise ValueError(f"客户端只能是 {CLIENT_NAMES} 之一,收到 {name!r}")
