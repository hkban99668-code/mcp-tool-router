"""OpenRouter 聊天接口客户端:让通用 LLM(Haiku)回答和 Jev 完全相同的问题。

Jev 要替代的是"通用 LLM 兜底",所以要和它在同一组问题上比:这里把 `request_questions`
渲染成一个 prompt,让模型只输出一个 JSON,再按 Jev 的 answers 形状解析出来,
这样 `answers_to_labels` / `apply_strategy` / `jev_fallback` 全都不用改就能直接用。

调用记录、异常类型、key 的处理全部复用 `jev.py`,不复制第二份。
"""

import json
import re
import os
import time

from .jev import (JevCall, JevError, JevUnavailable, fatal_http_error, post_json, redact)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# T-005 时 OpenRouter 上的模型名;它的缓存文件名是 "haiku"
LEGACY_CHAT_MODEL = "anthropic/claude-haiku-5.5"
# 默认走 commandcode 网关的 Anthropic messages 接口(OpenRouter 余额为 0,gu 2026-10-10 指定 commandcode)。
# HAIKU_BASE_URL / HAIKU_MODEL / HAIKU_API_FORMAT(anthropic|openai) 可覆盖;
# key 依次取 HAIKU_API_KEY、JEV_API_KEY、OPENROUTER_API_KEY。
MESSAGES_URL = "https://api.commandcode.ai/provider/v1/messages"
DEFAULT_CHAT_MODEL = "claude-haiku-5-5"
ANTHROPIC_VERSION = "2023-06-01"

_JSON_RULES = ("只输出一个 JSON 对象,key 是题目名,value 是答案:\n"
               "- choice 题:value 是选项 id 字符串,必须是 criteria 里列出的一个\n"
               "- noul 题:value 是 true 或 false\n"
               "- text 题:value 是一段文本")


def render_questions(questions: dict) -> str:
    """只渲染题目说明:题目名、类型、instructions、criteria 原样列出。

    批量客户端(见 `codebuddy_client.py`)要"题目说明只出现一次 + 列出多条请求",
    所以把这一段单独拆出来,和单条的 `render_prompt` 共用同一份文字。
    """
    lines = []
    for name, q in questions.items():
        lines.append(f"## {name}({q.get('type')})")
        lines.append(f"instructions:{q.get('instructions', '')}")
        lines.append("criteria:")
        for key, text in (q.get("criteria") or {}).items():
            lines.append(f"  - {key}:{text}")
        lines.append("")
    return "\n".join(lines)


def render_prompt(state: str, questions: dict) -> str:
    """把一组问题渲染成聊天消息:题目说明 + 一条请求。"""
    return "\n".join(["下面是用户请求:", state, "", "请逐题回答。" + _JSON_RULES, "",
                      render_questions(questions)])


def _strip_fences(text: str) -> str:
    """模型偶尔会把 JSON 包在 ``` 里,先剥掉再解析。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    return t.strip()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _usage_tokens(usage: dict) -> tuple:
    """从 usage 里取 (prompt, completion, reasoning) 三个 token 数;缺失的记 0。

    OpenAI 兼容端点用 prompt_tokens / completion_tokens,推理 token 在
    completion_tokens_details.reasoning_tokens;Anthropic 用 input_tokens / output_tokens。
    """
    det = usage.get("completion_tokens_details") or {}
    return (int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            int(det.get("reasoning_tokens") or 0))


def parse_answers_obj(obj, questions: dict) -> dict:
    """校验一个已经解析好的答案对象,转成 Jev 形状的 answers;缺题或非法就抛 JevError。

    批量客户端对每一条单独调它,所以校验规则和单条路径一定是同一套。
    """
    if not isinstance(obj, dict):
        raise JevError(f"答案不是 JSON 对象:{str(obj)[:200]}")
    # 模型偶尔照抄 prompt 里的小标题,把 key 写成 "translation(text)" / "need(choice)",去掉括号后缀再对题
    obj = {re.sub(r"[(（][^)）]*[)）]\s*$", "", str(k)).strip(): v for k, v in obj.items()}
    answers = {}
    for name, q in questions.items():
        if name not in obj:
            raise JevError(f"答案里缺题目 {name}:{str(obj)[:200]}")
        value = obj[name]
        if q.get("type") == "choice":
            ids = list((q.get("criteria") or {}).keys())
            if value not in ids:
                raise JevError(f"{name} 的答案 {value!r} 不在选项 {ids} 里")
            answers[name] = {"type": "choice", "choice": value,
                             "probabilities": {i: (1.0 if i == value else 0.0) for i in ids},
                             "confidence": 1.0}
        elif q.get("type") == "text":
            # 自由文本题(翻译用):原样带回去,不做校验
            answers[name] = {"type": "text", "text": "" if value is None else str(value)}
        else:
            answers[name] = {"type": "noul", "noul": 1.0 if _as_bool(value) else 0.0}
    return answers


def parse_answers(content: str, questions: dict) -> dict:
    """把模型输出的 JSON 解析成 Jev 形状的 answers;解析不了或缺题就抛 JevError。"""
    try:
        obj = json.loads(_strip_fences(content))
    except Exception:
        raise JevError(f"模型输出不是 JSON:{content[:200]}") from None
    return parse_answers_obj(obj, questions)


class OpenRouterChatClient:
    source = "haiku_fallback"
    error_source = "haiku_error"

    def __init__(self, model=None, timeout=30.0, retries=1, api_key=None, base_url=None,
                 api_format=None):
        """与 `JevClient` 相同的 key 规则:构造时就检查,没有就抛 JevUnavailable。

        api_format="anthropic" 走 <根>/messages(默认,commandcode);"openai" 走 <根>/chat/completions。
        """
        self.model = model or os.environ.get("HAIKU_MODEL") or DEFAULT_CHAT_MODEL
        base = base_url or os.environ.get("HAIKU_BASE_URL")
        fmt = api_format or os.environ.get("HAIKU_API_FORMAT")
        if base:
            fmt = fmt or ("openai" if "openrouter" in base else "anthropic")
            self.url = base.rstrip("/") + ("/messages" if fmt == "anthropic" else "/chat/completions")
        else:
            fmt = fmt or "anthropic"
            self.url = MESSAGES_URL if fmt == "anthropic" else CHAT_URL
        self.api_format = fmt
        self.timeout = timeout
        self.retries = retries
        self.api_key = (api_key or os.environ.get("HAIKU_API_KEY") or os.environ.get("JEV_API_KEY")
                        or os.environ.get("OPENROUTER_API_KEY"))
        self.calls: list = []
        if not self.api_key:
            raise JevUnavailable("没有 Haiku 的 key:传 api_key 参数,或设置 HAIKU_API_KEY / JEV_API_KEY / "
                                 "OPENROUTER_API_KEY")

    @property
    def cache_name(self) -> str:
        """缓存用的客户端名。

        Haiku 的旧名字一个都不动(T-005 的 OpenRouter 答案仍是 "haiku",commandcode 的
        claude-haiku-5-5 仍是 "haiku@claude-haiku-5-5"),旧缓存照常命中。
        非 Haiku 的聊天模型(T-017 的国产模型)前缀改成 "chat@",免得被误认为 Haiku。
        """
        if self.model == LEGACY_CHAT_MODEL:
            return "haiku"
        if "haiku" in self.model.lower():
            return f"haiku@{self.model.replace('/', '-')}"
        return f"chat@{self.model.replace('/', '-')}"

    def _body(self, prompt: str) -> bytes:
        if self.api_format == "anthropic":
            # claude-haiku-5-5 不再接受 temperature(400 "temperature is deprecated"),所以不传
            body = {"model": self.model, "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}]}
        else:
            # T-017:推理型国产模型(deepseek / glm / qwen / mimo 等)会先输出一批推理 token,
            # max_tokens 太小会把 JSON 截断导致解析失败;T-017 实测 qwen 的 reasoning 会到 ~2400,
            # 给足 8192。("usage": {"include": True} 是 OpenRouter 用来返回 cost 的参数;T-017 实测
            # commandcode 的 openai 端点也接受它(200,只多返回 token 数、不返回 cost),所以照旧带上。)
            body = {"model": self.model, "max_tokens": 8192,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "response_format": {"type": "json_object"},
                    "usage": {"include": True}}
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    def decide(self, state: str, questions: dict) -> dict:
        """问一组问题,返回 Jev 形状的 answers;失败抛 JevError。

        `latency_ms` 与 `JevClient` 一样从第一次尝试算起,重试的耗时也算在内。
        """
        body = self._body(render_prompt(state, questions))
        extra = {"anthropic-version": ANTHROPIC_VERSION} if self.api_format == "anthropic" else None
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                data = post_json(self.url, body, self.api_key, self.timeout, extra)
                answers = parse_answers(_content(data), questions)
            except Exception as e:      # 超时、HTTP 错误、输出不是 JSON、缺题
                fatal = fatal_http_error(e)
                if fatal is not None:    # 限流 / 欠费不重试,立刻向上抛
                    raise fatal from None
                if attempt < self.retries:
                    continue
                self.calls.append(JevCall((time.perf_counter() - t0) * 1000.0, 0.0, 0, False,
                                          redact(f"{type(e).__name__}: {e}", self.api_key)))
                raise JevError(redact(f"{type(e).__name__}: {e}", self.api_key)) from None
            usage = data.get("usage") or {}
            prompt_tok, completion_tok, reasoning_tok = _usage_tokens(usage)
            self.calls.append(JevCall((time.perf_counter() - t0) * 1000.0,
                                      float(usage.get("cost") or 0.0),
                                      prompt_tok, True,
                                      prompt_tokens=prompt_tok,
                                      completion_tokens=completion_tok,
                                      reasoning_tokens=reasoning_tok))
            return answers


def _content(data: dict) -> str:
    """取模型回复文本:Anthropic messages 是 content[].text,OpenAI 兼容接口是 choices[0].message.content。

    结构不对就抛 JevError(走重试)。
    """
    if isinstance(data.get("content"), list):
        texts = [b.get("text", "") for b in data["content"] if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            return "".join(texts)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise JevError(f"响应里没有 choices[0].message.content:{str(data)[:200]}") from None
