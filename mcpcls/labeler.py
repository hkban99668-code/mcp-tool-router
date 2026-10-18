"""LLM 打标与 LLM 兜底分类。

打标流程(对应方案"数据与特征"一节):
1. 类别定义卡(taxonomy.category_cards)作为 prompt 的固定部分;
2. LLM 输出 JSON:三个轴的标签 + 一句理由 + 自评置信度;
3. 双路打标:两套 prompt(或两个模型)各标一遍,一致的入库,不一致的进人工队列。

需要环境变量 ANTHROPIC_API_KEY。模型可用 --model / --model-b 指定。
"""

import json
import re

from .rules import tool_text
from .taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS, category_cards

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_A = """你是 MCP 工具调用分类的标注员。请严格按下面的类别定义给样本打标签。
{cards}

只输出一个 JSON 对象,不要输出其他内容。"""

SYSTEM_B = """你负责判断一条文本属于哪类 MCP 能力。先想清楚它要对什么对象做什么操作,再选类别。
类别定义如下,只能使用这里列出的 id:
{cards}

输出格式:单个 JSON 对象,不加任何解释性文字。"""

REQUEST_TMPL = """样本类型:用户请求
文本:{text}

输出字段:
- need:none / single / multi
- domains:能力域 id 列表(need 为 none 时为空列表)
- actions:动作类型 id 列表(need 为 none 时为空列表)
- reason:一句话理由
- confidence:0 到 1 的自评置信度"""

TOOL_TMPL = """样本类型:MCP 工具定义
{text}

输出字段:
- domains:能力域 id 列表
- actions:动作类型 id 列表(这个工具被调用时会做的操作)
- reason:一句话理由
- confidence:0 到 1 的自评置信度"""


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"LLM 输出里没有 JSON: {text[:200]}")
    d = json.loads(m.group(0))
    d["domains"] = [x for x in d.get("domains", []) if x in DOMAIN_IDS]
    d["actions"] = [x for x in d.get("actions", []) if x in ACTION_IDS]
    if "need" in d and d["need"] not in NEED_IDS:
        d["need"] = "single"
    return d


class LLMLabeler:
    def __init__(self, model: str = DEFAULT_MODEL, model_b: str = None, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.model_b = model_b or model
        cards = category_cards()
        self.systems = (SYSTEM_A.format(cards=cards), SYSTEM_B.format(cards=cards))

    def _ask(self, system: str, user: str, model: str) -> dict:
        msg = self.client.messages.create(
            model=model, max_tokens=400, system=system,
            messages=[{"role": "user", "content": user}])
        return _parse("".join(b.text for b in msg.content if getattr(b, "type", "") == "text"))

    def label_request(self, text: str, variant: int = 0) -> dict:
        model = self.model if variant == 0 else self.model_b
        return self._ask(self.systems[variant], REQUEST_TMPL.format(text=text), model)

    def label_tool(self, tool: dict, variant: int = 0) -> dict:
        model = self.model if variant == 0 else self.model_b
        return self._ask(self.systems[variant], TOOL_TMPL.format(text=tool_text(tool)), model)

    def dual_label(self, item, kind: str = "request") -> dict:
        """返回 {"label": ..., "agreed": bool, "a": ..., "b": ...}。"""
        fn = self.label_request if kind == "request" else self.label_tool
        a, b = fn(item, 0), fn(item, 1)
        agreed = (set(a["domains"]) == set(b["domains"]) and set(a["actions"]) == set(b["actions"])
                  and a.get("need") == b.get("need"))
        return {"label": a if agreed else None, "agreed": agreed, "a": a, "b": b}

    def as_fallback(self):
        """给 Router 用的兜底分类函数。"""
        def fallback(text):
            d = self.label_request(text)
            return {"need": d.get("need", "single"), "domains": d["domains"], "actions": d["actions"]}
        return fallback
