"""兜底结果的组合策略:本地分类头和云端答案怎么合成最终结果。

T-004 只会"低置信度就整体替换",实测反而把 need 准确率从 82.6% 拉到 79.7%:
本地模型需要 MCP 召回 98.3% 但 none 只判对 4/11,Jev 正好相反(none 11/11、召回 79.3%)。
所以这里把"用哪一部分"拆成可配置的策略,线上路由(`Router`)和离线对比(`compare`)
调用同一个纯函数,两边结果不可能对不上。

取哪一方的规则由策略决定,什么时候去问由触发方式决定(见 `Router.fallback_trigger`)。
"""

from .heads import HeadOutput

STRATEGIES = ("local", "replace", "need_only", "both_none")
TRIGGERS = ("low_conf", "always")


def _top(probs: dict) -> list:
    """取概率最高的一个;概率表为空时给空列表。"""
    return [max(probs, key=probs.get)] if probs else []


def _from_local(local: HeadOutput, labels: dict | None, need: str) -> tuple:
    """need 已定,再定 domains / actions:最终判 none 就置空;否则优先用本地的。

    本地为空(多数是本地判了 none,那时 head 不填 domains / actions)时取本地概率最高的一个,
    `HeadOutput` 没有 action_probs 时才退到云端答案。
    """
    if need == "none":
        return [], []
    domains = list(local.domains) or _top(local.domain_probs)
    actions = list(local.actions)
    if not actions:
        # 本地没给动作:有动作概率就取概率最高的,分类头根本不输出动作概率时才用云端答案的
        actions = (_top(local.action_probs) if local.action_probs
                   else list((labels or {}).get("actions") or []))
    return domains, actions


def apply_strategy(local: HeadOutput, labels: dict | None, strategy: str) -> dict:
    """local: 本地分类头输出;labels: answers_to_labels(...) 的结果,None 表示没问或问失败。

    返回 {"need", "domains", "actions", "used"},used 表示最终结果是否用到了 labels。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"策略只能是 {STRATEGIES} 之一,收到 {strategy!r}")
    if labels is None or strategy == "local":
        return {"need": local.need, "domains": list(local.domains),
                "actions": list(local.actions), "used": False}

    if strategy == "replace":
        return {"need": labels["need"], "domains": list(labels.get("domains") or []),
                "actions": list(labels.get("actions") or []), "used": True}

    if strategy == "need_only":
        need = labels["need"]
    else:       # both_none:两边都判 none 才算 none,否则取非 none 那一方(都非 none 用本地)
        need = "none" if (local.need == "none" and labels["need"] == "none") else (
            labels["need"] if local.need == "none" else local.need)
    domains, actions = _from_local(local, labels, need)
    return {"need": need, "domains": domains, "actions": actions, "used": True}
