"""能力域预测:检索结果投票(T-027)。

背景:`reports/eval_reference.md`(T-022)里,当前默认能力域头在定稿参考集上的主域准确率
只有 40.8%,147 条里 36 条各域概率都没过阈值。另一方面,真实目录 2405 个工具**每个**都有
银标 `domains`,而检索(R@5 英文六成以上)已经比较准。所以这里在检索到的 Top-k 工具的
`domains` 上投票,单独或与分类头分布融合,作为能力域的第二个来源。

三个纯函数 + 一个可选的 policy:
- `vote_probs`:Top-k 候选按 domains 加权投票并归一化成分布;
- `fuse_probs`:`β·头分布 + (1-β)·投票分布`;
- `pick_domains`:按阈值挑域,可选"至少给 top-1"(与 `heads._pick` 同一条阈值口径);
- `DomainVotePolicy`:把上面三个参数打包,给调用方(离线评估 / 将来的线上开关)一个入口。

只用 numpy 和 taxonomy,不加载模型、不联网。T-027 把它当实验用,**默认配置没有改动**
(见 `reports/eval_t027.md`);Router 也没有接它。
"""

from dataclasses import dataclass

from .taxonomy import DOMAIN_IDS

# 投票口径:按名次(rank)而不是检索分。理由:tfidf 余弦分的量纲随查询变(有的查询
# 分数整体很小、彼此接近),名次是尺度无关的;代码里已有的 RRF(T-025/T-026)也是名次口径。
WEIGHTS = ("rank", "score")
DEFAULT_WEIGHT = "rank"
# 挑域阈值与 `heads._pick` 的默认一致,不许在这里另调
DEFAULT_THRESHOLD = 0.5


def vote_probs(hits, weight: str = DEFAULT_WEIGHT) -> dict:
    """Top-k 候选按 `domains` 加权投票,归一化成分布(键始终是完整的 `DOMAIN_IDS`)。

    `hits` 是 `[(ToolRecord, score)]`,已按检索分降序。`weight="rank"` 时第 r 名(从 1 起)
    权重 `1/r`;`weight="score"` 时权重取检索分本身。没有任何候选(或候选都没域)时返回全 0。
    """
    if weight not in WEIGHTS:
        raise ValueError(f"投票权重只能是 {WEIGHTS} 之一,收到 {weight!r}")
    dist = {d: 0.0 for d in DOMAIN_IDS}
    for rank, (rec, score) in enumerate(hits, start=1):
        w = 1.0 / rank if weight == "rank" else float(score)
        for d in rec.domains:
            if d in dist:
                dist[d] += w
    total = sum(dist.values())
    if total > 0:
        for d in dist:
            dist[d] /= total
    return dist


def fuse_probs(head_probs: dict, vote: dict, beta: float) -> dict:
    """`β·头分布 + (1-β)·投票分布`,逐域相加,键为完整的 `DOMAIN_IDS`。

    β=1 退化为头分布,β=0 退化为投票分布。两个输入都按"每域一个分数"用,不要求是概率。
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta 要在 [0, 1],收到 {beta!r}")
    return {d: beta * float(head_probs.get(d, 0.0)) + (1.0 - beta) * float(vote.get(d, 0.0))
            for d in DOMAIN_IDS}


def argmax_domain(probs: dict) -> str:
    """分布里分数最大的域;分布为空时记 ""。平局按 `DOMAIN_IDS` 顺序取第一个。"""
    if not probs:
        return ""
    return max(DOMAIN_IDS, key=lambda d: (probs.get(d, 0.0), -DOMAIN_IDS.index(d)))


def pick_domains(probs: dict, threshold: float = DEFAULT_THRESHOLD,
                 floor_top1: bool = False) -> list:
    """按阈值挑出分数 >= threshold 的域。

    `floor_top1=True` 时,一个都没过阈值就至少给分数最高的那个(分数必须 > 0);
    为 False 时保持"阈值不变",可能返回空列表。与 `heads._pick` 是同一套阈值语义。
    """
    chosen = [d for d in DOMAIN_IDS if probs.get(d, 0.0) >= threshold]
    if chosen:
        return chosen
    if floor_top1:
        top = argmax_domain(probs)
        return [top] if top and probs.get(top, 0.0) > 0.0 else []
    return []


@dataclass
class DomainVotePolicy:
    """把候选数、融合权重、投票口径、挑域阈值打包。T-027 的离线评估用它对齐参数。

    默认值取 T-027 开发折上主口径最高的那组(D2 rank k=20 β=0.3);但它没有成为线上默认,
    见 `reports/eval_t027.md`。
    """

    k: int = 20
    beta: float = 0.3
    weight: str = DEFAULT_WEIGHT
    threshold: float = DEFAULT_THRESHOLD
    floor_top1: bool = True

    def resolve(self, head_probs: dict, hits) -> tuple:
        """(融合后的分布, 挑出的 domains)。`hits` 是检索 Top-k 的 `[(ToolRecord, score)]`。"""
        vote = vote_probs(hits[:self.k], self.weight)
        probs = fuse_probs(head_probs, vote, self.beta) if self.beta < 1.0 else dict(head_probs)
        return probs, pick_domains(probs, self.threshold, self.floor_top1)
