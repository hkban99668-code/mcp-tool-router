"""挂在共享编码器上的分类头。

请求侧:need(3 类 softmax)+ domain(多维多标签,跟 taxonomy 的域数走)+ action(6 维多标签)。
工具侧:domain + action,外加规则离散特征;风险由 rules.final_risk 推出,不单独训练。

编码器在这里只调用 encode,不更新参数,对应方案里"第一阶段冻结 jev,只训新分类头"。
"""

import pickle
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .encoder import Encoder, build_encoder
from .taxonomy import ACTION_IDS, DOMAIN_IDS, NEED_IDS


@dataclass
class HeadOutput:
    need: str = None
    need_probs: dict = field(default_factory=dict)
    domains: list = field(default_factory=list)
    domain_probs: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    action_probs: dict = field(default_factory=dict)
    confidence: float = 1.0


def _as_sparse(x):
    return x if sparse.issparse(x) else sparse.csr_matrix(x)


def _multilabel_confidence(probs: np.ndarray) -> float:
    """每个标签离 0.5 越远越确定,取最不确定的那个。"""
    return float(np.min(np.maximum(probs, 1 - probs)))


def _pick(probs: np.ndarray, classes, threshold=0.5):
    chosen = [c for c, p in zip(classes, probs) if p >= threshold]
    return chosen or [classes[int(np.argmax(probs))]]


class _ConstantClassifier:
    """训练集里只有一个类时顶替 `LogisticRegression` 的东西:恒定输出那一类,概率 1.0。

    `LogisticRegression.fit` 见到单类样本会抛
    "This solver needs samples of at least 2 classes ... only one class: 'single'",
    而真实数据按主域判定重新打标之后,v3 一致集里一条 multi 都没有(evaluate-real 当场崩在此处)。
    缺的类本来就学不出来,与其崩掉,不如明示"这一类预测不出来"。
    """

    def __init__(self, label):
        self.classes_ = np.array([label], dtype=object)

    def predict_proba(self, x):
        return np.ones((x.shape[0], 1))


class MultiHead:
    def __init__(self, encoder: Encoder, with_need: bool, feature_fn=None, action_feature_fn=None,
                 feature_weight: float = 1.0, C: float = 4.0,
                 need_feature_fn=None, need_two_stage: bool = False):
        """feature_fn 给能力域头加离散特征;action_feature_fn 给动作头,缺省同 feature_fn。
        need_feature_fn 只在 need 头上叠加特征(不影响能力域/动作,避免动了路由 Recall@k);
        need_two_stage 先判"要不要工具",再在要工具的样本上判 single/multi。
        """
        self.encoder = encoder
        self.with_need = with_need
        self.feature_fn = feature_fn
        self.action_feature_fn = action_feature_fn or feature_fn
        self.need_feature_fn = need_feature_fn
        self.feature_weight = feature_weight
        self.C = C
        self.need_two_stage = need_two_stage
        self.dv = DictVectorizer() if feature_fn else None
        self.dv_action = DictVectorizer() if self.action_feature_fn else None
        self.dv_need = DictVectorizer() if need_feature_fn else None
        self.need_clf = None
        self.need_stage2 = None  # 两级判定时的 second stage(single/multi)
        self.need_missing = set()  # 训练集里没有、因此预测不出来的 need 类(见 `_fit_need`)
        self.domain_mlb = MultiLabelBinarizer(classes=list(DOMAIN_IDS))
        self.action_mlb = MultiLabelBinarizer(classes=list(ACTION_IDS))

    # ---- 特征 ----
    def _features(self, texts, items=None, fit=False):
        """返回 (能力域特征, 动作特征, need 特征)。

        没有 need_feature_fn 时,need 头与能力域头共用同一份特征(与改动前一致);
        有 need_feature_fn 时在共用特征上再叠加 need 专用块。
        """
        x = _as_sparse(self.encoder.encode(texts))
        xd, xa, xn = x, x, x
        if self.dv is not None:
            feats = [self.feature_fn(it) for it in items]
            d = self.dv.fit_transform(feats) if fit else self.dv.transform(feats)
            xd = sparse.hstack([x, d * self.feature_weight]).tocsr()
            xn = xd
        if self.dv_action is not None:
            feats = [self.action_feature_fn(it) for it in items]
            d = self.dv_action.fit_transform(feats) if fit else self.dv_action.transform(feats)
            xa = sparse.hstack([x, d * self.feature_weight]).tocsr()
        if self.dv_need is not None:
            feats = [self.need_feature_fn(it) for it in items]
            d = self.dv_need.fit_transform(feats) if fit else self.dv_need.transform(feats)
            xn = sparse.hstack([xn, d * self.feature_weight]).tocsr()
        return xd, xa, xn

    def _lr(self):
        return LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")

    def _fit_need(self, x, y):
        """单类训练不了的原因见 `_ConstantClassifier`;记下的缺类供报告提示用。"""
        if len(y) == 0:
            return None
        if len(set(map(str, y))) < 2:
            return _ConstantClassifier(str(np.asarray(y)[0]))
        return self._lr().fit(x, y)

    # ---- 训练 ----
    def fit(self, texts, domains, actions, needs=None, items=None):
        if items is None and self.feature_fn is not None:
            items = list(texts)
        x, xa, xn = self._features(texts, items, fit=True)
        if self.with_need:
            none_mask = np.array([n == "none" for n in needs])
            self.need_missing = set(NEED_IDS) - set(map(str, needs))
            if self.need_two_stage:
                gate = np.array(["none" if n == "none" else "needs_mcp" for n in needs])
                self.need_clf = self._fit_need(xn, gate)
                self.need_stage2 = self._fit_need(xn[~none_mask], np.array(needs)[~none_mask])
            else:
                self.need_clf = self._fit_need(xn, np.array(needs))
            mask = ~none_mask
        else:
            mask = np.ones(len(texts), dtype=bool)
        xm, xam = x[mask], xa[mask]
        dm = [d for d, m in zip(domains, mask) if m]
        am = [a for a, m in zip(actions, mask) if m]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 某些标签在训练集里没有正例时 sklearn 会告警
            self.domain_clf = OneVsRestClassifier(self._lr()).fit(xm, self.domain_mlb.fit_transform(dm))
            self.action_clf = OneVsRestClassifier(self._lr()).fit(xam, self.action_mlb.fit_transform(am))
        return self

    # ---- 推理 ----
    def _need_probs(self, gate_row, sub_row) -> dict:
        """单级:直接取 3 类概率。两级:第一级给"要不要工具",第二级在要工具的样本上分 single/multi。

        返回的键始终是完整的 `NEED_IDS`(训练集缺的类给 0.0),否则下游
        `strategy` / Router 拿到的就不是"三个类的概率分布"了。
        """
        if self.need_clf is None:
            return {k: 0.0 for k in NEED_IDS}
        probs = dict(zip(list(map(str, self.need_clf.classes_)), map(float, gate_row)))
        if not self.need_two_stage:
            return {k: probs.get(k, 0.0) for k in NEED_IDS}
        sub = (dict(zip(list(map(str, self.need_stage2.classes_)), map(float, sub_row)))
               if sub_row is not None else {})
        p_needs = probs.get("needs_mcp", 0.0)
        return {"none": probs.get("none", 0.0),
                "single": p_needs * sub.get("single", 0.0),
                "multi": p_needs * sub.get("multi", 0.0)}

    def predict(self, texts, items=None) -> list:
        if items is None and self.feature_fn is not None:
            items = list(texts)
        self.check_taxonomy()
        x, xa, xn = self._features(texts, items)
        dp = self.domain_clf.predict_proba(x)
        ap = self.action_clf.predict_proba(xa)
        npb = self.need_clf.predict_proba(xn) if (self.with_need and self.need_clf is not None) \
            else None
        npb2 = self.need_stage2.predict_proba(xn) if (self.with_need and self.need_two_stage
                                                     and self.need_stage2 is not None) else None
        outs = []
        for i in range(x.shape[0]):
            o = HeadOutput()
            conf = 1.0
            if npb is not None:
                o.need_probs = self._need_probs(npb[i], npb2[i] if npb2 is not None else None)
                o.need = max(o.need_probs, key=o.need_probs.get)
                conf = float(max(o.need_probs.values()))
            o.domain_probs = dict(zip(DOMAIN_IDS, map(float, dp[i])))
            o.action_probs = dict(zip(ACTION_IDS, map(float, ap[i])))
            if o.need != "none":
                o.domains = _pick(dp[i], DOMAIN_IDS)
                o.actions = _pick(ap[i], ACTION_IDS)
                conf = min(conf, _multilabel_confidence(dp[i]))
            o.confidence = conf
            outs.append(o)
        return outs

    # ---- 持久化 ----
    def check_taxonomy(self) -> None:
        """训练时的类别数必须和当前 taxonomy 一致,不一致就报清楚的错。

        能力域从 12 个加到 14 个(T-009)之后,旧模型 `predict` 里的 `dict(zip(DOMAIN_IDS, probs))`
        会被静默截短到 12 个:多出来的域永远拿不到概率,而 reports 里的指标看着还是正常的数字。
        所以宁可在这里报错,让人去重训。
        """
        for name, mlb, ids in (("能力域", self.domain_mlb, DOMAIN_IDS),
                               ("动作", self.action_mlb, ACTION_IDS)):
            trained = getattr(mlb, "classes_", None)
            if trained is None:          # 还没 fit 过的头不检查
                continue
            trained = list(trained)
            if trained and len(trained) != len(ids):
                raise ValueError(
                    f"这个模型是在 {len(trained)} 个{name}上训练的,当前 taxonomy 有 {len(ids)} 个"
                    f"(训练时的名单:{', '.join(trained)})。"
                    f" taxonomy 改过就要重训:`python -m mcpcls.cli train`。")

    def save(self, path):
        enc = self.encoder
        state = dict(self.__dict__)
        for k in ("feature_fn", "action_feature_fn", "need_feature_fn"):
            fn = state[k]
            state[k] = fn.__name__ if fn else None
        if enc.name != "tfidf":  # 外部模型不进 pickle,加载时按配置重建
            state["encoder"] = enc.config()
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        from . import rules

        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj.check_taxonomy()          # taxonomy 变了就在加载时报错,不等到预测出来一堆对不上号的概率
        if isinstance(obj.encoder, dict):
            cfg = obj.encoder
            spec = {"st": f"st:{cfg.get('model_name', '')}", "jev": f"jev:{cfg.get('target', '')}"}[cfg["type"]]
            obj.encoder = build_encoder(spec)
        for k in ("feature_fn", "action_feature_fn", "need_feature_fn"):
            if getattr(obj, k):
                setattr(obj, k, getattr(rules, getattr(obj, k)))
        return obj
