"""编码器接口。原方案中的 jev 编码器通过 JevEncoder 接入,在它之上并联新的分类头。

三种实现:
- TfidfEncoder:默认,无需下载模型,中文用字符 n-gram,英文用词。用于跑通流程和做基线。
- SentenceTransformerEncoder:任意 sentence-transformers 模型,例如 BAAI/bge-small-zh-v1.5。
- JevEncoder:接入原方案的 jev。传入 "模块:函数" 字符串,函数签名为 f(list[str]) -> np.ndarray。

编码器统一"先冻结":fit 只在 TfidfEncoder 上学习词表,另外两种不训练。
"""

import importlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


class Encoder:
    name = "base"

    def fit(self, texts):
        return self

    def encode(self, texts):
        raise NotImplementedError

    def config(self) -> dict:
        return {"type": self.name}


class TfidfEncoder(Encoder):
    name = "tfidf"

    def __init__(self, max_features: int = 50000):
        self.max_features = max_features
        self.vec = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True,
                                     max_features=max_features)),
            ("word", TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z][A-Za-z0-9_]+",
                                     lowercase=True, sublinear_tf=True)),
        ])

    def fit(self, texts):
        self.vec.fit(list(texts))
        return self

    def encode(self, texts):
        return normalize(self.vec.transform(list(texts)))

    def config(self):
        return {"type": self.name, "max_features": self.max_features}


class SentenceTransformerEncoder(Encoder):
    name = "st"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer  # 可选依赖

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))

    def config(self):
        return {"type": self.name, "model_name": self.model_name}

    def __getstate__(self):
        # 只存模型名,不把权重灌进 pickle(T-019):部署包里的 router.pkl 因此从 GB 级
        # 降到和 tfidf 一个量级;权重留在 ~/.cache/huggingface,加载时按名字重建。
        return {"model_name": self.model_name}

    def __setstate__(self, state):
        from sentence_transformers import SentenceTransformer  # 可选依赖

        self.model_name = state["model_name"]
        self.model = SentenceTransformer(self.model_name)


class JevEncoder(Encoder):
    name = "jev"

    def __init__(self, target: str):
        mod, _, fn = target.partition(":")
        self.target = target
        self.fn = getattr(importlib.import_module(mod), fn)

    def encode(self, texts):
        return normalize(np.asarray(self.fn(list(texts)), dtype=np.float32))

    def config(self):
        return {"type": self.name, "target": self.target}


def build_encoder(spec: str) -> Encoder:
    """spec 例:"tfidf"、"st:BAAI/bge-small-zh-v1.5"、"jev:my_pkg.jev:encode"。"""
    kind, _, arg = spec.partition(":")
    if kind == "tfidf":
        return TfidfEncoder()
    if kind == "st":
        return SentenceTransformerEncoder(arg or "BAAI/bge-small-zh-v1.5")
    if kind == "jev":
        return JevEncoder(arg)
    raise ValueError(f"未知编码器: {spec}")
