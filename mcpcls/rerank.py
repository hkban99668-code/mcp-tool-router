"""语义重排(T-025):召回仍用 tfidf,只对它的 Top-N 做语义重排。

三种融合方式(fusion):
- reorder:只按语义相似度排 tfidf 的 Top-N(纯重排);
- linear:`alpha * tfidf分 + (1 - alpha) * 语义相似度`,tfidf 分是 `retrieve` 给的排序分;
- rrf:倒数排名融合,只看两边各自的名次,不看原始分数(常数 RRF_K=60)。

混合召回(T-026,可选):`recall_m > 0` 时把语义 Top-M 里不在 tfidf 召回池里的工具补进候选
(并集),再走上面的融合;补进来的工具拿不到 tfidf 召回分,按 0 参与融合。`recall_m=0`
(默认)时只用 tfidf 召回的 Top-N,行为与 T-025 逐字节一致。

工具向量预计算后按 (模型, 全部索引文本哈希) 缓存到 `.cache/tool_vecs/`:索引文本没变就
直接读盘,新工具接入或中文译文更新后哈希失配自动重建。部署时可在训练后显式调
`warm(catalog)` 把构建成本挪到离线;线上首个请求不用等。

没装 sentence-transformers 或模型不在本地 HF 缓存时抛 `RerankError`,提示用户怎么做;
不开重排(`--rerank` 不传)根本不会走到这里,行为与改前逐字节一致。
"""

import hashlib
import pathlib
import time

import numpy as np

from .catalog import ToolCatalog

ROOT = pathlib.Path(__file__).resolve().parent.parent
VEC_CACHE = ROOT / ".cache" / "tool_vecs"

FUSIONS = ("reorder", "linear", "rrf")
RRF_K = 60          # rrf 的平滑常数,论文常用值,不做扫描


class RerankError(RuntimeError):
    """开重排但环境不满足:没装 sentence-transformers,或模型不在本地缓存。"""


def _hf_cache_dir() -> pathlib.Path:
    try:
        from huggingface_hub import constants

        return pathlib.Path(constants.HF_HUB_CACHE)
    except Exception:
        return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_dir(model_name: str):
    """本地缓存里这个模型最新的可用快照目录;没有就返回 None。

    不用 `snapshot_download(local_files_only=True)` 整体校验:快照缺几个无关文件
    (图片、onnx 导出之类)时它会把整个模型判成"不完整",但推理要的权重、配置、
    tokenizer 其实都在。
    """
    snaps = _hf_cache_dir() / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not snaps.exists():
        return None
    dirs = sorted((p for p in snaps.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    fallback = None
    for p in dirs:   # 有权重、config 里认得出模型结构的快照才可用
        if not ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()):
            continue
        fallback = fallback or p
        cfg = p / "config.json"
        if cfg.exists() and "model_type" in cfg.read_text(encoding="utf-8", errors="ignore"):
            return p
    return fallback


def _load_st_model(model_name: str):
    """从本地 HF 缓存加载语义模型,全程不联网。环境不满足时抛带解决办法的 RerankError。"""
    try:
        from sentence_transformers import SentenceTransformer  # 可选依赖
    except ImportError as e:
        raise RerankError(
            f"开了重排但没装 sentence-transformers({e})。"
            "先 `pip install -r requirements-st.txt`,或去掉 --rerank 用纯 tfidf 召回。") from e
    if pathlib.Path(model_name).is_dir():          # 本地路径直接用
        local = pathlib.Path(model_name)
    else:
        local = _snapshot_dir(model_name)
        if local is None:
            raise RerankError(
                f"开了重排但模型 {model_name} 不在本地 HF 缓存({_hf_cache_dir()})。"
                f"在有网的环境执行 `huggingface-cli download {model_name}` 后把缓存同步过来,"
                "或改用本地已有的模型(BAAI/bge-m3 / intfloat/multilingual-e5-small),"
                "或去掉 --rerank 用纯 tfidf 召回。")
    try:
        return SentenceTransformer(str(local), device="cpu")
    except Exception as e:
        raise RerankError(
            f"模型 {model_name} 从本地缓存加载失败({e}):缓存可能不完整,"
            f"在有网的环境重新 `huggingface-cli download {model_name}`,"
            "或去掉 --rerank 用纯 tfidf 召回。") from e


# 同一个模型只加载一次:同一份评测里多个 (N, fusion, alpha) 组合共用权重
_MODEL_CACHE = {}


class SemanticReranker:
    """对 tfidf 召回的 Top-N 做语义重排。模型与工具向量都懒加载,实例可安全 pickle。"""

    def __init__(self, model_name: str, rerank_n: int = 20, fusion: str = "reorder",
                 alpha: float = 0.5, cache_dir=None, recall_m: int = 0):
        if fusion not in FUSIONS:
            raise ValueError(f"融合方式只能是 {FUSIONS} 之一,收到 {fusion!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 要在 [0, 1],收到 {alpha!r}")
        if int(rerank_n) < 1:
            raise ValueError(f"rerank_n 要 >= 1,收到 {rerank_n!r}")
        if int(recall_m) < 0:
            raise ValueError(f"recall_m 要 >= 0,收到 {recall_m!r}")
        self.model_name = model_name
        self.rerank_n = int(rerank_n)
        self.fusion = fusion
        self.alpha = float(alpha)
        self.recall_m = int(recall_m)   # T-026:0 = 只重排 tfidf 召回(与 T-025 一致)
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else VEC_CACHE
        self._model = None
        self._vecs = None        # 全目录工具向量,行序与 catalog.records 一致
        self._pos = {}           # tool_id -> 行号
        self._for = None         # 已为哪个 catalog 实例建好向量
        self.last_ms = 0.0       # 最近一次 rerank 的耗时(查询编码 + 打分),评估用
        self.first_build = None  # 第一次预计算工具向量的信息(cached / 耗时),报告用

    # ---- 懒加载与 pickle:权重和向量都不进 pickle,只存配置 --------------------
    def _st(self):
        if self._model is None:
            model = _MODEL_CACHE.get(self.model_name)
            if model is None:
                model = _load_st_model(self.model_name)
                _MODEL_CACHE[self.model_name] = model
            self._model = model
        return self._model

    def __getstate__(self):
        return {"model_name": self.model_name, "rerank_n": self.rerank_n,
                "fusion": self.fusion, "alpha": self.alpha,
                "recall_m": self.recall_m, "cache_dir": str(self.cache_dir)}

    def __setstate__(self, state):
        self.__init__(state["model_name"], state["rerank_n"], state["fusion"],
                      state["alpha"], state["cache_dir"], state.get("recall_m", 0))

    def config(self) -> dict:
        return {"model": self.model_name, "n": self.rerank_n, "fusion": self.fusion,
                "alpha": self.alpha, "recall_m": self.recall_m}

    # ---- 工具向量 -------------------------------------------------------------
    def warm(self, catalog) -> dict:
        """预计算整个目录的工具向量并落盘缓存;索引文本没变时直接读盘。"""
        texts = [ToolCatalog._index_text(r) for r in catalog.records]
        h = hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest()[:16]
        safe = self.model_name.replace("/", "__")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{safe}-{h}-{len(texts)}.npy"
        cached = path.exists()
        t0 = time.perf_counter()
        if cached:
            vecs = np.load(path)
        else:
            vecs = np.asarray(self._st().encode(texts, normalize_embeddings=True),
                              dtype=np.float32)
            tmp = path.with_name(path.name + ".tmp.npy")   # 原子写,中断不留半个文件
            np.save(tmp, vecs)
            tmp.rename(path)
        self._vecs = vecs
        self._pos = {r.tool_id: i for i, r in enumerate(catalog.records)}
        self._for = catalog
        info = {"cached": cached, "build_ms": (time.perf_counter() - t0) * 1000,
                "n_tools": len(texts), "dim": int(vecs.shape[1]), "path": str(path)}
        if self.first_build is None:
            self.first_build = info
        return info

    # ---- 重排 -----------------------------------------------------------------
    def _query_vec(self, query):
        """查询向量(已归一化),与 `warm` 里工具向量同一种编码。"""
        return np.asarray(self._st().encode([query], normalize_embeddings=True),
                          dtype=np.float32)[0]

    def semantic_top(self, query, m, catalog=None):
        """全目录语义相似度前 `m` 个工具 `[(ToolRecord, 分数)]`(T-026 调查用)。

        `catalog` 传了就自动预热全目录向量;没预热过又没给 catalog 时返回 `[]`。
        """
        if catalog is not None and self._for is not catalog:
            self.warm(catalog)
        cat = catalog if catalog is not None else self._for
        if self._vecs is None or cat is None:
            return []
        q = self._query_vec(query)
        sem = np.asarray(self._vecs @ q, dtype=np.float32).ravel()
        order = np.argsort(-sem, kind="stable")[:min(int(m), len(sem))]
        return [(cat.records[i], float(sem[i])) for i in order]

    def _expand_pool(self, pool, q, catalog):
        """并集召回(T-026):把语义 Top-`recall_m` 里不在池中的工具补进来,tfidf 分记 0。

        池子外(不在 tfidf 召回里的)工具拿不到 tfidf 召回分,按 0 参与融合,只靠语义相似度
        竞争:既不用把语义重排器再绑回 tfidf 编码器,也不会凭空给它们一个 tfidf 分数。
        """
        cat = catalog if catalog is not None else self._for
        if self._vecs is None or cat is None:
            return pool
        sem = np.asarray(self._vecs @ q, dtype=np.float32).ravel()
        order = np.argsort(-sem, kind="stable")[:min(self.recall_m, len(sem))]
        have = {self._pos[r.tool_id] for r, _ in pool}
        return pool + [(cat.records[i], 0.0) for i in order if i not in have]

    def rerank(self, query, hits, catalog=None):
        """重排 `[(ToolRecord, tfidf分)]`(已按分数降序),返回同样结构的列表、顺序换过。

        元组里的分数换成融合分(reorder=语义相似度,linear=α·tfidf+(1-α)·语义,
        rrf=倒数排名融合),线上候选的 score 字段跟着变。`catalog` 传了就自动预计算
        全目录向量(Router 一定传);不传时只为本次 hits 现算,不走磁盘缓存。
        `recall_m > 0` 时先把语义 Top-`recall_m` 并进池子再融合(见 `_expand_pool`)。
        """
        if not hits:
            self.last_ms = 0.0
            return hits
        t0 = time.perf_counter()
        if catalog is not None and self._for is not catalog:
            self.warm(catalog)
        q = self._query_vec(query)
        pool = hits
        if self.recall_m > 0:
            pool = self._expand_pool(list(hits), q, catalog)
        recs = [r for r, _ in pool]
        if self._vecs is not None:
            rows = [self._pos[r.tool_id] for r in recs]
            sem = np.asarray(self._vecs[rows] @ q, dtype=np.float32).ravel()
        else:
            vecs = np.asarray(self._st().encode([ToolCatalog._index_text(r) for r in recs],
                                                normalize_embeddings=True), dtype=np.float32)
            sem = np.asarray(vecs @ q, dtype=np.float32).ravel()
        t = np.asarray([s for _, s in pool], dtype=np.float32)
        out = self._fuse(pool, t, sem)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    def _fuse(self, hits, t, sem):
        n = len(hits)
        if self.fusion == "reorder":
            score = sem
        elif self.fusion == "linear":
            score = self.alpha * t + (1.0 - self.alpha) * sem
        else:  # rrf:只看名次,1/(K+名次) 相加
            rank_t = np.empty(n)
            rank_t[np.argsort(-t, kind="stable")] = np.arange(n)
            rank_s = np.empty(n)
            rank_s[np.argsort(-sem, kind="stable")] = np.arange(n)
            score = 1.0 / (RRF_K + rank_t) + 1.0 / (RRF_K + rank_s)
        order = np.argsort(-score, kind="stable")
        return [(hits[i][0], float(score[i])) for i in order]
