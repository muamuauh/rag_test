"""Embedder 抽象层。

两种实现：
  - LocalBGEM3Embedder：本地 BGE-M3，输出 dense + sparse（lexical_weights）
  - CloudEmbedder：LiteLLM 调云端 embedding API，仅输出 dense。
                   sparse 由 Milvus 内置 BM25 function 从原文自动生成。

通过 settings.embedding_provider 切换；上层用 get_embedder() 取实例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rag.config import settings


@dataclass
class EncodeResult:
    dense_vecs: np.ndarray  # shape (N, dense_dim)
    lexical_weights: list[dict[int, float]] | None  # None 表示不输出 sparse


@dataclass
class QueryEncodeResult:
    dense: np.ndarray
    sparse: dict[int, float] | None  # None 表示由 Milvus BM25 用原文计算


class Embedder(Protocol):
    dense_dim: int
    has_sparse: bool  # 是否输出 sparse 向量

    def encode(self, texts: list[str]) -> EncodeResult: ...
    def encode_query(self, text: str) -> QueryEncodeResult: ...


# ---------- Local: BGE-M3 ----------

class LocalBGEM3Embedder:
    """BGE-M3 本地推理，dense + sparse 双路。"""

    has_sparse: bool = True
    dense_dim: int = 1024  # BGE-M3 固定

    _instance: "LocalBGEM3Embedder | None" = None

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore  # noqa: PLC0415

        self.model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=settings.use_fp16,
            devices=settings.embedding_device,
        )

    @classmethod
    def instance(cls) -> "LocalBGEM3Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 12,
        max_length: int = 8192,
    ) -> EncodeResult:
        result: dict[str, Any] = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return EncodeResult(
            dense_vecs=result["dense_vecs"],
            lexical_weights=result["lexical_weights"],
        )

    def encode_query(self, text: str) -> QueryEncodeResult:
        r = self.encode([text])
        return QueryEncodeResult(
            dense=r.dense_vecs[0],
            sparse=r.lexical_weights[0] if r.lexical_weights else None,
        )


# ---------- Cloud: LiteLLM ----------

class CloudEmbedder:
    """调用云端 dense embedding API（OpenAI / 智谱 / 通义 / Jina 等）。

    sparse 不输出 —— 由 Milvus 端 BM25 function 从原文计算。
    """

    has_sparse: bool = False

    _instance: "CloudEmbedder | None" = None

    def __init__(self) -> None:
        self.model_name = settings.cloud_embedding_model
        self.dense_dim = settings.embedding_dim

    @classmethod
    def instance(cls) -> "CloudEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _call(self, texts: list[str]) -> np.ndarray:
        import litellm  # noqa: PLC0415

        # 三元组：显式 api_key / api_base 优先；留空则 fallback 到 LiteLLM 默认行为
        extra: dict[str, Any] = {}
        if settings.embedding_api_key:
            extra["api_key"] = settings.embedding_api_key
        if settings.embedding_base_url:
            extra["api_base"] = settings.embedding_base_url

        resp = litellm.embedding(model=self.model_name, input=texts, **extra)
        # 兼容不同 provider 的返回结构
        try:
            data = resp["data"]
            vecs = [item["embedding"] for item in data]
        except (KeyError, TypeError):
            vecs = [item.embedding for item in resp.data]
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.shape[1] != self.dense_dim:
            raise ValueError(
                f"Cloud embedding 返回维度 {arr.shape[1]} 与配置 EMBEDDING_DIM={self.dense_dim} 不一致；"
                f"请把 .env 中的 EMBEDDING_DIM 改为 {arr.shape[1]} 并清空 collection 重建。"
            )
        return arr

    def encode(self, texts: list[str], *, batch_size: int = 64) -> EncodeResult:
        # 多数 provider 单次最多 N 条，分批
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunks.append(self._call(texts[i : i + batch_size]))
        dense = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.dense_dim), dtype=np.float32)
        return EncodeResult(dense_vecs=dense, lexical_weights=None)

    def encode_query(self, text: str) -> QueryEncodeResult:
        r = self.encode([text])
        return QueryEncodeResult(dense=r.dense_vecs[0], sparse=None)


# ---------- Factory ----------

def get_embedder() -> Embedder:
    if settings.embedding_provider == "cloud":
        return CloudEmbedder.instance()
    return LocalBGEM3Embedder.instance()


# 保留旧名以兼容历史调用
BGEM3Embedder = LocalBGEM3Embedder
