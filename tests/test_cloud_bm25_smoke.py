"""验证 cloud + Milvus BM25 模式：

  - MilvusStore 用 BM25 function（sparse 由 Milvus 内部从 text 生成）
  - 插入只传 dense（不传 sparse_weights）
  - hybrid_search 时 sparse 通道传 raw text

云端 dense 用 mock（避免依赖外网/API key）。验证整条链路与 schema 配置正确。
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.retrieval.reranker import Reranker
from rag.schema import Chunk
from rag.vector_store.milvus_store import MilvusStore


DENSE_DIM = 64  # mock 用低维


class MockCloudEmbedder:
    """模拟云端 embedder：不输出 sparse；dense 用确定性随机向量。"""

    has_sparse = False
    dense_dim = DENSE_DIM

    def __init__(self) -> None:
        self._rng = np.random.default_rng(seed=42)
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            # 用文本 hash 做种子，保证 query 与 insert 时同一文本得到相同向量
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed=seed)
            v = rng.standard_normal(DENSE_DIM).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            self._cache[text] = v
        return self._cache[text]

    def encode(self, texts):
        from rag.embedding.embedder import EncodeResult  # noqa: PLC0415

        dense = np.stack([self._vec(t) for t in texts]).astype(np.float32)
        return EncodeResult(dense_vecs=dense, lexical_weights=None)

    def encode_query(self, text):
        from rag.embedding.embedder import QueryEncodeResult  # noqa: PLC0415

        return QueryEncodeResult(dense=self._vec(text), sparse=None)


def test_cloud_bm25_pipeline() -> None:
    emb = MockCloudEmbedder()

    chunks = [
        Chunk(text="Milvus 是一个开源向量数据库，支持稠密与稀疏混合检索。",
              source_path="t1.md", chunk_type="text"),
        Chunk(text="BGE-M3 supports dense and sparse embeddings for multilingual retrieval.",
              source_path="t2.md", chunk_type="text"),
        Chunk(text="北京是中国的首都，拥有故宫与天安门等著名景点。",
              source_path="t3.md", chunk_type="text"),
        Chunk(text="LiteLLM 提供统一的多 provider 接口，支持 OpenAI 与 Claude。",
              source_path="t4.md", chunk_type="text"),
    ]

    # 关键：use_bm25_function=True 触发新 schema 分支
    store = MilvusStore(dense_dim=DENSE_DIM, use_bm25_function=True)
    store.drop()
    store = MilvusStore(dense_dim=DENSE_DIM, use_bm25_function=True)

    result = emb.encode([c.text for c in chunks])
    # 关键：sparse_weights 不传，由 Milvus BM25 function 生成
    n = store.insert(chunks, result.dense_vecs, sparse_weights=None)
    assert n == len(chunks)

    # 关键：query_sparse 不传；query_text 让 Milvus 内部分词算 BM25
    q = emb.encode_query("Milvus 向量数据库")
    hits = store.hybrid_search(
        query_dense=q.dense,
        query_text="Milvus 向量数据库",
        query_sparse=None,
        top_k=4,
    )
    assert hits, "BM25 + dense 混合检索应该返回结果"
    # BM25 通道应能精确命中包含 "Milvus" 字面量的 chunk
    top_texts = [h["text"] for h in hits[:2]]
    assert any("Milvus" in t for t in top_texts), \
        f"top-2 应有 Milvus chunk，实际：{top_texts}"

    # 中文关键词命中
    hits = store.hybrid_search(
        query_dense=emb._vec("北京 首都"),
        query_text="北京 首都",
        top_k=4,
    )
    assert hits
    top_texts = [h["text"] for h in hits[:2]]
    assert any("北京" in t for t in top_texts), f"应命中北京 chunk，实际：{top_texts}"

    # 配合 reranker 的全链路
    reranker = Reranker.instance()
    reranked = reranker.rerank("Milvus 向量数据库", hits, top_k=2)
    assert len(reranked) <= 2
    assert all("rerank_score" in r for r in reranked)

    store.drop()


def test_external_sparse_required() -> None:
    """外部 sparse 模式调用 insert 不传 sparse_weights 应该报错。"""
    store = MilvusStore(dense_dim=4, use_bm25_function=False)
    store.drop()
    store = MilvusStore(dense_dim=4, use_bm25_function=False)
    with pytest.raises(ValueError, match="sparse_weights"):
        store.insert(
            [Chunk(text="x", source_path="x", chunk_type="text")],
            np.zeros((1, 4), dtype=np.float32),
            sparse_weights=None,
        )
    store.drop()
