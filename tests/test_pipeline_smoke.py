"""端到端冒烟测试：load → chunk → embed → milvus → retrieve → rerank。
不调用 LLM/VLM（避免依赖 API key）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.embedding.embedder import LocalBGEM3Embedder
from rag.loaders import load_directory
from rag.parsing.chunker import split_chunks
from rag.retrieval.retriever import Retriever
from rag.schema import Chunk
from rag.vector_store.milvus_store import MilvusStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """生成 mini 数据集到 tmp 目录：1 md + 1 txt（中英混合）。"""
    d = tmp_path_factory.mktemp("rag_fixtures")
    (d / "milvus.md").write_text(
        "# Milvus\n\nMilvus 是一个开源的向量数据库，支持稠密向量与稀疏向量的混合检索。\n",
        encoding="utf-8",
    )
    (d / "bge.md").write_text(
        "# BGE-M3\n\nBGE-M3 supports dense, sparse and multi-vector embeddings, "
        "and works well for multilingual (Chinese/English) retrieval.\n",
        encoding="utf-8",
    )
    (d / "geo.txt").write_text(
        "北京是中国的首都，有故宫、天安门和长城等著名景点。",
        encoding="utf-8",
    )
    return d


def test_pipeline_smoke(fixtures_dir: Path) -> None:
    raw = load_directory(fixtures_dir)
    assert len(raw) == 3, f"期望 3 个 chunk，实际 {len(raw)}"
    assert all(isinstance(c, Chunk) for c in raw)

    chunks = split_chunks(raw)
    chunks = [c for c in chunks if c.text.strip()]
    assert len(chunks) >= 3

    emb = LocalBGEM3Embedder.instance()
    result = emb.encode([c.text for c in chunks])
    assert result.dense_vecs.shape == (len(chunks), 1024)

    store = MilvusStore(dense_dim=emb.dense_dim, use_bm25_function=False)
    store.drop()
    store = MilvusStore(dense_dim=emb.dense_dim, use_bm25_function=False)
    n = store.insert(chunks, result.dense_vecs, result.lexical_weights)
    assert n == len(chunks)

    retriever = Retriever(store=store, embedder=emb)

    # 中文查询应该命中 Milvus 中文 chunk
    hits = retriever.retrieve("Milvus 是什么数据库？", retrieve_top_k=10, rerank_top_k=3)
    assert hits
    assert "milvus" in hits[0]["text"].lower() or "Milvus" in hits[0]["text"]

    # 英文查询应该命中 BGE 英文 chunk
    hits = retriever.retrieve("multilingual embedding model", retrieve_top_k=10, rerank_top_k=3)
    assert hits
    top_text = hits[0]["text"]
    assert "BGE" in top_text or "multilingual" in top_text.lower()

    # 跨语言：英文问中国首都，期望命中中文北京 chunk
    hits = retriever.retrieve("What is the capital of China?", retrieve_top_k=10, rerank_top_k=3)
    assert hits
    assert "北京" in hits[0]["text"]
