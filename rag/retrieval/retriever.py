from __future__ import annotations

from rag.config import settings
from rag.embedding.embedder import Embedder, get_embedder
from rag.retrieval.reranker import Reranker
from rag.vector_store.milvus_store import MilvusStore


class Retriever:
    """端到端检索器：query → embed → Milvus 混合检索 → reranker 重排 → top-k

    自动按 settings.embedding_provider 选择本地 BGE-M3 或云端 + Milvus BM25。
    """

    def __init__(
        self,
        store: MilvusStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.store = store or MilvusStore(
            dense_dim=self.embedder.dense_dim,
            use_bm25_function=not self.embedder.has_sparse,
        )
        self.reranker = reranker or Reranker.instance()

    def retrieve(
        self,
        query: str,
        retrieve_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[dict]:
        retrieve_top_k = retrieve_top_k or settings.retrieve_top_k
        rerank_top_k = rerank_top_k or settings.rerank_top_k

        q = self.embedder.encode_query(query)
        candidates = self.store.hybrid_search(
            query_dense=q.dense,
            query_text=query,
            query_sparse=q.sparse,
            top_k=retrieve_top_k,
        )
        return self.reranker.rerank(query, candidates, top_k=rerank_top_k)
