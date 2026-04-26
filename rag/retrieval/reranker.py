from __future__ import annotations

from FlagEmbedding import FlagReranker  # type: ignore

from rag.config import settings


class Reranker:
    """bge-reranker-v2-m3 包装。给 (query, passage) 对打分用于二阶段重排。"""

    _instance: "Reranker | None" = None

    def __init__(self) -> None:
        self.model = FlagReranker(
            settings.reranker_model,
            use_fp16=settings.use_fp16,
            devices=settings.embedding_device,
        )

    @classmethod
    def instance(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """candidates: hybrid_search 返回的 list[{'text','meta','score'}]
        返回排序后取 top_k，附加 rerank_score 字段。"""
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
