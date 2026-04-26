"""Milvus 向量库封装，支持两种 sparse 模式：

  - 外部 sparse（与本地 BGE-M3 配合）：调用方传入 sparse 字典，metric_type=IP
  - BM25 function（与云端 dense 配合）：Milvus 启用 analyzer，从 text 字段
    自动生成 sparse；查询时 sparse 通道直接传 raw text，metric_type=BM25

通过 settings.embedding_provider 决定。Collection schema 在 _ensure_collection
里按模式分支，已有 collection 不会被覆盖。
"""

from __future__ import annotations

from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from rag.config import settings
from rag.schema import Chunk


DENSE_FIELD = "dense"
SPARSE_FIELD = "sparse"
TEXT_FIELD = "text"
META_FIELD = "meta"


class MilvusStore:
    def __init__(
        self,
        *,
        dense_dim: int | None = None,
        use_bm25_function: bool | None = None,
    ) -> None:
        settings.ensure_dirs()
        uri = settings.milvus_uri if settings.milvus_is_remote else str(settings.milvus_path)
        self.client = MilvusClient(uri=uri)
        self.collection = settings.milvus_collection

        # 根据 provider 自动判定，调用方也可显式覆盖
        if use_bm25_function is None:
            use_bm25_function = settings.embedding_provider == "cloud"
        self.use_bm25_function = use_bm25_function

        self.dense_dim = dense_dim if dense_dim is not None else settings.embedding_dim
        self._ensure_collection()

    # ---- schema ----

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)

        if self.use_bm25_function:
            # BM25 模式：text 字段需要开 analyzer，Milvus 才能内部分词建 BM25 sparse
            schema.add_field(
                TEXT_FIELD,
                DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": settings.milvus_analyzer},
            )
        else:
            schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=65535)

        schema.add_field(META_FIELD, DataType.JSON)
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)

        if self.use_bm25_function:
            schema.add_function(
                Function(
                    name="text_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=[TEXT_FIELD],
                    output_field_names=[SPARSE_FIELD],
                )
            )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=DENSE_FIELD,
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        if self.use_bm25_function:
            index_params.add_index(
                field_name=SPARSE_FIELD,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"bm25_k1": 1.2, "bm25_b": 0.75},
            )
        else:
            index_params.add_index(
                field_name=SPARSE_FIELD,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    # ---- write ----

    def insert(
        self,
        chunks: list[Chunk],
        dense_vecs,
        sparse_weights: list[dict[int, float]] | None = None,
    ) -> int:
        """插入 chunks。

        - 外部 sparse 模式：sparse_weights 必填
        - BM25 模式：sparse_weights 应为 None；Milvus 从 text 字段自动生成
        """
        if not self.use_bm25_function and sparse_weights is None:
            raise ValueError("外部 sparse 模式必须传 sparse_weights")

        rows = []
        for i, (c, dv) in enumerate(zip(chunks, dense_vecs)):
            row = {
                TEXT_FIELD: c.text or "",
                META_FIELD: {
                    "source_path": c.source_path,
                    "chunk_type": c.chunk_type,
                    "page": c.page,
                    "image_path": c.image_path,
                    "metadata": c.metadata,
                },
                DENSE_FIELD: dv.tolist() if hasattr(dv, "tolist") else list(dv),
            }
            if not self.use_bm25_function:
                row[SPARSE_FIELD] = sparse_weights[i]  # type: ignore[index]
            rows.append(row)

        if not rows:
            return 0
        self.client.insert(self.collection, rows)
        self.client.flush(self.collection)
        return len(rows)

    # ---- read ----

    def hybrid_search(
        self,
        query_dense,
        query_text: str,
        query_sparse: dict[int, float] | None = None,
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        """混合检索。

        - 外部 sparse 模式：必须传 query_sparse（来自 BGE-M3）
        - BM25 模式：忽略 query_sparse，使用 query_text 让 Milvus 内部计算
        """
        dense_data = query_dense.tolist() if hasattr(query_dense, "tolist") else list(query_dense)
        dense_req = AnnSearchRequest(
            data=[dense_data],
            anns_field=DENSE_FIELD,
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
        )

        if self.use_bm25_function:
            sparse_req = AnnSearchRequest(
                data=[query_text],
                anns_field=SPARSE_FIELD,
                param={"metric_type": "BM25", "params": {}},
                limit=top_k,
            )
        else:
            if query_sparse is None:
                raise ValueError("外部 sparse 模式必须传 query_sparse")
            sparse_req = AnnSearchRequest(
                data=[query_sparse],
                anns_field=SPARSE_FIELD,
                param={"metric_type": "IP", "params": {}},
                limit=top_k,
            )

        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(60),
            limit=top_k,
            output_fields=[TEXT_FIELD, META_FIELD],
        )
        hits = results[0] if results else []
        out = []
        for h in hits:
            entity = h.get("entity", {}) if isinstance(h, dict) else {}
            out.append(
                {
                    "score": h.get("distance") if isinstance(h, dict) else None,
                    "text": entity.get(TEXT_FIELD, ""),
                    "meta": entity.get(META_FIELD, {}),
                }
            )
        return out

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
