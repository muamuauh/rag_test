from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter

from rag.config import settings
from rag.schema import Chunk


_splitter: SentenceSplitter | None = None


def _get_splitter() -> SentenceSplitter:
    global _splitter
    if _splitter is None:
        _splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    return _splitter


def split_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """对长文本 Chunk 二次切分，image_caption 类型保持不变。"""
    splitter = _get_splitter()
    out: list[Chunk] = []
    for c in chunks:
        if c.chunk_type != "text" or not c.text:
            out.append(c)
            continue
        pieces = splitter.split_text(c.text)
        if not pieces:
            out.append(c)
            continue
        for piece in pieces:
            out.append(
                Chunk(
                    text=piece,
                    source_path=c.source_path,
                    chunk_type="text",
                    page=c.page,
                    image_path=c.image_path,
                    metadata=dict(c.metadata),
                )
            )
    return out
