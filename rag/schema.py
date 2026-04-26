from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChunkType = Literal["text", "image_caption", "table"]


@dataclass
class Chunk:
    """统一的文档片段数据结构。所有 loader/parser 输出 Chunk 列表。"""

    text: str
    source_path: str
    chunk_type: ChunkType = "text"
    page: int | None = None
    image_path: str | None = None
    metadata: dict = field(default_factory=dict)
