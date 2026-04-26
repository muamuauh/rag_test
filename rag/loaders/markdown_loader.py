from __future__ import annotations

from pathlib import Path

from rag.schema import Chunk


def load_text(path: str | Path) -> list[Chunk]:
    """Markdown / TXT：整文件作为单个文本 Chunk，交给 chunker 切分。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [
        Chunk(
            text=text,
            source_path=str(path),
            chunk_type="text",
            metadata={"file_name": path.name},
        )
    ]
