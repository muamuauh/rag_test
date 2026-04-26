from __future__ import annotations

import logging
from pathlib import Path

from rag.loaders.image_loader import load_image
from rag.loaders.markdown_loader import load_text
from rag.loaders.office_loader import load_docx, load_pptx
from rag.loaders.pdf_loader import load_pdf
from rag.schema import Chunk

logger = logging.getLogger(__name__)

TEXT_EXT = {".md", ".markdown", ".txt"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# 名字以这些开头的文件直接跳过：Office 锁文件、系统文件、临时文件
_SKIP_NAME_PREFIX = ("~$", ".~", ".DS_Store", "Thumbs.db")


def _should_skip(path: Path) -> bool:
    name = path.name
    if name.startswith(_SKIP_NAME_PREFIX):
        return True
    # 隐藏目录：路径中任一段以 "." 开头（.git/、.idea/、__pycache__ 等）
    for part in path.parts:
        if part.startswith(".") and part not in (".", ".."):
            return True
        if part == "__pycache__":
            return True
    return False


def load_file(path: str | Path) -> list[Chunk]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(p)
    if suffix == ".docx":
        return load_docx(p)
    if suffix == ".pptx":
        return load_pptx(p)
    if suffix in TEXT_EXT:
        return load_text(p)
    if suffix in IMAGE_EXT:
        return load_image(p)
    return []  # 未支持的格式静默跳过


def load_directory(root: str | Path) -> list[Chunk]:
    """递归扫描目录，加载所有支持的文件。

    - 单个文件解析失败不会影响整批：异常被捕获并记录，继续处理下一个
    - 自动跳过：Office 锁文件 (~$xxx.docx)、系统文件 (.DS_Store/Thumbs.db)、隐藏目录
    - 未支持的扩展名（如 .xyz）静默跳过
    """
    root = Path(root)
    chunks: list[Chunk] = []
    failed: list[tuple[Path, Exception]] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _should_skip(p):
            logger.debug("skip %s", p)
            continue
        try:
            chunks.extend(load_file(p))
        except Exception as e:
            failed.append((p, e))
            logger.warning("加载失败 %s: %s", p, e)

    if failed:
        print(f"[load_directory] 共 {len(failed)} 个文件加载失败：")
        for p, e in failed:
            print(f"  - {p}  →  {type(e).__name__}: {e}")

    return chunks


__all__ = [
    "load_file",
    "load_directory",
    "load_pdf",
    "load_docx",
    "load_pptx",
    "load_text",
    "load_image",
]
