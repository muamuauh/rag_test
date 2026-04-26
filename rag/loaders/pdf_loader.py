from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from rag.config import settings
from rag.loaders.base import image_hash, is_significant_image
from rag.schema import Chunk


def load_pdf(path: str | Path) -> list[Chunk]:
    """解析 PDF：每页文本一个 Chunk + 内嵌图片各一个 image_caption 占位 Chunk。

    图片字节按 SHA256 落盘到 data/images/<hash>.<ext>，避免重复存储。
    """
    path = Path(path)
    chunks: list[Chunk] = []
    doc = fitz.open(path)
    images_dir = settings.data_images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source_path=str(path),
                    chunk_type="text",
                    page=page_idx + 1,
                    metadata={"file_name": path.name},
                )
            )

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            data: bytes = base["image"]
            ext: str = base.get("ext", "png")
            if not is_significant_image(data):
                continue  # logo / icon / 分隔条 / 解码失败：跳过
            h = image_hash(data)
            img_path = images_dir / f"{h}.{ext}"
            if not img_path.exists():
                img_path.write_bytes(data)
            chunks.append(
                Chunk(
                    text="",  # 由 Stage 3 VLM 填充
                    source_path=str(path),
                    chunk_type="image_caption",
                    page=page_idx + 1,
                    image_path=str(img_path),
                    metadata={"file_name": path.name, "image_hash": h},
                )
            )

    doc.close()
    return chunks
