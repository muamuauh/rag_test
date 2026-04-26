from __future__ import annotations

import shutil
from pathlib import Path

from rag.config import settings
from rag.loaders.base import image_hash
from rag.schema import Chunk


def load_image(path: str | Path) -> list[Chunk]:
    """独立图片：复制到 data/images/<hash>.<ext> 并产出 image_caption 占位 Chunk。"""
    path = Path(path)
    data = path.read_bytes()
    h = image_hash(data)
    ext = path.suffix.lstrip(".").lower() or "png"
    images_dir = settings.data_images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    img_path = images_dir / f"{h}.{ext}"
    if not img_path.exists():
        shutil.copyfile(path, img_path)
    return [
        Chunk(
            text="",
            source_path=str(path),
            chunk_type="image_caption",
            image_path=str(img_path),
            metadata={"file_name": path.name, "image_hash": h},
        )
    ]
