from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path


def file_id(path: str | Path) -> str:
    """文件路径 + 大小 + mtime 的稳定 hash，用作文档 ID。"""
    p = Path(path)
    stat = p.stat()
    raw = f"{p.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def image_hash(data: bytes) -> str:
    """图片字节内容的 SHA256，用作 caption 缓存 key。"""
    return hashlib.sha256(data).hexdigest()


def is_significant_image(
    data: bytes,
    *,
    min_bytes: int = 5_000,
    min_short_edge: int = 128,
    min_pixels: int = 64 * 64 * 4,
    max_aspect: float = 6.0,
    min_std: float = 12.0,
) -> bool:
    """判断图片是否值得做 VLM 描述。过滤 PDF/PPT 里常见的装饰性垃圾：

    - 字节数 < 5KB：基本是 logo / icon
    - 短边 < 128px：分隔条 / 装饰小图
    - 总像素 < 16K：信息量太少
    - 长宽比 > 6:1：典型页眉/页脚条带（如 1900x192）
    - RGB 颜色 std < 12：近乎纯色块（空白背景 / 单色填充）

    解码失败也判为"不值得"（后续 VLM 也会失败）。
    """
    if len(data) < min_bytes:
        return False
    try:
        # 局部 import，loaders/base.py 不强依赖 PIL/numpy
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        with Image.open(BytesIO(data)) as img:
            w, h = img.size
            # 缩到最多 64×64 算颜色方差，避免大图占内存
            arr = np.asarray(
                img.convert("RGB").resize((min(w, 64), min(h, 64)))
            )
    except Exception:
        return False

    short_edge = min(w, h)
    long_edge = max(w, h)

    if short_edge < min_short_edge:
        return False
    if w * h < min_pixels:
        return False
    if long_edge / max(short_edge, 1) > max_aspect:
        return False
    if float(arr.std()) < min_std:
        return False
    return True
