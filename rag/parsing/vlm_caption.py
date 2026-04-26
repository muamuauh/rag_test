from __future__ import annotations

import base64
import json
import logging
import threading
import time
from io import BytesIO
from pathlib import Path

import litellm
from PIL import Image

from rag.config import settings


logger = logging.getLogger(__name__)


CAPTION_PROMPT = (
    "你是一个图像描述助手。请用中文详细描述这张图片的内容，"
    "如果是图表（折线图/柱状图/流程图/示意图等），请说明：图表类型、坐标轴/字段含义、"
    "关键数据趋势、以及可能的结论。如果图中含有文字，请尽量原样转录。"
    "回复仅包含描述本身，不要前缀。"
)

# 发送给 VLM 前的预处理参数
VLM_MAX_EDGE = 1024      # 长边压到 1024 px（OpenAI low detail 会再缩到 512，留点余量）
VLM_JPEG_QUALITY = 85    # JPEG 压缩质量，对 RAG 描述场景质量足够
VLM_TIMEOUT = 60         # 单次调用超时（秒）；本地 VLM 可适当调大
VLM_NUM_RETRIES = 6      # 自动重试次数（429/网络/5xx 触发指数回退；6 次覆盖 ~63s）


class _TokenBucket:
    """跨线程的请求级限速器。所有 worker 共享一个实例。

    每个请求前调用 acquire()，确保平均速率不超过 max_rps。
    实现：固定间隔的"下一次允许时间"，先到先排。
    """

    def __init__(self, max_rps: float) -> None:
        self._interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now += wait
            self._next = max(self._next, now) + self._interval


_limiter: _TokenBucket | None = None
_limiter_lock = threading.Lock()


def _get_limiter() -> _TokenBucket | None:
    global _limiter
    rps = settings.vlm_max_rps or 0.0
    if rps <= 0:
        return None
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = _TokenBucket(rps)
                logger.info("VLM 限速器启用：%.2f req/s", rps)
    return _limiter


def _cache_path() -> Path:
    return settings.data_images_dir / "captions.jsonl"


_lock = threading.Lock()
_cache: dict[str, str] | None = None


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache
    cache: dict[str, str] = {}
    p = _cache_path()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                cache[obj["hash"]] = obj["caption"]
            except Exception:
                continue
    _cache = cache
    return cache


def _append_cache(image_hash: str, caption: str) -> None:
    cache = _load_cache()
    with _lock:
        cache[image_hash] = caption
        with _cache_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": image_hash, "caption": caption}, ensure_ascii=False) + "\n")


def _preprocess_for_vlm(image_path: Path) -> tuple[bytes, str]:
    """读图 → 缩放到 VLM_MAX_EDGE → JPEG 压缩 → (bytes, mime)。

    收益（OpenAI gpt-4o-mini 视觉为例）：
    - 减少 base64 体积 5-20× → 上行更快、prompt token 显著下降
    - 配合 detail:low 单图固定 85 prompt token，速度成本都降一个量级

    解码失败时回退到原始字节。
    """
    try:
        with Image.open(image_path) as img:
            # JPEG 不支持 RGBA / P，先转 RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            scale = VLM_MAX_EDGE / max(w, h)
            if scale < 1.0:
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=VLM_JPEG_QUALITY, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except Exception as e:
        logger.warning("图片预处理失败 %s: %s；回退到原始字节", image_path, e)
        return image_path.read_bytes(), _mime_for(image_path)


def _mime_for(image_path: Path) -> str:
    ext = image_path.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"image/{ext or 'png'}"


def caption_image(image_path: str | Path, image_hash: str) -> str:
    """对图片调用 VLM 生成中文描述；按 image_hash 缓存。"""
    image_path = Path(image_path)
    cache = _load_cache()
    if image_hash in cache:
        return cache[image_hash]

    img_bytes, mime = _preprocess_for_vlm(image_path)
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    extra: dict = {}
    if settings.vlm_api_key:
        extra["api_key"] = settings.vlm_api_key
    if settings.vlm_base_url:
        extra["api_base"] = settings.vlm_base_url

    # 主动限速：到点了才发请求，避免触发 provider 的 RPM/TPM 上限
    limiter = _get_limiter()
    if limiter is not None:
        limiter.acquire()

    resp = litellm.completion(
        model=settings.vlm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    # detail=low：OpenAI 把图缩到 512×512、固定 85 prompt token
                    # 大多数 provider（智谱/通义/AIHubMix 转发）都识别此字段，未识别也只是被忽略
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                ],
            }
        ],
        timeout=VLM_TIMEOUT,
        num_retries=VLM_NUM_RETRIES,
        **extra,
    )
    caption = resp["choices"][0]["message"]["content"].strip()
    _append_cache(image_hash, caption)
    return caption
