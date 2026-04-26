from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from rag.config import settings
from rag.embedding.embedder import get_embedder
from rag.loaders import load_directory, load_file
from rag.loaders.base import is_significant_image
from rag.parsing.chunker import split_chunks
from rag.parsing.vlm_caption import caption_image
from rag.schema import Chunk
from rag.vector_store.milvus_store import MilvusStore


def _default_vlm_workers() -> int:
    """根据 VLM 端点自动推荐并发数：
    - 本地（Ollama 等）：GPU 受限，并发太多会显存爆，默认 2
    - 云端 API：I/O 密集，默认 8
    """
    base = (settings.vlm_base_url or "").lower()
    if any(host in base for host in ("localhost", "127.0.0.1", "host.docker.internal")):
        return 2
    return 8


def fill_image_captions(
    chunks: list[Chunk],
    use_vlm: bool = True,
    workers: int | None = None,
) -> list[Chunk]:
    """对 image_caption 类型的 Chunk 调用 VLM 填 text 字段。

    - 带 tqdm 进度条
    - 用 ThreadPoolExecutor 并行调 VLM API（I/O 密集，多个请求并发显著加速）
    - 缓存命中（captions.jsonl 里已有）时立即返回，不消耗 API token
    - 失败 / 禁用 VLM / 无 caption 的 image_caption 会被丢弃
    - 非 image 类型 Chunk 原样保留
    """
    if workers is None:
        workers = _default_vlm_workers()

    text_chunks = [c for c in chunks if c.chunk_type != "image_caption"]
    image_chunks = [c for c in chunks if c.chunk_type == "image_caption" and c.image_path]

    if not use_vlm or not image_chunks:
        return text_chunks

    # 二道过滤：兜住老数据（loader 层过滤器加上之前已经入盘的装饰图）
    filtered: list[Chunk] = []
    skipped = 0
    for c in image_chunks:
        try:
            data = Path(c.image_path).read_bytes()  # type: ignore[arg-type]
        except Exception:
            skipped += 1
            continue
        if is_significant_image(data):
            filtered.append(c)
        else:
            skipped += 1
    if skipped:
        print(f"  过滤 {skipped} 张装饰/损坏图（不送 VLM），剩余 {len(filtered)} 张")
    image_chunks = filtered
    if not image_chunks:
        return text_chunks

    captioned: list[Chunk] = []
    failed = 0

    def _worker(c: Chunk) -> tuple[Chunk, str | None, Exception | None]:
        h = c.metadata.get("image_hash", "")
        try:
            return c, caption_image(c.image_path, h), None
        except Exception as e:
            return c, None, e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, c) for c in image_chunks]
        with tqdm(total=len(futures), desc=f"VLM caption (×{workers})", unit="img") as pbar:
            for fut in as_completed(futures):
                c, caption, err = fut.result()
                if err is not None:
                    failed += 1
                    pbar.write(f"[VLM 失败] {c.image_path}: {err}")
                elif caption:
                    c.text = caption
                    captioned.append(c)
                pbar.update(1)

    if failed:
        print(f"  VLM 失败 {failed} 张（已跳过）")
    return text_chunks + captioned


def ingest_paths(
    paths: list[Path],
    use_vlm: bool = True,
    batch_size: int = 32,
    vlm_workers: int | None = None,
) -> int:
    raw_chunks: list[Chunk] = []
    for p in paths:
        if p.is_dir():
            raw_chunks.extend(load_directory(p))
        elif p.is_file():
            raw_chunks.extend(load_file(p))
    if not raw_chunks:
        print("没有可加载的文档")
        return 0
    text_n = sum(1 for c in raw_chunks if c.chunk_type != "image_caption")
    img_n = sum(1 for c in raw_chunks if c.chunk_type == "image_caption")
    print(f"加载到 {len(raw_chunks)} 个原始 Chunk（文本 {text_n}，图片 {img_n}）")

    raw_chunks = fill_image_captions(raw_chunks, use_vlm=use_vlm, workers=vlm_workers)
    print(f"VLM 处理后剩余 {len(raw_chunks)} 个 Chunk")

    chunks = split_chunks(raw_chunks)
    chunks = [c for c in chunks if c.text.strip()]
    print(f"切分后共 {len(chunks)} 个 Chunk")

    embedder = get_embedder()
    store = MilvusStore(
        dense_dim=embedder.dense_dim,
        use_bm25_function=not embedder.has_sparse,
    )

    inserted = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="embed+insert"):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        result = embedder.encode(texts)
        n = store.insert(batch, result.dense_vecs, result.lexical_weights)
        inserted += n
    print(f"已插入 {inserted} 条到 Milvus collection '{settings.milvus_collection}'")
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="入库 pipeline：解析 → caption → 切分 → embed → Milvus")
    parser.add_argument("paths", nargs="+", help="文件或目录路径")
    parser.add_argument("--no-vlm", action="store_true", help="跳过 VLM 图片描述（节省 API 调用）")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    parser.add_argument("--vlm-workers", type=int, default=None,
                        help="VLM 并发数。默认按端点自动选：本地 Ollama=2，云端 API=8")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    ingest_paths(
        paths,
        use_vlm=not args.no_vlm,
        batch_size=args.batch_size,
        vlm_workers=args.vlm_workers,
    )


if __name__ == "__main__":
    main()
