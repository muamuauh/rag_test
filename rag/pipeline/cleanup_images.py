"""清理 data/images/ 中不符合 is_significant_image 标准的图片，
并同步从 captions.jsonl 中删除对应的 caption 缓存条目。

用法：
    python -m rag.pipeline.cleanup_images              # dry-run，只统计不删
    python -m rag.pipeline.cleanup_images --apply      # 真删
    python -m rag.pipeline.cleanup_images --apply --verbose  # 真删 + 打印每张

说明：
    - 命名约定：data/images/<sha256>.<ext> 文件名前缀就是 caption 缓存的 key
    - dry-run 会打印分类直方图与样本，让你确认过滤是否合理
    - 真删后建议 ingest 重跑（已 cache 的好图秒过，不会重复花钱）
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from rag.config import settings
from rag.loaders.base import is_significant_image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _classify(path: Path) -> tuple[str, dict]:
    """返回 (是否保留, 详细信息)。同步 is_significant_image 的判定逻辑，
    但额外返回 fail 原因便于诊断。"""
    info: dict = {"size_kb": path.stat().st_size / 1024}
    try:
        data = path.read_bytes()
    except Exception as e:
        return "read_error", {**info, "err": str(e)}

    info["bytes"] = len(data)
    if len(data) < 5_000:
        return "tiny_bytes", info

    try:
        import numpy as np
        with Image.open(path) as img:
            w, h = img.size
            arr = np.asarray(img.convert("RGB").resize((min(w, 64), min(h, 64))))
        info.update(w=w, h=h, std=float(arr.std()))
    except Exception as e:
        info["err"] = str(e)
        return "decode_error", info

    short = min(w, h)
    long_ = max(w, h)
    info["aspect"] = round(long_ / max(short, 1), 2)

    if short < 128:
        return "narrow", info
    if w * h < 64 * 64 * 4:
        return "few_pixels", info
    if long_ / max(short, 1) > 6.0:
        return "strip", info
    if float(arr.std()) < 12.0:
        return "uniform_color", info
    return "keep", info


def _load_caption_hashes() -> set[str]:
    cache = settings.data_images_dir / "captions.jsonl"
    if not cache.exists():
        return set()
    out = set()
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.add(json.loads(line)["hash"])
        except Exception:
            pass
    return out


def _rewrite_captions(keep_hashes: set[str]) -> int:
    """重写 captions.jsonl，只保留 keep_hashes 中的条目。返回删除数。"""
    cache = settings.data_images_dir / "captions.jsonl"
    if not cache.exists():
        return 0
    kept_lines: list[str] = []
    deleted = 0
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("hash") in keep_hashes:
            kept_lines.append(line)
        else:
            deleted += 1
    backup = cache.with_suffix(".jsonl.bak")
    cache.replace(backup)  # 先备份
    cache.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真删（默认 dry-run）")
    parser.add_argument("--verbose", action="store_true", help="打印每张图的判定")
    parser.add_argument("--dir", default=None, help="覆盖 data/images 路径")
    args = parser.parse_args()

    images_dir = Path(args.dir) if args.dir else settings.data_images_dir
    files = sorted([p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    print(f"扫描 {images_dir}：{len(files)} 张图片")

    counter: Counter[str] = Counter()
    samples: dict[str, list[tuple[Path, dict]]] = {}
    delete_paths: list[Path] = []
    keep_hashes: set[str] = set()

    for p in files:
        verdict, info = _classify(p)
        counter[verdict] += 1
        samples.setdefault(verdict, []).append((p, info))
        if verdict == "keep":
            keep_hashes.add(p.stem)
        else:
            delete_paths.append(p)
        if args.verbose:
            print(f"  [{verdict:>14}] {p.name[:24]}  {info}")

    print()
    print("=== 分类汇总 ===")
    for verdict, n in counter.most_common():
        pct = 100 * n / len(files) if files else 0
        print(f"  {verdict:>15}: {n:>5} 张  ({pct:>5.1f}%)")
        # 打印 3 个样本
        for p, info in samples[verdict][:3]:
            shape = f"{info.get('w','?')}x{info.get('h','?')}"
            std = info.get("std", "?")
            std_s = f"{std:.1f}" if isinstance(std, float) else std
            print(f"    例: {p.name[:24]}  {shape}  {info['size_kb']:.1f}KB  std={std_s}")

    print()
    cap_hashes = _load_caption_hashes()
    cap_to_drop = cap_hashes - keep_hashes
    print(f"captions.jsonl 共 {len(cap_hashes)} 条，将删除 {len(cap_to_drop)} 条对应被过滤图片的缓存")

    if not args.apply:
        print()
        print(f"[dry-run] 共会删除 {len(delete_paths)} 张图 + {len(cap_to_drop)} 条 caption。")
        print("加 --apply 真正执行；加 --verbose 看每张图的判定。")
        return

    # 真删
    print()
    print(f"[apply] 删除 {len(delete_paths)} 张图…")
    deleted_files = 0
    for p in delete_paths:
        try:
            p.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"  删除失败 {p.name}: {e}")
    print(f"  已删除 {deleted_files}/{len(delete_paths)} 张")

    print(f"[apply] 重写 captions.jsonl 并备份原文件为 captions.jsonl.bak …")
    n = _rewrite_captions(keep_hashes)
    print(f"  从 caption 缓存删除 {n} 条")


if __name__ == "__main__":
    main()
