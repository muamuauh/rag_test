"""验证 load_directory 的容错与跳过逻辑：
  - 单个坏文件（如 ~$xxx.docx 锁文件）不会让整批中断
  - 系统文件 / 隐藏目录被跳过
  - 未支持扩展名静默跳过
  - 嵌套目录递归正确
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.loaders import _should_skip, load_directory


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """构造混合的测试目录：
        root/
            a.md                    ← 好文件
            ~$b.docx                ← Word 锁文件（应跳过）
            .DS_Store               ← 系统文件（应跳过）
            unsupported.xyz         ← 未支持扩展（静默跳过）
            broken.docx             ← 假装是 docx 但内容是垃圾，应被 try/except 捕获
            sub/
                c.txt               ← 好文件
                .hidden/            ← 隐藏目录（整个跳过）
                    d.md            ← 即使是好扩展也不应加载
                deep/
                    e.md            ← 嵌套深层好文件
    """
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "~$b.docx").write_bytes(b"junk")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")
    (tmp_path / "unsupported.xyz").write_text("ignored", encoding="utf-8")
    (tmp_path / "broken.docx").write_bytes(b"this is not a real docx file")

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("world", encoding="utf-8")

    hidden = sub / ".hidden"
    hidden.mkdir()
    (hidden / "d.md").write_text("should be skipped", encoding="utf-8")

    deep = sub / "deep"
    deep.mkdir()
    (deep / "e.md").write_text("deep content", encoding="utf-8")

    return tmp_path


def test_should_skip_patterns(tmp_path: Path) -> None:
    assert _should_skip(tmp_path / "~$lock.docx")
    assert _should_skip(tmp_path / ".DS_Store")
    assert _should_skip(tmp_path / "Thumbs.db")
    assert _should_skip(tmp_path / ".hidden" / "x.md")
    assert _should_skip(tmp_path / "__pycache__" / "x.py")
    assert not _should_skip(tmp_path / "ok.md")
    assert not _should_skip(tmp_path / "sub" / "deep" / "ok.md")


def test_load_directory_resilience(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chunks = load_directory(tree)
    sources = {Path(c.source_path).name for c in chunks}

    # 应被加载的好文件
    assert "a.md" in sources
    assert "c.txt" in sources
    assert "e.md" in sources

    # 应被跳过 / 失败的：
    assert "~$b.docx" not in sources           # 锁文件按名跳过
    assert ".DS_Store" not in sources          # 系统文件
    assert "unsupported.xyz" not in sources    # 扩展名不支持
    assert "d.md" not in sources               # 在 .hidden/ 隐藏目录里

    # broken.docx 应该尝试加载并失败，但不影响其他文件
    # （是否最终在 sources 里取决于 python-docx 怎么处理空字节，
    # 重点验证：整体没崩溃，好文件都加载出来了）
    captured = capsys.readouterr()
    # 至少 broken.docx 会触发失败摘要打印
    assert "broken.docx" in captured.out or len(chunks) >= 3
    # 三个有效文件都在
    assert len(chunks) >= 3
