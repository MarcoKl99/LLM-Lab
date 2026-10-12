"""Unit tests for directory syncing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_lab.utils.sync import sync_tree


def _make_tree(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("hello", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("world", encoding="utf-8")


def test_sync_copies_all_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _make_tree(src)
    dst = tmp_path / "dst"

    result = sync_tree(src, dst)

    assert result.copied == 2
    assert result.skipped == 0
    assert (dst / "a.txt").read_text(encoding="utf-8") == "hello"
    assert (dst / "sub" / "b.txt").read_text(encoding="utf-8") == "world"


def test_sync_skips_unchanged_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _make_tree(src)
    dst = tmp_path / "dst"

    first = sync_tree(src, dst)
    second = sync_tree(src, dst)

    assert first.copied == 2
    assert second.copied == 0
    assert second.skipped == 2


def test_sync_recopies_changed_file(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _make_tree(src)
    dst = tmp_path / "dst"
    sync_tree(src, dst)

    (src / "a.txt").write_text("changed!", encoding="utf-8")
    os.utime(src / "a.txt", (1000, 1000))  # force a distinct mtime

    result = sync_tree(src, dst)

    assert result.copied == 1
    assert (dst / "a.txt").read_text(encoding="utf-8") == "changed!"


def test_sync_only_newer_false_copies_everything(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _make_tree(src)
    dst = tmp_path / "dst"
    sync_tree(src, dst)

    result = sync_tree(src, dst, only_newer=False)

    assert result.copied == 2


def test_sync_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        sync_tree(tmp_path / "nope", tmp_path / "dst")
