"""Copy a directory tree to another location (e.g. a Google Drive mount).

Used on Colab to train against fast **local** disk and periodically mirror the
results (checkpoints + TensorBoard logs) to persistent storage. Only files whose
size and modification time differ are copied, so repeated syncs are cheap.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyncResult:
    copied: int
    skipped: int
    bytes_copied: int

    def __str__(self) -> str:
        return (
            f"copied {self.copied} file(s) "
            f"({self.bytes_copied / 1e6:.1f} MB), skipped {self.skipped}"
        )


def _is_same(source: Path, dest: Path) -> bool:
    """True if ``dest`` already matches ``source`` (same size and mtime)."""
    if not dest.exists():
        return False
    src, dst = source.stat(), dest.stat()
    return src.st_size == dst.st_size and int(src.st_mtime) == int(dst.st_mtime)


def sync_tree(
    source: str | Path,
    dest: str | Path,
    *,
    only_newer: bool = True,
) -> SyncResult:
    """Copy every file under ``source`` into ``dest``, preserving structure.

    With ``only_newer`` (default), files already present in ``dest`` with the same
    size and modification time are skipped. ``shutil.copy2`` preserves mtimes, so
    a second sync of unchanged files copies nothing.
    """
    source, dest = Path(source), Path(dest)
    if not source.is_dir():
        raise NotADirectoryError(f"source is not a directory: {source}")

    copied = skipped = bytes_copied = 0
    for src_file in sorted(source.rglob("*")):
        if not src_file.is_file():
            continue
        dst_file = dest / src_file.relative_to(source)
        if only_newer and _is_same(src_file, dst_file):
            skipped += 1
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
        bytes_copied += src_file.stat().st_size

    return SyncResult(copied=copied, skipped=skipped, bytes_copied=bytes_copied)
