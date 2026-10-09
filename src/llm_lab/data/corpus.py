"""Raw corpus acquisition.

Two shapes of corpus are supported behind :func:`load_documents`:

* single-document sources (e.g. Tiny Shakespeare), and
* multi-document sources streamed from HuggingFace ``datasets`` (e.g.
  TinyStories).

``load_documents`` yields strings so the rest of the pipeline treats every
corpus uniformly and can insert document boundaries where appropriate.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SOURCES: dict[str, str] = {
    "tiny_shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/"
        "master/data/tinyshakespeare/input.txt"
    ),
}


def download_text(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest``, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_text(name: str = "tiny_shakespeare", raw_dir: Path = RAW_DIR) -> str:
    """Return the raw text of a known single-file corpus, downloading on first use."""
    if name not in SOURCES:
        raise ValueError(f"Unknown corpus {name!r}. Available: {sorted(SOURCES)}")

    path = raw_dir / f"{name}.txt"
    if not path.exists():
        download_text(SOURCES[name], path)
    return path.read_text(encoding="utf-8")


def _shakespeare_documents(raw_dir: Path, limit: int | None) -> Iterator[str]:
    yield load_text("tiny_shakespeare", raw_dir=raw_dir)


def _tinystories_documents(raw_dir: Path, limit: int | None) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield example["text"]


DOCUMENT_LOADERS: dict[str, Callable[[Path, int | None], Iterator[str]]] = {
    "tiny_shakespeare": _shakespeare_documents,
    "tinystories": _tinystories_documents,
}


def load_documents(
    name: str,
    raw_dir: Path = RAW_DIR,
    limit: int | None = None,
) -> Iterator[str]:
    """Yield the documents of a known corpus (streamed where supported)."""
    if name not in DOCUMENT_LOADERS:
        raise ValueError(
            f"Unknown corpus {name!r}. Available: {sorted(DOCUMENT_LOADERS)}"
        )
    return DOCUMENT_LOADERS[name](raw_dir, limit)
