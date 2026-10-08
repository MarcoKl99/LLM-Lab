"""Raw corpus acquisition.

Downloads a corpus once into ``data/raw/`` and caches it on disk. The
acquisition details are kept behind :func:`load_text` so that a different
source (e.g. HuggingFace ``datasets``) can be added later without touching
the rest of the data pipeline.
"""

from __future__ import annotations

import urllib.request
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
    """Return the raw text of a known corpus, downloading it on first use."""
    if name not in SOURCES:
        raise ValueError(f"Unknown corpus {name!r}. Available: {sorted(SOURCES)}")

    path = raw_dir / f"{name}.txt"
    if not path.exists():
        download_text(SOURCES[name], path)
    return path.read_text(encoding="utf-8")
