"""Mirror a directory tree to another location (e.g. a Google Drive mount).

Useful on Colab: train against fast local disk, then sync the run directory to
persistent storage. Only changed files are copied, so it is safe to run often.

Usage:
    uv run python scripts/sync_artifacts.py \
        --source artifacts/runs/scaling/xl \
        --dest /content/drive/MyDrive/LLM-Lab/artifacts/runs/scaling/xl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.utils.sync import sync_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a directory tree.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Copy every file (skip the size/mtime 'only newer' check).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = sync_tree(args.source, args.dest, only_newer=not args.all)
    print(f"{args.source} -> {args.dest}: {result}")


if __name__ == "__main__":
    main()
