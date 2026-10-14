"""Print random document(s) from a corpus, for manual inspection.

Streams documents and reservoir-samples a few (uniformly, in one pass), then
prints them. Handy for eyeballing what a corpus actually looks like.

Usage:
    uv run python scripts/show_document.py --corpus fineweb_edu --samples 2
    uv run python scripts/show_document.py --corpus fineweb_edu --scan 50000 --chars 2000
"""

from __future__ import annotations

import argparse
import random
from itertools import islice

from llm_lab.data.corpus import load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show random corpus documents.")
    parser.add_argument("--corpus", default="fineweb_edu")
    parser.add_argument("--samples", type=int, default=1, help="How many to show.")
    parser.add_argument(
        "--scan",
        type=int,
        default=20000,
        help="How many documents to stream while sampling (a random one of these).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--chars",
        type=int,
        default=0,
        help="Truncate each document to this many characters (0 = full).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    reservoir: list[str] = []
    seen = 0
    for index, document in enumerate(islice(load_documents(args.corpus), args.scan)):
        seen = index + 1
        if index < args.samples:
            reservoir.append(document)
        else:
            j = rng.randint(0, index)
            if j < args.samples:
                reservoir[j] = document

    print(f"corpus: {args.corpus}   sampled {len(reservoir)} of {seen:,} streamed documents")
    for document in reservoir:
        body = document if args.chars <= 0 else document[: args.chars]
        print("=" * 78)
        print(f"[{len(document):,} chars]")
        print("-" * 78)
        print(body)
        print()


if __name__ == "__main__":
    main()
