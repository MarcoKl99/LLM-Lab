"""Inspect how a processed token file is laid out.

Shows that ``train.pt`` / ``val.pt`` are a single flat stream of token ids with
an ``<|endoftext|>`` marker separating documents (stories), and prints the first
few documents alongside their token ids.

Usage:
    uv run python scripts/inspect_tokens.py \
        --processed-dir data/processed/tinystories \
        --tokenizer artifacts/tokenizer/tinystories_bpe.json \
        --num-documents 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_lab.data.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a processed token file.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/tinystories"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("artifacts/tokenizer/tinystories_bpe.json")
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-documents", type=int, default=3)
    parser.add_argument("--max-ids", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = torch.load(args.processed_dir / f"{args.split}.pt")
    tokenizer = BPETokenizer.load(args.tokenizer)
    eos_id = tokenizer.token_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer has no <|endoftext|> token")

    eos_positions = (tokens == eos_id).nonzero(as_tuple=True)[0].tolist()

    print(f"file:       {args.processed_dir / f'{args.split}.pt'}")
    print(f"shape:      {tuple(tokens.shape)}  dtype: {tokens.dtype}")
    print(f"eos token:  id {eos_id} = '<|endoftext|>'  ({len(eos_positions):,} occurrences)")
    print()

    preview = tokens[:24].tolist()
    print("flat stream (first 24 tokens, eos shown as ▮):")
    print("  " + " ".join("▮" if i == eos_id else str(i) for i in preview))
    print()

    starts = [0] + [position + 1 for position in eos_positions]
    ends = eos_positions + [len(tokens)]
    count = min(args.num_documents, len(starts))

    print(f"layout: [doc 0] ▮ [doc 1] ▮ [doc 2] ▮ ...   (showing first {count} documents)")
    print()
    for index in range(count):
        ids = tokens[starts[index] : ends[index]].tolist()
        text = tokenizer.decode(ids)
        shown = ids[: args.max_ids]
        suffix = " ..." if len(ids) > args.max_ids else ""
        print(f"--- document {index}: {len(ids)} tokens ---")
        print(f"text: {text!r}")
        print(f"ids:  {shown}{suffix}")
        print(f"delimiter after: {eos_id}  (<|endoftext|>)")
        print()


if __name__ == "__main__":
    main()
