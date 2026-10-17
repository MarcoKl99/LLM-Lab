"""Prepare data for training: train a tokenizer and write token tensors.

Usage:
    uv run python scripts/prepare_data.py --config configs/experiments/lstm_baseline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from llm_lab.config import load_config
from llm_lab.data.prepare import prepare_data

load_dotenv(find_dotenv(), override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data = config["data"]

    result = prepare_data(
        corpus=data["corpus"],
        vocab_size=data["tokenizer"]["vocab_size"],
        tokenizer_path=Path(data["tokenizer"]["output"]),
        raw_dir=Path(data["raw_dir"]),
        processed_dir=Path(data["processed_dir"]),
        val_fraction=data["val_fraction"],
        max_documents=data.get("max_documents"),
        reuse_tokenizer=data["tokenizer"].get("reuse", False),
    )

    print(f"corpus:       {data['corpus']}")
    print(f"chars:        {result.chars:,}")
    print(f"vocab_size:   {result.vocab_size:,}")
    print(f"train tokens: {result.train_tokens:,}")
    print(f"val tokens:   {result.val_tokens:,}")
    print(f"tokenizer  -> {result.tokenizer_path}")
    print(f"tensors    -> {result.processed_dir}/{{train,val}}.pt")


if __name__ == "__main__":
    main()
