"""Prepare data for training: train a tokenizer and write token tensors.

Usage:
    uv run python scripts/prepare_data.py --config configs/data/tiny_shakespeare.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.config import load_config
from llm_lab.data.prepare import prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    result = prepare_data(
        corpus=config["corpus"],
        vocab_size=config["tokenizer"]["vocab_size"],
        tokenizer_path=Path(config["tokenizer"]["output"]),
        raw_dir=Path(config["raw_dir"]),
        processed_dir=Path(config["processed_dir"]),
        val_fraction=config["val_fraction"],
    )

    print(f"corpus:       {config['corpus']}")
    print(f"chars:        {result.chars:,}")
    print(f"vocab_size:   {result.vocab_size:,}")
    print(f"train tokens: {result.train_tokens:,}")
    print(f"val tokens:   {result.val_tokens:,}")
    print(f"tokenizer  -> {result.tokenizer_path}")
    print(f"tensors    -> {result.processed_dir}/{{train,val}}.pt")


if __name__ == "__main__":
    main()
