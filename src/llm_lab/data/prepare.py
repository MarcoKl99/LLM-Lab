"""Data preparation: train a tokenizer and materialize train/val token tensors.

This is the bridge from raw text to the artifacts a training run consumes.
Keeping it here (rather than in a script) makes it testable and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from llm_lab.data.corpus import load_text
from llm_lab.data.dataset import encode_corpus, split_train_val
from llm_lab.data.tokenizer import BPETokenizer


@dataclass
class PrepareResult:
    chars: int
    vocab_size: int
    train_tokens: int
    val_tokens: int
    tokenizer_path: Path
    processed_dir: Path


def prepare_from_text(
    text: str,
    vocab_size: int,
    tokenizer_path: Path,
    processed_dir: Path,
    val_fraction: float,
) -> PrepareResult:
    """Train a tokenizer on ``text``, save it, and write train/val tensors."""
    tokenizer = BPETokenizer.train([text], vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)

    tokens = encode_corpus(tokenizer, text)
    train_tokens, val_tokens = split_train_val(tokens, val_fraction=val_fraction)

    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_tokens, processed_dir / "train.pt")
    torch.save(val_tokens, processed_dir / "val.pt")

    return PrepareResult(
        chars=len(text),
        vocab_size=tokenizer.vocab_size,
        train_tokens=len(train_tokens),
        val_tokens=len(val_tokens),
        tokenizer_path=tokenizer_path,
        processed_dir=processed_dir,
    )


def prepare_data(
    corpus: str,
    vocab_size: int,
    tokenizer_path: Path,
    raw_dir: Path,
    processed_dir: Path,
    val_fraction: float,
) -> PrepareResult:
    """Acquire ``corpus`` and prepare it into tokenizer + tensor artifacts."""
    text = load_text(corpus, raw_dir=raw_dir)
    return prepare_from_text(
        text,
        vocab_size=vocab_size,
        tokenizer_path=tokenizer_path,
        processed_dir=processed_dir,
        val_fraction=val_fraction,
    )
