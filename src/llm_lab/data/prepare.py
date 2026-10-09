"""Data preparation: train a tokenizer and materialize train/val token tensors.

This is the bridge from raw documents to the artifacts a training run consumes.
Keeping it here (rather than in a script) makes it testable and reusable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from llm_lab.data.corpus import load_documents
from llm_lab.data.dataset import encode_documents, split_train_val
from llm_lab.data.tokenizer import BPETokenizer

EOS_TOKEN = "<|endoftext|>"


@dataclass
class PrepareResult:
    chars: int
    vocab_size: int
    train_tokens: int
    val_tokens: int
    tokenizer_path: Path
    processed_dir: Path


def prepare_from_documents(
    documents: Iterable[str],
    vocab_size: int,
    tokenizer_path: Path,
    processed_dir: Path,
    val_fraction: float,
    eos_token: str | None = EOS_TOKEN,
) -> PrepareResult:
    """Train a tokenizer on ``documents``, save it, and write train/val tensors."""
    documents = list(documents)
    tokenizer = BPETokenizer.train(documents, vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)

    eos_id = tokenizer.token_id(eos_token) if eos_token is not None else None
    tokens = encode_documents(tokenizer, documents, eos_id=eos_id)
    train_tokens, val_tokens = split_train_val(tokens, val_fraction=val_fraction)

    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_tokens, processed_dir / "train.pt")
    torch.save(val_tokens, processed_dir / "val.pt")

    return PrepareResult(
        chars=sum(len(document) for document in documents),
        vocab_size=tokenizer.vocab_size,
        train_tokens=len(train_tokens),
        val_tokens=len(val_tokens),
        tokenizer_path=tokenizer_path,
        processed_dir=processed_dir,
    )


def prepare_from_text(
    text: str,
    vocab_size: int,
    tokenizer_path: Path,
    processed_dir: Path,
    val_fraction: float,
) -> PrepareResult:
    """Prepare a single-document corpus (no document-boundary token)."""
    return prepare_from_documents(
        [text],
        vocab_size=vocab_size,
        tokenizer_path=tokenizer_path,
        processed_dir=processed_dir,
        val_fraction=val_fraction,
        eos_token=None,
    )


def prepare_data(
    corpus: str,
    vocab_size: int,
    tokenizer_path: Path,
    raw_dir: Path,
    processed_dir: Path,
    val_fraction: float,
    max_documents: int | None = None,
) -> PrepareResult:
    """Acquire ``corpus`` and prepare it into tokenizer + tensor artifacts."""
    documents = load_documents(corpus, raw_dir=raw_dir, limit=max_documents)
    return prepare_from_documents(
        documents,
        vocab_size=vocab_size,
        tokenizer_path=tokenizer_path,
        processed_dir=processed_dir,
        val_fraction=val_fraction,
    )
