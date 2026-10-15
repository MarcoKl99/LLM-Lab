"""Data preparation: train a tokenizer and materialize train/val token tensors.

This is the bridge from raw documents to the artifacts a training run consumes.
Keeping it here (rather than in a script) makes it testable and reusable.

Large corpora are handled with **bounded memory**: the documents are streamed to
a temporary cache file once, then read back twice — once to train the tokenizer,
once to encode. We never hold the whole corpus (or a full document list) in RAM.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from llm_lab.data.corpus import load_documents
from llm_lab.data.dataset import encode_documents, split_train_val, storage_dtype
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


def _cache_documents(documents: Iterable[str], path: Path) -> int:
    """Stream ``documents`` to ``path`` as JSONL; return the total char count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chars = 0
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document) + "\n")
            chars += len(document)
    return chars


def _read_cached(path: Path) -> Iterator[str]:
    """Yield the documents cached by :func:`_cache_documents` (streaming)."""
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def prepare_from_documents(
    documents: Iterable[str],
    vocab_size: int,
    tokenizer_path: Path,
    processed_dir: Path,
    val_fraction: float,
    eos_token: str | None = EOS_TOKEN,
) -> PrepareResult:
    """Train a tokenizer on ``documents``, save it, and write train/val tensors.

    Documents are streamed to a temporary cache so they can be read twice
    (tokenizer training, then encoding) while keeping memory bounded.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / "documents.jsonl"
    try:
        chars = _cache_documents(documents, cache_path)

        tokenizer = BPETokenizer.train(_read_cached(cache_path), vocab_size=vocab_size)
        tokenizer.save(tokenizer_path)

        eos_id = tokenizer.token_id(eos_token) if eos_token is not None else None
        tokens = encode_documents(tokenizer, _read_cached(cache_path), eos_id=eos_id)
    finally:
        cache_path.unlink(missing_ok=True)

    train_tokens, val_tokens = split_train_val(tokens, val_fraction=val_fraction)

    # Store token ids in the smallest lossless dtype (int16 for vocab <= 32768),
    # which makes the token files 4x smaller than int64.
    dtype = storage_dtype(tokenizer.vocab_size)
    train_tokens, val_tokens = train_tokens.to(dtype), val_tokens.to(dtype)

    torch.save(train_tokens, processed_dir / "train.pt")
    torch.save(val_tokens, processed_dir / "val.pt")

    return PrepareResult(
        chars=chars,
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
