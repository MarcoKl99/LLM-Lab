"""Unit tests for the data preparation step."""

from __future__ import annotations

from pathlib import Path

import torch

from llm_lab.data.dataset import encode_corpus
from llm_lab.data.prepare import prepare_from_documents, prepare_from_text
from llm_lab.data.tokenizer import BPETokenizer

TEXT = "the quick brown fox jumps over the lazy dog. " * 20


def test_prepare_writes_artifacts_and_splits(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    processed_dir = tmp_path / "processed"

    result = prepare_from_text(
        TEXT,
        vocab_size=300,
        tokenizer_path=tokenizer_path,
        processed_dir=processed_dir,
        val_fraction=0.2,
    )

    assert tokenizer_path.exists()
    train = torch.load(processed_dir / "train.pt")
    val = torch.load(processed_dir / "val.pt")

    tokenizer = BPETokenizer.load(tokenizer_path)
    total = encode_corpus(tokenizer, TEXT)

    assert result.vocab_size == tokenizer.vocab_size
    assert result.train_tokens == len(train)
    assert result.val_tokens == len(val)
    # token ids are stored in a small lossless dtype (int16 for vocab <= 32768)
    assert train.dtype == torch.int16
    assert val.dtype == torch.int16
    assert torch.equal(train.long(), total[: len(train)])
    assert torch.equal(val.long(), total[len(train) :])


def test_prepare_leaves_no_document_cache(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    prepare_from_text(
        TEXT,
        vocab_size=300,
        tokenizer_path=tmp_path / "tokenizer.json",
        processed_dir=processed_dir,
        val_fraction=0.2,
    )

    # the temporary document cache is cleaned up after preparing
    assert not (processed_dir / "documents.jsonl").exists()


def test_prepare_from_documents_separates_with_eos(tmp_path: Path) -> None:
    documents = ["the cat sat", "the dog ran", "a fox jumped"] * 20
    tokenizer_path = tmp_path / "tokenizer.json"
    processed_dir = tmp_path / "processed"

    prepare_from_documents(
        documents,
        vocab_size=300,
        tokenizer_path=tokenizer_path,
        processed_dir=processed_dir,
        val_fraction=0.2,
    )

    tokenizer = BPETokenizer.load(tokenizer_path)
    eos_id = tokenizer.token_id("<|endoftext|>")
    assert eos_id is not None

    expected: list[int] = []
    for document in documents:
        expected += tokenizer.encode(document) + [eos_id]

    train = torch.load(processed_dir / "train.pt")
    val = torch.load(processed_dir / "val.pt")
    total = torch.tensor(expected)

    assert len(train) + len(val) == len(total)
    assert torch.equal(train.long(), total[: len(train)])
    assert torch.equal(val.long(), total[len(train) :])


def test_prepare_reuses_existing_tokenizer(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"

    # The first prepare trains and saves a tokenizer.
    prepare_from_documents(
        [TEXT],
        vocab_size=300,
        tokenizer_path=tokenizer_path,
        processed_dir=tmp_path / "first",
        val_fraction=0.2,
    )
    original = tokenizer_path.read_bytes()

    # The second reuses it: the file is untouched and the tokens are identical.
    result = prepare_from_documents(
        [TEXT],
        vocab_size=300,
        tokenizer_path=tokenizer_path,
        processed_dir=tmp_path / "second",
        val_fraction=0.2,
        reuse_tokenizer=True,
    )

    assert tokenizer_path.read_bytes() == original
    assert result.vocab_size == BPETokenizer.load(tokenizer_path).vocab_size
    assert torch.equal(
        torch.load(tmp_path / "first" / "train.pt"),
        torch.load(tmp_path / "second" / "train.pt"),
    )
