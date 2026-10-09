"""Unit tests for the BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_lab.data.tokenizer import SPECIAL_TOKENS, BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks and the fox runs away. "
    "hello world, hello there, hello again and again. "
    "a b c d e f g h i j k l m n o p q r s t u v w x y z. "
) * 100


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train([CORPUS], vocab_size=320)


def test_vocab_size(tokenizer: BPETokenizer) -> None:
    assert tokenizer.vocab_size == 320


def test_special_tokens_in_vocab(tokenizer: BPETokenizer) -> None:
    for token in SPECIAL_TOKENS:
        assert tokenizer._tokenizer.token_to_id(token) is not None


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "Hello, World!",
        "line one\nline two\n",
        "café ☕ 你好",
        "numbers 12345 and symbols !@#$%^&*()",
        "   leading and trailing spaces   ",
    ],
)
def test_roundtrip_is_lossless(tokenizer: BPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encoding_compresses(tokenizer: BPETokenizer) -> None:
    assert len(tokenizer.encode(CORPUS)) < len(CORPUS)


def test_save_load_roundtrip(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    reloaded = BPETokenizer.load(path)
    sample = "the quick brown fox"
    assert reloaded.vocab_size == tokenizer.vocab_size
    assert reloaded.encode(sample) == tokenizer.encode(sample)


def test_training_is_deterministic() -> None:
    first = BPETokenizer.train([CORPUS], vocab_size=320)
    second = BPETokenizer.train([CORPUS], vocab_size=320)
    assert first.encode("hello world") == second.encode("hello world")


def test_token_id_lookup(tokenizer: BPETokenizer) -> None:
    assert tokenizer.token_id("<|endoftext|>") is not None
    assert tokenizer.token_id("<not-a-real-token>") is None


def test_encode_batch_matches_encode(tokenizer: BPETokenizer) -> None:
    texts = ["hello world", "the quick brown fox"]
    assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]
