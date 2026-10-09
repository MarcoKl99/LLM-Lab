"""Unit tests for the next-token windowing dataset."""

from __future__ import annotations

import pytest
import torch

from llm_lab.data.dataset import (
    LMDataset,
    build_dataloader,
    encode_corpus,
    encode_documents,
    split_train_val,
)
from llm_lab.data.tokenizer import BPETokenizer


def test_len_is_number_of_windows() -> None:
    assert len(LMDataset(torch.arange(10), context_window=4)) == 6


def test_first_window() -> None:
    dataset = LMDataset(torch.arange(10), context_window=4)
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y, torch.tensor([1, 2, 3, 4]))


def test_last_window() -> None:
    dataset = LMDataset(torch.arange(10), context_window=4)
    x, y = dataset[len(dataset) - 1]
    assert torch.equal(x, torch.tensor([5, 6, 7, 8]))
    assert torch.equal(y, torch.tensor([6, 7, 8, 9]))


def test_target_is_input_shifted_by_one() -> None:
    tokens = torch.arange(10)
    dataset = LMDataset(tokens, context_window=4)
    for i in range(len(dataset)):
        x, y = dataset[i]
        assert torch.equal(x, tokens[i : i + dataset.context_window])
        assert torch.equal(y, tokens[i + 1 : i + dataset.context_window + 1])


def test_window_shapes() -> None:
    x, y = LMDataset(torch.arange(10), context_window=4)[0]
    assert x.shape == (4,)
    assert y.shape == (4,)


def test_rejects_non_1d_tokens() -> None:
    with pytest.raises(ValueError):
        LMDataset(torch.zeros(3, 4), context_window=2)


def test_rejects_too_few_tokens() -> None:
    with pytest.raises(ValueError):
        LMDataset(torch.arange(3), context_window=3)


def test_rejects_bad_context_window() -> None:
    with pytest.raises(ValueError):
        LMDataset(torch.arange(10), context_window=0)


def test_split_sizes() -> None:
    train, val = split_train_val(torch.arange(100), val_fraction=0.1)
    assert len(train) == 90
    assert len(val) == 10


def test_split_is_contiguous_tail() -> None:
    tokens = torch.arange(100)
    train, val = split_train_val(tokens, val_fraction=0.2)
    assert torch.equal(train, tokens[:80])
    assert torch.equal(val, tokens[80:])


def test_split_returns_independent_copies() -> None:
    train, val = split_train_val(torch.arange(100), val_fraction=0.2)
    assert train.untyped_storage().nbytes() == len(train) * train.element_size()
    assert val.untyped_storage().nbytes() == len(val) * val.element_size()


def test_rejects_bad_val_fraction() -> None:
    with pytest.raises(ValueError):
        split_train_val(torch.arange(100), val_fraction=0.0)


def test_iteration_yields_every_window_once() -> None:
    dataset = LMDataset(torch.arange(10), context_window=4)
    assert len(list(dataset)) == len(dataset)


def test_getitem_out_of_range_raises() -> None:
    dataset = LMDataset(torch.arange(10), context_window=4)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_negative_index_supported() -> None:
    dataset = LMDataset(torch.arange(10), context_window=4)
    assert torch.equal(dataset[-1][0], dataset[len(dataset) - 1][0])


def test_dataloader_batch_shapes() -> None:
    loader = build_dataloader(torch.arange(10), context_window=4, batch_size=3)
    xb, yb = next(iter(loader))
    assert xb.shape == (3, 4)
    assert yb.shape == (3, 4)


def test_dataloader_target_is_shifted_input() -> None:
    loader = build_dataloader(torch.arange(10), context_window=4, batch_size=2)
    xb, yb = next(iter(loader))
    assert torch.equal(yb[:, :-1], xb[:, 1:])


def test_dataloader_number_of_batches() -> None:
    loader = build_dataloader(torch.arange(10), context_window=4, batch_size=4)
    assert len(loader) == 2


def test_encode_corpus_matches_tokenizer() -> None:
    tokenizer = BPETokenizer.train(["hello hello hello world"], vocab_size=270)
    text = "hello world"
    encoded = encode_corpus(tokenizer, text)
    assert encoded.dtype == torch.long
    assert encoded.tolist() == tokenizer.encode(text)


def test_encode_documents_without_eos_concatenates() -> None:
    tokenizer = BPETokenizer.train(["hello world " * 20], vocab_size=300)
    tokens = encode_documents(tokenizer, ["hello", "world"])
    assert tokens.tolist() == tokenizer.encode("hello") + tokenizer.encode("world")


def test_encode_documents_inserts_eos_between_documents() -> None:
    tokenizer = BPETokenizer.train(["hello world " * 20], vocab_size=300)
    eos_id = tokenizer.token_id("<|endoftext|>")
    assert eos_id is not None

    tokens = encode_documents(tokenizer, ["hello", "world"], eos_id=eos_id)
    expected = (
        tokenizer.encode("hello")
        + [eos_id]
        + tokenizer.encode("world")
        + [eos_id]
    )
    assert tokens.tolist() == expected
