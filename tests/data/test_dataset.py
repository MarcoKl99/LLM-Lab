"""Unit tests for the next-token windowing dataset."""

from __future__ import annotations

import itertools

import pytest
import torch

from llm_lab.data.dataset import (
    LMDataset,
    build_dataloader,
    encode_corpus,
    encode_documents,
    limit_tokens,
    split_train_val,
    storage_dtype,
)
from llm_lab.data.sampler import ChunkedRandomSampler, ShuffledChunkSampler
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


def test_limit_tokens_truncates_to_prefix() -> None:
    tokens = torch.arange(100)
    assert torch.equal(limit_tokens(tokens, 30), tokens[:30])


def test_limit_tokens_none_returns_all() -> None:
    tokens = torch.arange(100)
    assert torch.equal(limit_tokens(tokens, None), tokens)


def test_limit_tokens_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        limit_tokens(torch.arange(10), 0)


def test_storage_dtype_picks_smallest_lossless() -> None:
    assert storage_dtype(1024) == torch.int16
    assert storage_dtype(2**15) == torch.int16  # ids 0..32767
    assert storage_dtype(2**15 + 1) == torch.int32
    assert storage_dtype(100_000) == torch.int32


def test_dataset_casts_windows_back_to_long() -> None:
    tokens = torch.arange(10, dtype=torch.int16)
    x, y = LMDataset(tokens, context_window=4)[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert torch.equal(x, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y, torch.tensor([1, 2, 3, 4]))


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


def test_shuffled_chunk_sampler_is_a_permutation() -> None:
    sampler = ShuffledChunkSampler(1000, chunk_size=128)

    indices = list(sampler)

    assert len(indices) == 1000
    assert len(sampler) == 1000
    assert sorted(indices) == list(range(1000))


def test_shuffled_chunk_sampler_handles_uneven_final_chunk() -> None:
    sampler = ShuffledChunkSampler(1000, chunk_size=256)

    indices = list(sampler)

    assert sorted(indices) == list(range(1000))


def test_shuffled_chunk_sampler_is_deterministic_with_generator() -> None:
    first = list(
        ShuffledChunkSampler(
            200, generator=torch.Generator().manual_seed(0), chunk_size=32
        )
    )
    second = list(
        ShuffledChunkSampler(
            200, generator=torch.Generator().manual_seed(0), chunk_size=32
        )
    )

    assert first == second


def test_shuffled_dataloader_yields_batches() -> None:
    loader = build_dataloader(
        torch.arange(100), context_window=4, batch_size=8, shuffle=True
    )

    xb, yb = next(iter(loader))

    assert xb.shape == (8, 4)
    assert yb.shape == (8, 4)


def test_chunked_random_sampler_yields_valid_indices() -> None:
    sampler = ChunkedRandomSampler(1000, chunk_size=128)

    indices = list(sampler)

    assert len(indices) == 1000
    assert len(sampler) == 1000
    assert all(0 <= index < 1000 for index in indices)


def test_chunked_random_sampler_mixes_globally() -> None:
    # Regression guard: a bounded draw must already span the whole range, because
    # our runs see far less than one epoch. (A shard/block shuffle fails this.)
    sampler = ChunkedRandomSampler(1_000_000, chunk_size=4096)

    indices = list(itertools.islice(iter(sampler), 4096))

    assert max(indices) - min(indices) > 500_000


def test_shuffled_chunk_sampler_is_clustered_early() -> None:
    # Documents *why* ShuffledChunkSampler is not the default: with a large chunk
    # it reads one contiguous region until that chunk is exhausted, so early
    # training only ever sees a tiny slice of the corpus.
    sampler = ShuffledChunkSampler(1_000_000, chunk_size=100_000)

    indices = list(itertools.islice(iter(sampler), 5000))

    assert max(indices) - min(indices) < 100_000  # confined to a single chunk


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
    # stored in the smallest lossless dtype (int16 for a small vocab)
    assert encoded.dtype == torch.int16
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
