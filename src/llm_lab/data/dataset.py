"""Turn a flat token stream into next-token prediction windows.

A language model learns by predicting the next token, so training examples are
``(input, target)`` pairs where ``target`` is ``input`` shifted right by one.
This module produces those pairs from a 1D tensor of token ids.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from llm_lab.data.tokenizer import BPETokenizer


def storage_dtype(vocab_size: int) -> torch.dtype:
    """Smallest integer dtype that holds every token id in ``[0, vocab_size)``.

    Token ids are stored on disk in the smallest **lossless** dtype (int16 for a
    vocab up to 32768), which shrinks the token files 4x versus int64. The model
    still sees int64 ids — :class:`LMDataset` casts each window back to ``long``.
    """
    if vocab_size <= 2**15:  # ids 0..32767 fit a signed int16
        return torch.int16
    if vocab_size <= 2**31:
        return torch.int32
    return torch.int64


def encode_documents(
    tokenizer: BPETokenizer,
    documents: Iterable[str],
    eos_id: int | None = None,
    batch_size: int = 1000,
) -> Tensor:
    """Encode documents into one 1D ``LongTensor``.

    If ``eos_id`` is given, it is appended after every document so the model
    learns document boundaries rather than predicting across unrelated texts.
    """
    dtype = storage_dtype(tokenizer.vocab_size)
    pieces: list[Tensor] = []
    batch: list[str] = []

    def flush() -> None:
        if not batch:
            return
        flat: list[int] = []
        for ids in tokenizer.encode_batch(batch):
            flat.extend(ids)
            if eos_id is not None:
                flat.append(eos_id)
        pieces.append(torch.tensor(flat, dtype=dtype))
        batch.clear()

    for text in documents:
        batch.append(text)
        if len(batch) == batch_size:
            flush()
    flush()

    if not pieces:
        return torch.empty(0, dtype=dtype)
    return torch.cat(pieces)


def encode_corpus(tokenizer: BPETokenizer, text: str) -> Tensor:
    """Encode a single text into a 1D ``LongTensor`` of token ids."""
    return encode_documents(tokenizer, [text])


def limit_tokens(tokens: Tensor, max_tokens: int | None) -> Tensor:
    """Return the first ``max_tokens`` of ``tokens`` (or all of them if ``None``).

    Used to vary the training-data budget while keeping the tokenizer and the
    train/val split fixed: every run then sees a nested prefix of the same
    stream.
    """
    if max_tokens is None:
        return tokens
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    return tokens[:max_tokens]


def split_train_val(
    tokens: Tensor, val_fraction: float = 0.1
) -> tuple[Tensor, Tensor]:
    """Split a 1D token tensor into contiguous train/val parts (val is the tail).

    Returns independent copies so neither part keeps the full backing storage.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    split = int(len(tokens) * (1.0 - val_fraction))
    return tokens[:split].clone(), tokens[split:].clone()


class LMDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sliding windows of ``context_window`` tokens as ``(input, target)`` pairs.

    Item ``i`` returns ``tokens[i : i + context_window]`` as input and the same
    window shifted right by one as target, so the model learns to predict the
    next token at every position.
    """

    def __init__(self, tokens: Tensor, context_window: int) -> None:
        if tokens.dim() != 1:
            raise ValueError(f"tokens must be 1D, got shape {tuple(tokens.shape)}")
        if context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        if len(tokens) < context_window + 1:
            raise ValueError(
                f"need at least context_window + 1 tokens, got {len(tokens)}"
            )
        self.tokens = tokens
        self.context_window = context_window

    def __len__(self) -> int:
        return len(self.tokens) - self.context_window

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(
                f"index {index} is out of range for LMDataset of size {len(self)}"
            )
        # Tokens may be stored in a smaller dtype (e.g. int16); the model needs
        # int64 ids, so cast the (small) window back to long here.
        x = self.tokens[index : index + self.context_window].long()
        y = self.tokens[index + 1 : index + self.context_window + 1].long()
        return x, y


class ChunkedRandomSampler(Sampler[int]):
    """Yield random indices in bounded chunks.

    PyTorch's default ``RandomSampler`` builds a **full permutation** of the
    dataset (``torch.randperm(n).tolist()``), which is O(N) in both memory and
    time — prohibitive for our multi-billion-window datasets. This sampler draws
    random indices in fixed-size chunks instead, so memory stays ``O(chunk_size)``.

    Sampling is *with replacement*, which is fine for language-model pre-training.
    """

    def __init__(
        self,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
        chunk_size: int = 1 << 20,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        produced = 0
        while produced < self.num_samples:
            size = min(self.chunk_size, self.num_samples - produced)
            chunk = torch.randint(
                0, self.num_samples, (size,), generator=self.generator
            )
            yield from chunk.tolist()
            produced += size

    def __len__(self) -> int:
        return self.num_samples


def build_dataloader(
    tokens: Tensor,
    context_window: int,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Wrap ``tokens`` in an :class:`LMDataset` and a :class:`DataLoader`.

    Set ``shuffle=True`` for the training split and leave it ``False`` for
    validation. Shuffling uses :class:`ChunkedRandomSampler` (bounded memory)
    rather than PyTorch's default full-permutation sampler.
    """
    dataset = LMDataset(tokens, context_window=context_window)
    if shuffle:
        sampler = ChunkedRandomSampler(len(dataset))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
