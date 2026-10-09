"""Turn a flat token stream into next-token prediction windows.

A language model learns by predicting the next token, so training examples are
``(input, target)`` pairs where ``target`` is ``input`` shifted right by one.
This module produces those pairs from a 1D tensor of token ids.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from llm_lab.data.tokenizer import BPETokenizer


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
        pieces.append(torch.tensor(flat, dtype=torch.long))
        batch.clear()

    for text in documents:
        batch.append(text)
        if len(batch) == batch_size:
            flush()
    flush()

    if not pieces:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(pieces)


def encode_corpus(tokenizer: BPETokenizer, text: str) -> Tensor:
    """Encode a single text into a 1D ``LongTensor`` of token ids."""
    return encode_documents(tokenizer, [text])


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
        x = self.tokens[index : index + self.context_window]
        y = self.tokens[index + 1 : index + self.context_window + 1]
        return x, y


def build_dataloader(
    tokens: Tensor,
    context_window: int,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Wrap ``tokens`` in an :class:`LMDataset` and a :class:`DataLoader`.

    Set ``shuffle=True`` for the training split and leave it ``False`` for
    validation.
    """
    dataset = LMDataset(tokens, context_window=context_window)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
