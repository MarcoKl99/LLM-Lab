"""Index samplers for the training ``DataLoader``.

Two bounded-memory alternatives to PyTorch's default ``RandomSampler``, which
builds a full ``torch.randperm(n).tolist()`` — O(N) in memory and time, and
prohibitive for our multi-billion-window datasets:

- :class:`ChunkedRandomSampler` — independent uniform draws **with replacement**,
  in bounded chunks. Global mixing on *every* step. **This is the default.**
- :class:`ShuffledChunkSampler` — a full **permutation**: shuffle the chunk order,
  then permute the indices within each chunk. Full coverage, but it only mixes
  globally once a run traverses many chunks. **Do not use for our runs** (it
  emits a warning on construction) — see below.

Why ``ChunkedRandomSampler`` is the default
-------------------------------------------
Our runs see far less than one epoch of windows (e.g. 60,074 steps × 32 = ~1.9M
of ~2B windows, i.e. ``m/n ≈ 0.1%``), so *coverage* is irrelevant — but *global
mixing on every step* is essential.

``ShuffledChunkSampler`` shuffles the chunk order and permutes within a chunk, so
with the default ``chunk_size = 1 << 20`` the first ~32,768 steps all come from a
**single** 1M-token region. The model then only ever sees ~1M unique token
positions, memorises them (train loss collapses to ~0.4) while validation loss
*rises*. That is exactly the failure observed on the ``context/xl3_ctx1024`` run.

The rule of thumb: a shard shuffle is only sound when

    shard_size  ≪  run_length

i.e. when the run consumes **many** chunks (many epochs over a small corpus, or a
``chunk_size`` far smaller than the run). Ours had ``shard_size ≈ run_length``,
so the "shuffle" never left the first shard.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ChunkedRandomSampler(Sampler[int]):
    """Yield random indices in bounded chunks (**with replacement**).

    Each index is drawn independently and uniformly from ``[0, num_samples)``, so
    every step sees a globally random sample of the corpus while memory stays
    ``O(chunk_size)``. Duplicates are possible (and harmless): the sampler never
    materialises a full permutation. When ``num_samples`` is much larger than the
    number of draws (our case) the draws are almost all distinct anyway.
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


class ShuffledChunkSampler(Sampler[int]):
    """A bounded-memory **permutation** of ``[0, num_samples)``.

    .. warning::
        **Do not use this for our training runs.** This is a *shard* shuffle and
        is only sound when ``chunk_size ≪ run_length`` (i.e. a run consumes many
        chunks). With the default ``chunk_size = 1 << 20`` the first ~32,768 steps
        all come from a single 1M-token region: the model sees only ~1M unique
        token positions, memorises them, and validation loss rises while training
        loss collapses. That is exactly what happened on ``context/xl3_ctx1024``.

        Use :class:`ChunkedRandomSampler` instead unless you know the run
        traverses many chunks.

    Constructing it emits a :class:`UserWarning` so the trap cannot be hit by
    accident. It is kept because it is correct in the right regime (full coverage,
    one pass per epoch) and useful for comparison.
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
        warnings.warn(
            "ShuffledChunkSampler is a shard shuffle and is NOT suitable for "
            "short runs: with a large chunk_size the first tens of thousands of "
            "steps all come from one contiguous region, so the model only sees a "
            "tiny slice of the corpus and memorises it (train loss collapses "
            "while val loss rises). Use ChunkedRandomSampler unless the run "
            "traverses many chunks. See the module docstring.",
            UserWarning,
            stacklevel=2,
        )
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        num_chunks = (self.num_samples + self.chunk_size - 1) // self.chunk_size
        chunk_order = torch.randperm(num_chunks, generator=self.generator).tolist()
        for chunk in chunk_order:
            start = chunk * self.chunk_size
            size = min(self.chunk_size, self.num_samples - start)
            within = torch.randperm(size, generator=self.generator).tolist()
            for offset in within:
                yield start + offset

    def __len__(self) -> int:
        return self.num_samples
