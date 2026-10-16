"""Index samplers for the training ``DataLoader``.

Two bounded-memory alternatives to PyTorch's default ``RandomSampler``, which
builds a full ``torch.randperm(n).tolist()`` — O(N) in memory and time, and
prohibitive for our multi-billion-window datasets:

- :class:`ChunkedRandomSampler` — independent uniform draws **with replacement**,
  in bounded chunks. Global mixing on *every* step.
- :class:`ShuffledChunkSampler` — a full **permutation**: shuffle the chunk order,
  then permute the indices within each chunk. Full coverage, but it only mixes
  globally once you traverse many chunks.

Why ``ChunkedRandomSampler`` is the default
-------------------------------------------
Our runs see far less than one epoch of windows (e.g. 60,074 steps × 32 = ~1.9M
of ~2B windows), so *coverage* is irrelevant — but *global mixing on every step*
is essential.

``ShuffledChunkSampler`` shuffles the chunk order and permutes within a chunk, so
with the default ``chunk_size = 1 << 20`` the first ~32,768 steps all come from a
**single** 1M-token region. The model then only ever sees ~1M unique token
positions, memorises them (train loss collapses to ~0.4) while validation loss
*rises*. The block/shard shuffle is only sound when a run traverses a whole
epoch; otherwise use :class:`ChunkedRandomSampler`.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ChunkedRandomSampler(Sampler[int]):
    """Yield random indices in bounded chunks (**with replacement**).

    Each index is drawn independently and uniformly from ``[0, num_samples)``, so
    every step sees a globally random sample of the corpus while memory stays
    ``O(chunk_size)``. Duplicates are possible (and harmless): the sampler never
    materialises a full permutation.
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

    Shuffles *chunks* of indices against each other and permutes the indices
    *within* each chunk, so every index appears exactly **once** per epoch while
    memory stays ``O(chunk_size + num_chunks)``.

    Caveat: this is a *shard* shuffle. It only mixes globally across a full pass;
    for a run that stops after a small fraction of an epoch it degenerates into
    reading one contiguous region. See the module docstring.
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
