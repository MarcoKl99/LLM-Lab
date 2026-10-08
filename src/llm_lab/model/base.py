"""The shared language-model interface.

Every architecture in this project maps a batch of token ids to next-token
logits. Keeping that contract fixed is what lets the data pipeline, trainer,
evaluation and generation be reused across architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class LanguageModel(nn.Module, ABC):
    """Maps ``input_ids`` of shape ``(B, T)`` to ``logits`` of shape ``(B, T, V)``.

    ``logits[b, t]`` holds the scores for the token that follows position ``t``.
    Subclasses own their embeddings and the final projection to the vocabulary.
    """

    vocab_size: int

    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor:
        """Return next-token logits of shape ``(B, T, vocab_size)``."""
        raise NotImplementedError
