"""Byte-level BPE tokenizer.

A thin wrapper around HuggingFace ``tokenizers`` that exposes the small
interface the rest of the project needs. Byte-level BPE is lossless (no
unknown tokens) and matches the GPT-2 style we will use for the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS: list[str] = ["<|endoftext|>"]


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 1024,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens or SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [encoding.ids for encoding in self._tokenizer.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def token_id(self, token: str) -> int | None:
        """Return the id of ``token`` (e.g. a special token), or ``None``."""
        return self._tokenizer.token_to_id(token)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))
