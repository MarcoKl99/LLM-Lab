"""Unit tests for autoregressive text generation."""

from __future__ import annotations

import pytest
import torch

from llm_lab.data.tokenizer import BPETokenizer
from llm_lab.generation import generate
from llm_lab.model.lstm import LSTMLanguageModel


def _setup() -> tuple[LSTMLanguageModel, BPETokenizer]:
    tokenizer = BPETokenizer.train(["hello world " * 50], vocab_size=300)
    model = LSTMLanguageModel(
        vocab_size=tokenizer.vocab_size, hidden_size=16, num_layers=1
    )
    return model, tokenizer


def test_generate_extends_prompt() -> None:
    model, tokenizer = _setup()
    text = generate(
        model,
        tokenizer,
        "hello",
        max_new_tokens=5,
        context_window=8,
        device=torch.device("cpu"),
    )
    assert text.startswith("hello")
    assert len(tokenizer.encode(text)) >= len(tokenizer.encode("hello")) + 1


def test_generate_greedy_is_deterministic() -> None:
    model, tokenizer = _setup()
    kwargs = {"max_new_tokens": 5, "context_window": 8, "temperature": 0.0}
    first = generate(model, tokenizer, "hello", **kwargs)
    second = generate(model, tokenizer, "hello", **kwargs)
    assert first == second


def test_generate_restores_training_mode() -> None:
    model, tokenizer = _setup()
    model.train()
    generate(model, tokenizer, "hello", max_new_tokens=2, context_window=8)
    assert model.training is True


def test_generate_rejects_empty_prompt() -> None:
    model, tokenizer = _setup()
    with pytest.raises(ValueError):
        generate(model, tokenizer, "", max_new_tokens=2, context_window=8)
