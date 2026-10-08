"""Unit tests for the LSTM language model and the model factory."""

from __future__ import annotations

import pytest
import torch

from llm_lab.model.build import build_model
from llm_lab.model.lstm import LSTMLanguageModel


def test_forward_output_shape() -> None:
    model = LSTMLanguageModel(vocab_size=50, hidden_size=16, num_layers=1)
    logits = model(torch.randint(0, 50, (2, 5)))
    assert logits.shape == (2, 5, 50)


def test_vocab_size_attribute() -> None:
    assert LSTMLanguageModel(vocab_size=50).vocab_size == 50


def test_is_stateless_across_calls() -> None:
    model = LSTMLanguageModel(vocab_size=50, hidden_size=16, num_layers=1)
    model.eval()
    input_ids = torch.randint(0, 50, (2, 5))
    with torch.no_grad():
        first = model(input_ids)
        second = model(input_ids)
    assert torch.equal(first, second)


def test_gradients_flow_to_all_parameters() -> None:
    model = LSTMLanguageModel(vocab_size=50, hidden_size=16, num_layers=1)
    logits = model(torch.randint(0, 50, (2, 5)))
    logits.sum().backward()
    assert model.embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_build_model_lstm() -> None:
    model = build_model({"type": "lstm", "hidden_size": 8}, vocab_size=20)
    assert isinstance(model, LSTMLanguageModel)


def test_build_model_unknown_type() -> None:
    with pytest.raises(ValueError):
        build_model({"type": "nope"}, vocab_size=20)
