"""Unit tests for the GPT model and its factory entry."""

from __future__ import annotations

import pytest
import torch

from llm_lab.model.build import build_model
from llm_lab.model.gpt import GPT


def _model(**overrides: object) -> GPT:
    config: dict[str, object] = dict(
        vocab_size=32, context_window=8, n_layer=2, n_head=2, n_embd=16
    )
    config.update(overrides)
    return GPT(**config)  # type: ignore[arg-type]


def test_forward_output_shape() -> None:
    assert _model()(torch.randint(0, 32, (2, 6))).shape == (2, 6, 32)


def test_accepts_shorter_than_context_window() -> None:
    assert _model()(torch.randint(0, 32, (1, 3))).shape == (1, 3, 32)


def test_rejects_sequence_longer_than_context_window() -> None:
    with pytest.raises(ValueError):
        _model(context_window=4)(torch.randint(0, 32, (1, 5)))


def test_weights_are_tied() -> None:
    model = _model(tie_weights=True)
    assert model.lm_head.weight is model.token_embedding.weight


def test_weights_untied_when_disabled() -> None:
    model = _model(tie_weights=False)
    assert model.lm_head.weight is not model.token_embedding.weight


def test_gradients_flow() -> None:
    model = _model()
    model(torch.randint(0, 32, (2, 6))).sum().backward()
    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attn.qkv.weight.grad is not None


def test_gpt_is_causal() -> None:
    model = _model()
    model.eval()

    ids = torch.randint(0, 32, (1, 6))
    changed = ids.clone()
    changed[:, 3:] = torch.randint(0, 32, (1, 3))

    with torch.no_grad():
        original = model(ids)
        modified = model(changed)

    assert torch.allclose(original[:, :3, :], modified[:, :3, :])


def test_build_model_gpt() -> None:
    model = build_model(
        {"type": "gpt", "n_layer": 2, "n_head": 2, "n_embd": 16},
        vocab_size=32,
        context_window=8,
    )
    assert isinstance(model, GPT)


def test_build_model_gpt_requires_context_window() -> None:
    with pytest.raises(ValueError):
        build_model({"type": "gpt"}, vocab_size=32)
