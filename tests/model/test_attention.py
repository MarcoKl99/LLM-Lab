"""Unit tests for causal self-attention."""

from __future__ import annotations

import pytest
import torch

from llm_lab.model.attention import CausalSelfAttention


def test_output_shape_matches_input() -> None:
    attn = CausalSelfAttention(n_embd=16, n_head=4)
    assert attn(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(n_embd=10, n_head=3)


def test_attention_is_causal() -> None:
    attn = CausalSelfAttention(n_embd=16, n_head=4)
    attn.eval()

    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[:, 3:, :] = torch.randn(1, 3, 16)

    with torch.no_grad():
        original = attn(x)
        modified = attn(changed)

    assert torch.allclose(original[:, :3, :], modified[:, :3, :])
