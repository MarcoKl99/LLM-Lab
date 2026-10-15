"""Unit tests for causal self-attention."""

from __future__ import annotations

import math

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


def test_matches_explicit_masked_softmax_reference() -> None:
    """The fused SDPA path agrees with an explicit QKᵀ / mask / softmax / V."""
    torch.manual_seed(0)
    n_embd, n_head, length = 16, 4, 7
    attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head)
    attn.eval()  # dropout off, so the comparison is deterministic

    x = torch.randn(2, length, n_embd)

    with torch.no_grad():
        out = attn(x)

        q, k, v = attn.qkv(x).split(n_embd, dim=2)
        shape = (x.shape[0], length, n_head, attn.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        expected = (weights @ v).transpose(1, 2).contiguous().view(2, length, n_embd)
        expected = attn.proj(expected)

    assert torch.allclose(out, expected, atol=1e-5)
