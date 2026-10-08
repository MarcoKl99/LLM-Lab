"""Unit tests for the transformer block."""

from __future__ import annotations

import torch

from llm_lab.model.block import MLP, Block


def test_block_preserves_shape() -> None:
    assert Block(n_embd=16, n_head=4)(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_mlp_preserves_shape() -> None:
    assert MLP(n_embd=16)(torch.randn(2, 5, 16)).shape == (2, 5, 16)
