"""Causal multi-head self-attention.

Uses ``F.scaled_dot_product_attention``, which on CUDA dispatches to
FlashAttention / memory-efficient kernels: the L x L score matrix is never
materialised, so attention memory is O(L) instead of O(L^2) and it is faster.
``is_causal=True`` applies the causal mask internally.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, n_embd = x.shape

        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(batch, length, n_embd)
        return self.resid_dropout(self.proj(out))
