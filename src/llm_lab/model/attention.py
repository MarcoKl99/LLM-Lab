"""Causal multi-head self-attention (manual implementation).

Explicit QKᵀ/√d, a causal mask, softmax and a weighted sum over values. Kept
transparent on purpose; ``F.scaled_dot_product_attention`` is a drop-in
efficiency upgrade we can adopt later.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, n_embd = x.shape

        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, length, n_embd)
        return self.resid_dropout(self.proj(out))
