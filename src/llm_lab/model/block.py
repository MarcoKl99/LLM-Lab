"""Pre-norm transformer block: attention + MLP, each wrapped in a residual."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from llm_lab.model.attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
