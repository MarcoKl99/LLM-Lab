"""Decoder-only transformer language model (GPT).

Learned absolute position embeddings, pre-norm blocks, and (by default) tied
input/output embedding weights.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from llm_lab.model.base import LanguageModel
from llm_lab.model.block import Block


class GPT(LanguageModel):
    def __init__(
        self,
        vocab_size: int,
        context_window: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(context_window, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(n_embd, n_head, dropout) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, length = input_ids.shape
        if length > self.context_window:
            raise ValueError(
                f"sequence length {length} exceeds context_window {self.context_window}"
            )

        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.ln_f(x))
