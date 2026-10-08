"""A stateless LSTM language model baseline.

The hidden state is reset on every forward pass, so each window is processed
independently — matching the transformer's fixed context and making the two
architectures directly comparable.
"""

from __future__ import annotations

from torch import Tensor, nn

from llm_lab.model.base import LanguageModel


class LSTMLanguageModel(LanguageModel):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        return self.lm_head(output)
