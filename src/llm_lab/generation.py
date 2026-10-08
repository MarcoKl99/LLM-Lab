"""Autoregressive text generation, shared by every architecture.

Generation only needs the model contract (``forward(input_ids) -> logits``) and a
tokenizer, so it works identically for the LSTM baseline and the transformer.
"""

from __future__ import annotations

import torch

from llm_lab.data.tokenizer import BPETokenizer
from llm_lab.model.base import LanguageModel


@torch.no_grad()
def generate(
    model: LanguageModel,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    context_window: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> str:
    """Sample ``max_new_tokens`` continuations of ``prompt`` and return the text.

    The models are stateless, so each step re-runs ``forward`` over the last
    ``context_window`` tokens. ``temperature <= 0`` selects greedy decoding.
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must tokenize to at least one token")

    for _ in range(max_new_tokens):
        window = ids[-context_window:]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1, :]

        if temperature <= 0:
            next_id = int(torch.argmax(logits))
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[-1]
                logits = torch.where(logits < threshold, float("-inf"), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))

        ids.append(next_id)

    if was_training:
        model.train()

    return tokenizer.decode(ids)
