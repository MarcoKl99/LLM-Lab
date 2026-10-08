"""Construct a language model from a config section."""

from __future__ import annotations

from typing import Any

from llm_lab.model.base import LanguageModel
from llm_lab.model.gpt import GPT
from llm_lab.model.lstm import LSTMLanguageModel


def build_model(
    config: dict[str, Any],
    vocab_size: int,
    context_window: int | None = None,
) -> LanguageModel:
    """Build a model from its ``config`` section (with a ``type`` key)."""
    model_type = config["type"]

    if model_type == "lstm":
        return LSTMLanguageModel(
            vocab_size=vocab_size,
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.0),
        )

    if model_type == "gpt":
        if context_window is None:
            raise ValueError("gpt model requires context_window")
        return GPT(
            vocab_size=vocab_size,
            context_window=context_window,
            n_layer=config.get("n_layer", 4),
            n_head=config.get("n_head", 4),
            n_embd=config.get("n_embd", 128),
            dropout=config.get("dropout", 0.0),
            tie_weights=config.get("tie_weights", True),
        )

    raise ValueError(f"Unknown model type: {model_type!r}")
