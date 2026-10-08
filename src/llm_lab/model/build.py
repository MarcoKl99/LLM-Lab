"""Construct a language model from a config section."""

from __future__ import annotations

from typing import Any

from llm_lab.model.base import LanguageModel
from llm_lab.model.lstm import LSTMLanguageModel


def build_model(config: dict[str, Any], vocab_size: int) -> LanguageModel:
    """Build a model from its ``config`` section (with a ``type`` key)."""
    model_type = config["type"]

    if model_type == "lstm":
        return LSTMLanguageModel(
            vocab_size=vocab_size,
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.0),
        )

    raise ValueError(f"Unknown model type: {model_type!r}")
