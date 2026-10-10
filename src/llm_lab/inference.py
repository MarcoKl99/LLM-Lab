"""Load a trained model + tokenizer for inference.

Mirrors how :mod:`scripts.train` builds a model, so a checkpoint produced by
training can be reconstructed from the same config and used for prompting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from llm_lab.config import run_dir_for
from llm_lab.data.tokenizer import BPETokenizer
from llm_lab.model.base import LanguageModel
from llm_lab.model.build import build_model
from llm_lab.utils.device import resolve_device


def load_for_inference(
    config: dict[str, Any],
    *,
    checkpoint: str = "best",
    run_dir: str | Path | None = None,
    device: str = "auto",
) -> tuple[LanguageModel, BPETokenizer, torch.device]:
    """Build the model from ``config``, load ``checkpoint``, and return it ready to prompt.

    ``checkpoint`` may be ``"best"`` or ``"latest"`` (resolved inside the run
    directory) or an explicit path to a ``.pt`` file. ``run_dir`` overrides the
    config-derived run directory.
    """
    resolved_device = resolve_device(device)

    tokenizer = BPETokenizer.load(config["data"]["tokenizer"]["output"])
    model = build_model(
        config["model"],
        vocab_size=tokenizer.vocab_size,
        context_window=config["data"]["context_window"],
    )

    if checkpoint in ("best", "latest"):
        directory = Path(run_dir) if run_dir is not None else run_dir_for(config)
        checkpoint_path = directory / f"{checkpoint}.pt"
    else:
        checkpoint_path = Path(checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    state = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=False
    )
    model.load_state_dict(state["model"])
    model.to(resolved_device)
    model.eval()

    return model, tokenizer, resolved_device
