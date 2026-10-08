"""General-purpose utilities."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a device string; ``"auto"`` picks cuda > mps > cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
