"""Experiment logging: console-friendly metrics plus TensorBoard."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Writes scalar metrics to TensorBoard under ``log_dir``."""

    def __init__(self, log_dir: str | Path) -> None:
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)

    def close(self) -> None:
        self._writer.close()
