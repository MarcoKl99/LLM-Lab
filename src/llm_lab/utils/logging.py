"""Experiment logging: console-friendly metrics plus TensorBoard."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Writes metrics, text and hyperparameters to TensorBoard under ``log_dir``."""

    def __init__(self, log_dir: str | Path) -> None:
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        """Log scalar metrics (e.g. loss, LR) at ``step``."""
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Log free-form text (e.g. the full config) under a text panel."""
        self._writer.add_text(tag, text, step)

    def log_hparams(
        self,
        hparams: Mapping[str, Any],
        metrics: Mapping[str, float],
    ) -> None:
        """Log a flat dict of hyperparameters plus final metrics to the HPARAMS tab."""
        self._writer.add_hparams(dict(hparams), dict(metrics))

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
