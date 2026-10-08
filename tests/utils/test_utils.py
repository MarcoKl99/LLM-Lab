"""Unit tests for device resolution and logging."""

from __future__ import annotations

from pathlib import Path

import torch

from llm_lab.utils.device import resolve_device
from llm_lab.utils.logging import Logger


def test_resolve_device_explicit() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_known_type() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_logger_writes_event_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "tb"
    logger = Logger(log_dir)
    logger.log(1, {"train/loss": 1.0, "val/loss": 0.5})
    logger.close()

    assert any(log_dir.glob("events.out.tfevents.*"))
