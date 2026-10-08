"""Unit tests for reading TensorBoard scalars."""

from __future__ import annotations

from pathlib import Path

from llm_lab.utils.logging import Logger
from llm_lab.utils.tensorboard import read_scalars


def test_read_scalars(tmp_path: Path) -> None:
    with Logger(tmp_path / "tb") as logger:
        logger.log(1, {"train/loss": 1.0, "val/loss": 2.0})
        logger.log(2, {"train/loss": 0.5})

    scalars = read_scalars(tmp_path / "tb")

    assert scalars["train/loss"] == [(1, 1.0), (2, 0.5)]
    assert scalars["val/loss"] == [(1, 2.0)]
