"""Unit tests for the optimizer and learning-rate schedule."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from llm_lab.training.optimizer import build_optimizer, build_scheduler, lr_factor

SCHEDULE = {"warmup_steps": 10, "max_steps": 100, "min_lr_ratio": 0.1}


def test_lr_factor_warms_up_linearly() -> None:
    assert lr_factor(0, **SCHEDULE) < lr_factor(5, **SCHEDULE) < lr_factor(9, **SCHEDULE)


def test_lr_factor_peaks_after_warmup() -> None:
    assert lr_factor(9, **SCHEDULE) == pytest.approx(1.0)
    assert lr_factor(10, **SCHEDULE) == pytest.approx(1.0)


def test_lr_factor_decays_to_floor() -> None:
    assert lr_factor(100, **SCHEDULE) == pytest.approx(0.1)


def test_lr_factor_is_non_increasing_after_warmup() -> None:
    values = [lr_factor(step, **SCHEDULE) for step in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_scheduler_scales_optimizer_lr() -> None:
    model = nn.Linear(2, 2)
    optimizer = build_optimizer(model, learning_rate=1.0)
    scheduler = build_scheduler(optimizer, **SCHEDULE)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_factor(0, **SCHEDULE))

    model(torch.randn(4, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr_factor(1, **SCHEDULE))
