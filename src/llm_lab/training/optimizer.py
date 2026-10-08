"""Optimizer and learning-rate schedule."""

from __future__ import annotations

import math

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> Optimizer:
    """AdamW over all model parameters."""
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def lr_factor(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linear warmup, then cosine decay down to ``min_lr_ratio`` of the base LR."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
) -> LRScheduler:
    """A :class:`LambdaLR` applying :func:`lr_factor` to the optimizer's LR."""
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_factor(
            step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )
