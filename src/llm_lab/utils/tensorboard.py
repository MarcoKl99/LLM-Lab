"""Read scalar curves from a run's TensorBoard logs."""

from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_scalars(log_dir: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Return ``{tag: [(step, value), ...]}`` for every scalar in ``log_dir``."""
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()

    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [(event.step, event.value) for event in accumulator.Scalars(tag)]
    return scalars
