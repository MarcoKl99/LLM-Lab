"""Load experiment configuration from YAML files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dictionary."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def flatten_config(
    config: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Flatten a nested config into ``a.b.c`` keys (for logging/hparams)."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def run_dir_for(config: Mapping[str, Any]) -> Path:
    """Return the run directory: ``checkpoint_dir / experiment / run_name``."""
    return (
        Path(config["training"]["checkpoint_dir"])
        / config["experiment"]
        / config["run_name"]
    )
