"""Load experiment configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dictionary."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)
