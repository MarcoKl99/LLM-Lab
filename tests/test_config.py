"""Unit tests for the config loader."""

from __future__ import annotations

from pathlib import Path

from llm_lab.config import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb:\n  c: two\n", encoding="utf-8")

    assert load_config(path) == {"a": 1, "b": {"c": "two"}}
