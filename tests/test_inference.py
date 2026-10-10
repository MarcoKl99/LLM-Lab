"""Unit tests for loading a model from a checkpoint for inference."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_lab.data.tokenizer import BPETokenizer
from llm_lab.inference import load_for_inference
from llm_lab.model.lstm import LSTMLanguageModel


def _config(tmp_path: Path, tokenizer: BPETokenizer) -> dict:
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(tokenizer_path)
    return {
        "experiment": "exp",
        "run_name": "run",
        "device": "cpu",
        "data": {"context_window": 8, "tokenizer": {"output": str(tokenizer_path)}},
        "model": {"type": "lstm", "hidden_size": 16, "num_layers": 1},
        "training": {"checkpoint_dir": str(tmp_path / "runs")},
    }


def _save_checkpoint(path: Path, tokenizer: BPETokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = LSTMLanguageModel(
        vocab_size=tokenizer.vocab_size, hidden_size=16, num_layers=1
    )
    torch.save({"model": model.state_dict()}, path)


def test_load_for_inference_loads_best_checkpoint(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train(["hello world " * 20], vocab_size=300)
    config = _config(tmp_path, tokenizer)
    _save_checkpoint(tmp_path / "runs" / "exp" / "run" / "best.pt", tokenizer)

    model, loaded_tokenizer, device = load_for_inference(config, device="cpu")

    assert device == torch.device("cpu")
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert model.vocab_size == tokenizer.vocab_size
    assert model.training is False  # eval mode
    logits = model(torch.randint(0, tokenizer.vocab_size, (2, 4)))
    assert logits.shape == (2, 4, tokenizer.vocab_size)


def test_load_for_inference_accepts_explicit_checkpoint_path(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train(["hello world " * 20], vocab_size=300)
    config = _config(tmp_path, tokenizer)
    custom = tmp_path / "somewhere" / "custom.pt"
    _save_checkpoint(custom, tokenizer)

    model, _, _ = load_for_inference(config, checkpoint=str(custom), device="cpu")

    assert model.vocab_size == tokenizer.vocab_size


def test_load_for_inference_missing_checkpoint_raises(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train(["hello world " * 20], vocab_size=300)
    config = _config(tmp_path, tokenizer)

    with pytest.raises(FileNotFoundError):
        load_for_inference(config, device="cpu")
