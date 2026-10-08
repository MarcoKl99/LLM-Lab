"""Unit tests for the shared trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_lab.model.lstm import LSTMLanguageModel
from llm_lab.training.optimizer import build_optimizer, build_scheduler
from llm_lab.training.trainer import Trainer
from llm_lab.utils.logging import Logger


def _loader(vocab: int, n: int, context: int, batch_size: int) -> DataLoader:
    x = torch.randint(0, vocab, (n, context))
    y = torch.randint(0, vocab, (n, context))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _build_trainer(tmp_path: Path) -> tuple[Trainer, Path]:
    vocab, context = 20, 8
    model = LSTMLanguageModel(vocab_size=vocab, hidden_size=16, num_layers=1)
    optimizer = build_optimizer(model, learning_rate=1e-2)
    scheduler = build_scheduler(
        optimizer, warmup_steps=2, max_steps=10, min_lr_ratio=0.1
    )
    run_dir = tmp_path / "run"

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        _loader(vocab, 64, context, 8),
        _loader(vocab, 16, context, 8),
        Logger(tmp_path / "tb"),
        device=torch.device("cpu"),
        run_dir=run_dir,
        grad_clip=1.0,
        eval_batches=2,
    )
    return trainer, run_dir


def test_trainer_runs_and_records_history(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)

    history = trainer.train(max_steps=10, eval_interval=5, log_interval=5)

    assert len(history["train_loss"]) == 10
    assert len(history["val_loss"]) == 2
    assert len(history["val_perplexity"]) == 2
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "best.pt").exists()


def test_checkpoint_contains_model_state(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, log_interval=5)

    checkpoint = torch.load(run_dir / "latest.pt")
    assert checkpoint["step"] == 5
    assert "model" in checkpoint
