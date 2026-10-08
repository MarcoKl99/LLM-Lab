"""Unit tests for the shared trainer."""

from __future__ import annotations

from collections.abc import Callable
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


def _build_trainer(
    tmp_path: Path,
    model: LSTMLanguageModel | None = None,
    sample_fn: Callable[[int], None] | None = None,
) -> tuple[Trainer, Path]:
    vocab, context = 20, 8
    model = model or LSTMLanguageModel(vocab_size=vocab, hidden_size=16, num_layers=1)
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
        sample_fn=sample_fn,
    )
    return trainer, run_dir


def test_trainer_runs_and_records_history(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)

    history = trainer.train(
        max_steps=10, eval_interval=5, log_interval=5, progress=False
    )
    trainer.logger.close()

    assert len(history["train_loss"]) == 10
    assert len(history["val_loss"]) == 2
    assert len(history["val_perplexity"]) == 2
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "best.pt").exists()


def test_checkpoint_contains_model_state(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, log_interval=5, progress=False)
    trainer.logger.close()

    checkpoint = torch.load(run_dir / "latest.pt")
    assert checkpoint["step"] == 5
    assert "model" in checkpoint


def test_load_checkpoint_restores_model_state(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, progress=False)
    trainer.logger.close()
    saved = torch.load(run_dir / "latest.pt")

    fresh = LSTMLanguageModel(vocab_size=20, hidden_size=16, num_layers=1)
    other, _ = _build_trainer(tmp_path, model=fresh)
    step = other.load_checkpoint(run_dir / "latest.pt")
    other.logger.close()

    assert step == 5
    for name, tensor in other.model.state_dict().items():
        assert torch.equal(tensor, saved["model"][name])


def test_trainer_invokes_sample_fn_at_evals(tmp_path: Path) -> None:
    calls: list[int] = []
    trainer, _ = _build_trainer(tmp_path, sample_fn=calls.append)
    trainer.train(max_steps=10, eval_interval=5, progress=False)
    trainer.logger.close()

    assert calls == [5, 10]


def test_train_resume_when_already_finished_is_noop(tmp_path: Path) -> None:
    trainer, _ = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, progress=False)
    trainer.logger.close()

    resumed, _ = _build_trainer(tmp_path)
    history = resumed.train(max_steps=5, eval_interval=5, progress=False, resume=True)
    resumed.logger.close()

    assert history["train_loss"] == []
    assert history["val_loss"] == []


def test_train_resumes_from_checkpoint(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, progress=False)
    trainer.logger.close()

    resumed, _ = _build_trainer(tmp_path)
    history = resumed.train(max_steps=8, eval_interval=8, progress=False, resume=True)
    resumed.logger.close()

    assert len(history["train_loss"]) == 3
    assert (run_dir / "latest.pt").exists()
