"""Unit tests for the shared trainer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_lab.model.lstm import LSTMLanguageModel
from llm_lab.training.optimizer import build_optimizer, build_scheduler, lr_factor
from llm_lab.training.trainer import Trainer
from llm_lab.utils.logging import Logger
from llm_lab.utils.tensorboard import read_scalars


def _loader(vocab: int, n: int, context: int, batch_size: int) -> DataLoader:
    x = torch.randint(0, vocab, (n, context))
    y = torch.randint(0, vocab, (n, context))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _build_trainer(
    tmp_path: Path,
    model: LSTMLanguageModel | None = None,
    sample_fn: Callable[[int], None] | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    tokens_per_step: int | None = None,
    unique_train_tokens: int | None = None,
    learning_rate: float = 1e-2,
) -> tuple[Trainer, Path]:
    vocab, context = 20, 8
    model = model or LSTMLanguageModel(vocab_size=vocab, hidden_size=16, num_layers=1)
    optimizer = build_optimizer(model, learning_rate=learning_rate)
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
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        tokens_per_step=tokens_per_step,
        unique_train_tokens=unique_train_tokens,
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


def test_resume_picks_up_scheduler_learning_rate(tmp_path: Path) -> None:
    # Run the schedule to step 5 (base LR 1e-2, warmup 2, cosine over 10 steps).
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, log_interval=5, progress=False)
    trainer.logger.close()

    # Resume with a *different* config LR: the checkpoint's scheduler state wins.
    resumed, _ = _build_trainer(tmp_path, learning_rate=1e-3)
    resumed.load_checkpoint(run_dir / "latest.pt")
    resumed.logger.close()

    assert resumed.scheduler.base_lrs == [1e-2]
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
        1e-2 * lr_factor(5, warmup_steps=2, max_steps=10, min_lr_ratio=0.1)
    )

    # The next step continues the checkpointed schedule, not a fresh warmup.
    resumed.scheduler.step()
    assert resumed.scheduler.get_last_lr()[0] == pytest.approx(
        1e-2 * lr_factor(6, warmup_steps=2, max_steps=10, min_lr_ratio=0.1)
    )


def test_early_stopping_triggers_after_patience(tmp_path: Path) -> None:
    trainer, _ = _build_trainer(
        tmp_path, early_stop_patience=2, early_stop_min_delta=1e9
    )
    history = trainer.train(max_steps=30, eval_interval=5, progress=False)
    trainer.logger.close()

    # first eval sets the baseline, then two non-improving evals stop it at step 15
    assert len(history["val_loss"]) == 3
    assert len(history["train_loss"]) == 15


def test_no_early_stopping_when_patience_is_none(tmp_path: Path) -> None:
    trainer, _ = _build_trainer(tmp_path, early_stop_min_delta=1e9)
    history = trainer.train(max_steps=15, eval_interval=5, progress=False)
    trainer.logger.close()

    assert len(history["train_loss"]) == 15
    assert len(history["val_loss"]) == 3


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


def test_trainer_logs_throughput_and_compute_metrics(tmp_path: Path) -> None:
    # batch_size 8 * context 8 = 64 tokens/step
    trainer, _ = _build_trainer(
        tmp_path, tokens_per_step=64, unique_train_tokens=100
    )
    trainer.train(max_steps=10, eval_interval=5, log_interval=5, progress=False)
    trainer.logger.close()

    scalars = read_scalars(tmp_path / "tb")
    assert scalars["throughput/tokens_seen"][-1] == (10, 640.0)
    step, epochs = scalars["throughput/epochs"][-1]
    assert step == 10
    assert epochs == pytest.approx(6.4, abs=1e-3)
    assert "throughput/tokens_per_sec" in scalars
    assert "compute/flops" in scalars


def test_trainer_skips_throughput_metrics_when_unspecified(tmp_path: Path) -> None:
    trainer, _ = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, log_interval=5, progress=False)
    trainer.logger.close()

    scalars = read_scalars(tmp_path / "tb")
    assert "throughput/tokens_seen" not in scalars
    assert "compute/flops" not in scalars


def test_train_resumes_from_checkpoint(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, progress=False)
    trainer.logger.close()

    resumed, _ = _build_trainer(tmp_path)
    history = resumed.train(max_steps=8, eval_interval=8, progress=False, resume=True)
    resumed.logger.close()

    assert len(history["train_loss"]) == 3
    assert (run_dir / "latest.pt").exists()


def test_checkpoints_leave_no_tmp_files(tmp_path: Path) -> None:
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, progress=False)
    trainer.logger.close()

    assert (run_dir / "latest.pt").exists()
    assert not list(run_dir.glob("*.tmp"))


def test_atomic_save_preserves_existing_checkpoint_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, run_dir = _build_trainer(tmp_path)
    trainer.train(max_steps=5, eval_interval=5, progress=False)
    before = (run_dir / "latest.pt").read_bytes()

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr("llm_lab.training.trainer.torch.save", boom)
    with pytest.raises(RuntimeError):
        trainer._save_checkpoint(10, is_best=False)

    # the failure hit the temp file, so the previous checkpoint is untouched
    assert (run_dir / "latest.pt").read_bytes() == before
    trainer.logger.close()
