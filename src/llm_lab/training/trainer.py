"""The training loop, shared by every architecture.

The trainer is architecture-agnostic: it calls ``model(x)`` for logits and owns
the next-token cross-entropy objective, so an LSTM and a transformer train
through exactly the same code.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_lab.model.base import LanguageModel
from llm_lab.utils.logging import Logger


def _infinite_batches(
    loader: DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches forever, restarting the loader each epoch (re-shuffling)."""
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(
        self,
        model: LanguageModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: Logger,
        *,
        device: torch.device,
        run_dir: Path,
        grad_clip: float | None = None,
        eval_batches: int | None = None,
        sample_fn: Callable[[int], None] | None = None,
        early_stop_patience: int | None = None,
        early_stop_min_delta: float = 0.0,
        tokens_per_step: int | None = None,
        unique_train_tokens: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        self.run_dir = Path(run_dir)
        self.grad_clip = grad_clip
        self.eval_batches = eval_batches
        self.sample_fn = sample_fn
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.vocab_size = model.vocab_size
        self.best_val_loss = float("inf")
        self.tokens_per_step = tokens_per_step
        self.unique_train_tokens = unique_train_tokens
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))

    def _progress_metrics(
        self,
        step: int,
        start_step: int,
        loss: float,
        lr: float,
        start_time: float,
    ) -> dict[str, float]:
        """Scalars logged each ``log_interval``: loss/LR plus scaling accounting.

        ``tokens_seen`` and ``flops`` are cumulative over the whole run (so they
        are correct across resumes); throughput is measured for this session.
        FLOPs use the ``6 * N * D`` estimate with ``N`` = total parameters.
        """
        metrics: dict[str, float] = {"train/loss": loss, "train/lr": lr}
        if self.tokens_per_step is None:
            return metrics

        tokens_seen = step * self.tokens_per_step
        session_tokens = (step - start_step + 1) * self.tokens_per_step
        elapsed = time.perf_counter() - start_time
        metrics["throughput/tokens_seen"] = float(tokens_seen)
        metrics["throughput/tokens_per_sec"] = session_tokens / elapsed
        metrics["throughput/elapsed_seconds"] = elapsed
        metrics["compute/flops"] = float(6 * self.n_params * tokens_seen)
        if self.unique_train_tokens:
            metrics["throughput/epochs"] = tokens_seen / self.unique_train_tokens
        return metrics

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Return mean validation loss and perplexity."""
        self.model.eval()
        total, count = 0.0, 0
        for i, (x, y) in enumerate(self.val_loader):
            if self.eval_batches is not None and i >= self.eval_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            total += self._loss(x, y).item()
            count += 1
        self.model.train()

        mean_loss = total / max(1, count)
        return mean_loss, math.exp(mean_loss)

    def train(
        self,
        *,
        max_steps: int,
        eval_interval: int,
        log_interval: int = 10,
        progress: bool = True,
        resume: bool = False,
    ) -> dict[str, list[float]]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
        }

        start_step = 1
        if resume:
            checkpoint_path = self.run_dir / "latest.pt"
            if checkpoint_path.exists():
                start_step = self.load_checkpoint(checkpoint_path) + 1
                print(f"resumed from {checkpoint_path} at step {start_step - 1}")

        if start_step > max_steps:
            print(f"checkpoint already at step {start_step - 1} (max_steps={max_steps})")
            return history

        self.model.train()
        batches = _infinite_batches(self.train_loader)
        evals_without_improvement = 0
        start_time = time.perf_counter()
        bar = tqdm(
            range(start_step, max_steps + 1),
            initial=start_step - 1,
            total=max_steps,
            desc=self.run_dir.name,
            unit="step",
            disable=not progress,
        )
        for step in bar:
            x, y = next(batches)
            x, y = x.to(self.device), y.to(self.device)

            loss = self._loss(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            history["train_loss"].append(loss.item())

            lr = self.scheduler.get_last_lr()[0]
            bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            if step % log_interval == 0:
                self.logger.log(
                    step,
                    self._progress_metrics(
                        step, start_step, loss.item(), lr, start_time
                    ),
                )

            if step % eval_interval == 0 or step == max_steps:
                val_loss, val_perplexity = self.evaluate()
                history["val_loss"].append(val_loss)
                history["val_perplexity"].append(val_perplexity)
                self.logger.log(
                    step, {"val/loss": val_loss, "val/perplexity": val_perplexity}
                )
                tqdm.write(
                    f"step {step:>6} | val loss {val_loss:.4f} | ppl {val_perplexity:.2f}"
                )

                improved = val_loss < self.best_val_loss - self.early_stop_min_delta
                if improved:
                    self.best_val_loss = val_loss
                self._save_checkpoint(step, is_best=improved)

                if self.sample_fn is not None:
                    self.sample_fn(step)

                if improved:
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    if (
                        self.early_stop_patience is not None
                        and evals_without_improvement >= self.early_stop_patience
                    ):
                        tqdm.write(
                            f"early stopping at step {step} "
                            f"(no improvement for {evals_without_improvement} evals)"
                        )
                        break

        return history

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore model/optimizer/scheduler state; return the saved step."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_val_loss = checkpoint["best_val_loss"]
        return int(checkpoint["step"])

    def _save_checkpoint(self, step: int, *, is_best: bool) -> None:
        state = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, self.run_dir / "latest.pt")
        if is_best:
            torch.save(state, self.run_dir / "best.pt")
