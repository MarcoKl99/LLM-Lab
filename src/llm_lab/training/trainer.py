"""The training loop, shared by every architecture.

The trainer is architecture-agnostic: it calls ``model(x)`` for logits and owns
the next-token cross-entropy objective, so an LSTM and a transformer train
through exactly the same code.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

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
        self.vocab_size = model.vocab_size
        self.best_val_loss = float("inf")

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))

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
    ) -> dict[str, list[float]]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()
        batches = _infinite_batches(self.train_loader)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
        }

        for step in range(1, max_steps + 1):
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

            if step % log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.log(step, {"train/loss": loss.item(), "train/lr": lr})
                print(f"step {step:>6} | loss {loss.item():.4f} | lr {lr:.2e}")

            if step % eval_interval == 0 or step == max_steps:
                val_loss, val_perplexity = self.evaluate()
                history["val_loss"].append(val_loss)
                history["val_perplexity"].append(val_perplexity)
                self.logger.log(
                    step, {"val/loss": val_loss, "val/perplexity": val_perplexity}
                )
                print(f"         | val loss {val_loss:.4f} | ppl {val_perplexity:.2f}")

                is_best = val_loss < self.best_val_loss
                self.best_val_loss = min(self.best_val_loss, val_loss)
                self._save_checkpoint(step, is_best=is_best)

        self.logger.close()
        return history

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
