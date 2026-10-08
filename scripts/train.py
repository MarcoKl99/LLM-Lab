"""Train a language model.

Usage:
    uv run python scripts/train.py --config configs/experiments/lstm_baseline.yaml
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml

from llm_lab.config import flatten_config, load_config
from llm_lab.data.dataset import build_dataloader
from llm_lab.data.tokenizer import BPETokenizer
from llm_lab.generation import generate
from llm_lab.model.base import LanguageModel
from llm_lab.model.build import build_model
from llm_lab.training.optimizer import build_optimizer, build_scheduler
from llm_lab.training.trainer import Trainer
from llm_lab.utils.device import resolve_device
from llm_lab.utils.logging import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_sample_fn(
    config: dict[str, Any],
    model: LanguageModel,
    tokenizer: BPETokenizer,
    logger: Logger,
    device: torch.device,
) -> Callable[[int], None]:
    """Return a callback that logs a generated sample to TensorBoard at each eval."""
    sampling = config["sampling"]

    def sample_fn(step: int) -> None:
        text = generate(
            model,
            tokenizer,
            sampling["prompt"],
            max_new_tokens=sampling["max_new_tokens"],
            context_window=config["data"]["context_window"],
            temperature=sampling.get("temperature", 1.0),
            top_k=sampling.get("top_k"),
            device=device,
        )
        logger.log_text("samples", f"step {step}\n\n{text}", step)

    return sample_fn


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 0))
    device = resolve_device(config.get("device", "auto"))

    data = config["data"]
    tokenizer = BPETokenizer.load(data["tokenizer"]["output"])
    processed_dir = Path(data["processed_dir"])
    train_tokens = torch.load(processed_dir / "train.pt")
    val_tokens = torch.load(processed_dir / "val.pt")

    training = config["training"]
    train_loader = build_dataloader(
        train_tokens,
        context_window=data["context_window"],
        batch_size=training["batch_size"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_tokens,
        context_window=data["context_window"],
        batch_size=training["batch_size"],
        shuffle=False,
    )

    model = build_model(config["model"], vocab_size=tokenizer.vocab_size)

    optimizer = build_optimizer(
        model,
        learning_rate=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=training["warmup_steps"],
        max_steps=training["max_steps"],
        min_lr_ratio=training["min_lr_ratio"],
    )

    run_dir = Path(training["checkpoint_dir"]) / config["run_name"]
    n_params = sum(p.numel() for p in model.parameters())

    with Logger(run_dir / "tb") as logger:
        logger.log_text("config", yaml.safe_dump(config, sort_keys=False))
        logger.log(0, {"model/num_parameters": n_params})

        sample_fn = (
            build_sample_fn(config, model, tokenizer, logger, device)
            if "sampling" in config
            else None
        )

        trainer = Trainer(
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            logger,
            device=device,
            run_dir=run_dir,
            grad_clip=training.get("grad_clip"),
            eval_batches=training.get("eval_batches"),
            sample_fn=sample_fn,
        )

        print(f"run: {config['run_name']}  device: {device}  params: {n_params:,}")

        trainer.train(
            max_steps=training["max_steps"],
            eval_interval=training["eval_interval"],
            log_interval=training.get("log_interval", 10),
            resume=training.get("resume", False),
        )

        best_val_loss = trainer.best_val_loss
        logger.log_hparams(
            flatten_config(config) | {"model.num_parameters": n_params},
            {
                "val_loss": best_val_loss,
                "val_perplexity": math.exp(best_val_loss),
            },
        )


if __name__ == "__main__":
    main()
