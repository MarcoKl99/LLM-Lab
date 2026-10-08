"""Plot a run's training curves from its TensorBoard logs.

Intended for documentation: reads a run directory (its ``tb/`` logs for the
curves and ``latest.pt`` for metadata) and writes a PNG of the loss,
perplexity and learning-rate curves.

Usage:
    uv run python scripts/plot_curves.py --run-dir artifacts/runs/lstm_baseline \
        --out docs/baseline/lstm_baseline_curves.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from llm_lab.utils.tensorboard import read_scalars  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves for a run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir
    out = args.out or (run_dir / "curves.png")

    scalars = read_scalars(run_dir / "tb")

    def plot(tag: str, ax: plt.Axes, **kwargs: object) -> None:
        if tag in scalars:
            steps, values = zip(*scalars[tag])
            ax.plot(steps, values, **kwargs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    plot("train/loss", axes[0], label="train")
    plot("val/loss", axes[0], label="val", marker="o")
    axes[0].set(xlabel="step", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    plot("val/perplexity", axes[1], color="tab:orange", marker="o")
    axes[1].set(xlabel="step", ylabel="perplexity", title="Validation perplexity")
    axes[1].grid(alpha=0.3)

    plot("train/lr", axes[2], color="tab:green")
    axes[2].set(xlabel="step", ylabel="learning rate", title="Learning rate")
    axes[2].grid(alpha=0.3)

    title = run_dir.name
    checkpoint_path = run_dir / "latest.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        best = checkpoint.get("best_val_loss")
        title += (
            f"  (step {checkpoint.get('step')}, "
            f"best val loss {best:.4f}, ppl {math.exp(best):.1f})"
        )
    fig.suptitle(title)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
