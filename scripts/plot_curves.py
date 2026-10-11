"""Plot training curves from a run's TensorBoard logs.

With a single run directory it writes the per-run figure (train/val loss,
validation perplexity, learning rate). With several it writes a comparison
figure (train and validation loss, one line per run).

Usage:
    uv run python scripts/plot_curves.py --run-dir artifacts/runs/baseline/lstm \
        --out docs/shakespeare/lstm_curves.png

    uv run python scripts/plot_curves.py \
        --run-dir artifacts/runs/scaling_data/d22m artifacts/runs/scaling_data/d55m \
                  artifacts/runs/scaling_data/d110m artifacts/runs/scaling_data/d164m \
        --out docs/scaling_data/data_scaling_curves.png
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from llm_lab.utils.tensorboard import read_scalars  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        nargs="+",
        help="One run directory (per-run figure) or several (comparison figure).",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _plot_tag(
    scalars: Mapping[str, list[tuple[int, float]]],
    tag: str,
    ax: plt.Axes,
    **kwargs: object,
) -> None:
    if tag in scalars:
        steps, values = zip(*scalars[tag])
        ax.plot(steps, values, **kwargs)


def plot_single(run_dir: Path, out: Path) -> None:
    scalars = read_scalars(run_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    _plot_tag(scalars, "train/loss", axes[0], label="train")
    _plot_tag(scalars, "val/loss", axes[0], label="val", marker="o")
    axes[0].set(xlabel="step", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    _plot_tag(scalars, "val/perplexity", axes[1], color="tab:orange", marker="o")
    axes[1].set(xlabel="step", ylabel="perplexity", title="Validation perplexity")
    axes[1].grid(alpha=0.3)

    _plot_tag(scalars, "train/lr", axes[2], color="tab:green")
    axes[2].set(xlabel="step", ylabel="learning rate", title="Learning rate")
    axes[2].grid(alpha=0.3)

    title = run_dir.name
    checkpoint_path = run_dir / "latest.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
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


def plot_comparison(run_dirs: Sequence[Path], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run_dir in run_dirs:
        scalars = read_scalars(run_dir)
        name = run_dir.name
        _plot_tag(scalars, "train/loss", axes[0], label=name)
        _plot_tag(scalars, "val/loss", axes[1], label=name, marker="o", markersize=3)

    axes[0].set(xlabel="step", ylabel="loss", title="Train loss")
    axes[1].set(xlabel="step", ylabel="loss", title="Validation loss")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Run comparison")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main() -> None:
    args = parse_args()
    if len(args.run_dir) == 1:
        out = args.out or (args.run_dir[0] / "curves.png")
        plot_single(args.run_dir[0], out)
    else:
        out = args.out or Path("comparison_curves.png")
        plot_comparison(args.run_dir, out)


if __name__ == "__main__":
    main()
