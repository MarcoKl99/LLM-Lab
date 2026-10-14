"""Export a run's train/val loss curves as CSV, for documentation.

Reads the TensorBoard scalars and prints one row per ``--log-interval`` steps:

    step,train_loss,val_loss

Values are taken from the logged points (train and val are logged at different
intervals, so a cell is left blank when that step has no point).

Usage:
    uv run python scripts/export_losses.py \
        --run-dir artifacts/runs/fineweb_edu/xl3 --log-interval 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.utils.tensorboard import read_scalars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/val loss curves as CSV.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Emit a row every N steps.",
    )
    parser.add_argument("--decimals", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scalars = read_scalars(args.run_dir)
    train = dict(scalars.get("train/loss", []))
    val = dict(scalars.get("val/loss", []))

    steps = sorted(s for s in set(train) | set(val) if s % args.log_interval == 0)

    print("step,train_loss,val_loss")
    for step in steps:
        t = f"{train[step]:.{args.decimals}f}" if step in train else ""
        v = f"{val[step]:.{args.decimals}f}" if step in val else ""
        print(f"{step},{t},{v}")


if __name__ == "__main__":
    main()
