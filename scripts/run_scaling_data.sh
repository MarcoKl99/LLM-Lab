#!/usr/bin/env bash
#
# Run the GPT data-scaling sweep: one training run per data budget, sequentially.
#
# Prerequisite — prepare the shared pool ONCE (fixed tokenizer + fixed val):
#   uv run python scripts/prepare_data.py --config configs/experiments/scaling_data/pool.yaml
#
# All runs reuse that pool and vary only data.max_train_tokens, so the tokenizer
# and validation set stay fixed. Runs use training.resume: true, so re-invoking
# resumes any incomplete runs.
#
# Usage:
#   scripts/run_scaling_data.sh              # all: d22m d55m d110m d164m
#   scripts/run_scaling_data.sh d22m d55m    # a subset, in the given order
#
set -uo pipefail

cd "$(dirname "$0")/.."

points=("$@")
if [ ${#points[@]} -eq 0 ]; then
  points=(d22m d55m d110m d164m)
fi

failed=()
for point in "${points[@]}"; do
  config="configs/experiments/scaling_data/${point}.yaml"

  if [ ! -f "$config" ]; then
    echo "!! missing config: $config" >&2
    failed+=("$point")
    continue
  fi

  echo "=== scaling_data/${point}  start $(date '+%Y-%m-%d %H:%M:%S') ==="
  if uv run python scripts/train.py --config "$config"; then
    echo "=== scaling_data/${point}  done  $(date '+%Y-%m-%d %H:%M:%S') ==="
  else
    echo "!! scaling_data/${point} FAILED  $(date '+%Y-%m-%d %H:%M:%S')" >&2
    failed+=("$point")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "!! failed points: ${failed[*]}" >&2
  exit 1
fi

echo "=== data-scaling sweep complete ==="
