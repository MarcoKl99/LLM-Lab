#!/usr/bin/env bash
#
# Run the GPT model-size scaling at a fixed data budget (D = 164M tokens) and a
# fixed step budget (20k). Only the model (width/depth) and the width-scaled LR
# differ from the XL reference (scaling_data/d164m).
#
# Prerequisite — the shared pool must exist (same one as the data/compute sweeps):
#   uv run python scripts/prepare_data.py --config configs/experiments/scaling_data/pool.yaml
#
# The XL reference (27M) already exists as artifacts/runs/scaling_data/d164m;
# this script trains the larger sizes.
#
# Usage:
#   scripts/run_scaling_model.sh            # all: xl2 xl3
#   scripts/run_scaling_model.sh xl2        # a subset, in the given order
#
set -uo pipefail

cd "$(dirname "$0")/.."

sizes=("$@")
if [ ${#sizes[@]} -eq 0 ]; then
  sizes=(xl2 xl3)
fi

failed=()
for size in "${sizes[@]}"; do
  config="configs/experiments/scaling_model/${size}.yaml"

  if [ ! -f "$config" ]; then
    echo "!! missing config: $config" >&2
    failed+=("$size")
    continue
  fi

  echo "=== scaling_model/${size}  start $(date '+%Y-%m-%d %H:%M:%S') ==="
  if uv run python scripts/train.py --config "$config"; then
    echo "=== scaling_model/${size}  done  $(date '+%Y-%m-%d %H:%M:%S') ==="
  else
    echo "!! scaling_model/${size} FAILED  $(date '+%Y-%m-%d %H:%M:%S')" >&2
    failed+=("$size")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "!! failed sizes: ${failed[*]}" >&2
  exit 1
fi

echo "=== model-size scaling complete ==="
