#!/usr/bin/env bash
#
# Run the GPT compute sweep: fixed data budget (D = 164M), varying step budget.
#
# Prerequisite — the shared pool must exist (same one as the data sweep):
#   uv run python scripts/prepare_data.py --config configs/experiments/scaling_data/pool.yaml
#
# The 20k point already exists as artifacts/runs/scaling_data/d164m; this script
# trains only the 10k, 40k and 80k points. Each run anneals its own cosine
# schedule to its max_steps, so every endpoint is a fair comparison.
#
# Usage:
#   scripts/run_scaling_compute.sh              # all: d164m_10k d164m_40k d164m_80k
#   scripts/run_scaling_compute.sh d164m_10k    # a subset, in the given order
#
set -uo pipefail

cd "$(dirname "$0")/.."

points=("$@")
if [ ${#points[@]} -eq 0 ]; then
  points=(d164m_10k d164m_40k d164m_80k)
fi

failed=()
for point in "${points[@]}"; do
  config="configs/experiments/scaling_compute/${point}.yaml"

  if [ ! -f "$config" ]; then
    echo "!! missing config: $config" >&2
    failed+=("$point")
    continue
  fi

  echo "=== scaling_compute/${point}  start $(date '+%Y-%m-%d %H:%M:%S') ==="
  if uv run python scripts/train.py --config "$config"; then
    echo "=== scaling_compute/${point}  done  $(date '+%Y-%m-%d %H:%M:%S') ==="
  else
    echo "!! scaling_compute/${point} FAILED  $(date '+%Y-%m-%d %H:%M:%S')" >&2
    failed+=("$point")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "!! failed points: ${failed[*]}" >&2
  exit 1
fi

echo "=== compute sweep complete ==="
