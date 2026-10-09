#!/usr/bin/env bash
#
# Run the GPT scaling sweep: one training run per model size, sequentially.
#
# Configs live in configs/experiments/scaling/<size>.yaml. Runs use
# training.resume: true, so re-invoking this script resumes any incomplete runs
# instead of starting over.
#
# Usage:
#   scripts/run_scaling.sh          # all sizes, in order: s m l xl
#   scripts/run_scaling.sh m l      # a subset, in the given order
#
set -uo pipefail

cd "$(dirname "$0")/.."

sizes=("$@")
if [ ${#sizes[@]} -eq 0 ]; then
  sizes=(s m l xl)
fi

failed=()
for size in "${sizes[@]}"; do
  config="configs/experiments/scaling/${size}.yaml"

  if [ ! -f "$config" ]; then
    echo "!! missing config: $config" >&2
    failed+=("$size")
    continue
  fi

  echo "=== scaling/${size}  start $(date '+%Y-%m-%d %H:%M:%S') ==="
  if uv run python scripts/train.py --config "$config"; then
    echo "=== scaling/${size}  done  $(date '+%Y-%m-%d %H:%M:%S') ==="
  else
    echo "!! scaling/${size} FAILED  $(date '+%Y-%m-%d %H:%M:%S')" >&2
    failed+=("$size")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "!! failed sizes: ${failed[*]}" >&2
  exit 1
fi

echo "=== scaling sweep complete ==="
