#!/usr/bin/env bash
#
# Run the scaled-up Chinchilla experiment: prepare the ~8B-token corpus (once),
# then train the ~354M model compute-optimally (k = 20 tokens/param, ~864k steps).
#
# The prepare step is skipped if the processed tensors and tokenizer already exist.
# The tokenizer is reused from the 97.7M run (see the config's `reuse: true`).
#
# Usage:
#   scripts/run_xl4.sh
#
# Before the full run, consider the cheap sanity check first:
#   uv run python scripts/train.py --config \
#       configs/experiments/fineweb_edu/chinchilla/xl4_sanity.yaml
#
set -uo pipefail

cd "$(dirname "$0")/.."

config="configs/experiments/fineweb_edu/chinchilla/xl4.yaml"

is_prepared() {
  uv run python - "$1" <<'PY'
import sys
from pathlib import Path

from llm_lab.config import load_config

data = load_config(sys.argv[1])["data"]
processed = Path(data["processed_dir"])
tokenizer = Path(data["tokenizer"]["output"])
ready = (
    (processed / "train.pt").exists()
    and (processed / "val.pt").exists()
    and tokenizer.exists()
)
sys.exit(0 if ready else 1)
PY
}

if is_prepared "$config"; then
  echo "=== chinchilla/xl4  data + tokenizer already present — skipping prepare ==="
else
  echo "=== chinchilla/xl4  prepare  $(date '+%Y-%m-%d %H:%M:%S') ==="
  if ! uv run python scripts/prepare_data.py --config "$config"; then
    echo "!! chinchilla/xl4  prepare FAILED" >&2
    exit 1
  fi
fi

echo "=== chinchilla/xl4  train    $(date '+%Y-%m-%d %H:%M:%S') ==="
if ! uv run python scripts/train.py --config "$config"; then
  echo "!! chinchilla/xl4  train FAILED" >&2
  exit 1
fi

echo "=== chinchilla/xl4 complete ==="
