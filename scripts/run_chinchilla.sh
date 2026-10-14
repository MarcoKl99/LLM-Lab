#!/usr/bin/env bash
#
# Run the Chinchilla experiment: prepare the ~2B-token corpus (once), then train
# xl3 compute-optimally (k = 20 tokens/param, ~238k steps).
#
# The prepare step is skipped if the processed tensors and tokenizer already exist.
#
# Usage:
#   scripts/run_chinchilla.sh
#
set -uo pipefail

cd "$(dirname "$0")/.."

config="configs/experiments/fineweb_edu/chinchilla/xl3.yaml"

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
  echo "=== chinchilla/xl3  data + tokenizer already present — skipping prepare ==="
else
  echo "=== chinchilla/xl3  prepare  $(date '+%Y-%m-%d %H:%M:%S') ==="
  if ! uv run python scripts/prepare_data.py --config "$config"; then
    echo "!! chinchilla/xl3  prepare FAILED" >&2
    exit 1
  fi
fi

echo "=== chinchilla/xl3  train    $(date '+%Y-%m-%d %H:%M:%S') ==="
if ! uv run python scripts/train.py --config "$config"; then
  echo "!! chinchilla/xl3  train FAILED" >&2
  exit 1
fi

echo "=== chinchilla complete ==="
