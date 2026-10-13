#!/usr/bin/env bash
#
# Run the tokenizer sweep on FineWeb-Edu: for each vocab size, prepare the data
# (train tokenizer + tokenize the corpus) and then train the model. Sequential.
#
# The prepare step is SKIPPED when the processed tensors AND the tokenizer already
# exist, so re-running the sweep (e.g. to resume training) is cheap.
#
# The vocab-4096 baseline is the existing run `fineweb_edu/xl3` (not re-run).
#
# Usage:
#   scripts/run_tokenizer_sweep.sh            # all: v8192 v16384 v32768
#   scripts/run_tokenizer_sweep.sh v8192      # a subset, in the given order
#
set -uo pipefail

cd "$(dirname "$0")/.."

vocabs=("$@")
if [ ${#vocabs[@]} -eq 0 ]; then
  vocabs=(v8192 v16384 v32768)
fi

# Exit 0 if the config's processed tensors and tokenizer already exist.
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

failed=()
for v in "${vocabs[@]}"; do
  config="configs/experiments/fineweb_edu/tokenizer/${v}.yaml"

  if [ ! -f "$config" ]; then
    echo "!! missing config: $config" >&2
    failed+=("$v")
    continue
  fi

  if is_prepared "$config"; then
    echo "=== tokenizer/${v}  data + tokenizer already present — skipping prepare ==="
  else
    echo "=== tokenizer/${v}  prepare  $(date '+%Y-%m-%d %H:%M:%S') ==="
    if ! uv run python scripts/prepare_data.py --config "$config"; then
      echo "!! tokenizer/${v}  prepare FAILED" >&2
      failed+=("$v")
      continue
    fi
  fi

  echo "=== tokenizer/${v}  train    $(date '+%Y-%m-%d %H:%M:%S') ==="
  if uv run python scripts/train.py --config "$config"; then
    echo "=== tokenizer/${v}  done     $(date '+%Y-%m-%d %H:%M:%S') ==="
  else
    echo "!! tokenizer/${v}  train FAILED" >&2
    failed+=("$v")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "!! failed vocabs: ${failed[*]}" >&2
  exit 1
fi

echo "=== tokenizer sweep complete ==="
