# LLM-Lab 👨‍🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Building a decoder-only language model from scratch — one small, reproducible
experiment at a time — to understand how **data, architecture, optimization and
compute** turn random parameters into a useful language model.

The goal is not just "train a model", but to build a **clean, config-driven
experimental system** and understand every step along the way.

## Status 🚦

The full pipeline is in place and validated end-to-end with two architectures
that share one training harness, across two corpora:

| | Data | Model | Params | Val loss | Val ppl |
|---|---|---|---|---|---|
| [`lstm_baseline`](docs/shakespeare/lstm_baseline.md) | Tiny Shakespeare | LSTM | 1,577,984 | 3.233 | 25.36 |
| [`gpt_baseline`](docs/shakespeare/gpt_baseline.md) | Tiny Shakespeare | GPT (decoder-only) | 932,608 | 3.216 | 24.93 |
| [`lstm_tinystories`](docs/tinystories/lstm_tinystories.md) | TinyStories (~22M tokens) | LSTM | 2,169,856 | 2.324 | 10.21 |
| [`gpt_tinystories`](docs/tinystories/gpt_tinystories.md) | TinyStories (~22M tokens) | GPT (decoder-only) | 2,145,984 | **1.694** | **5.44** |

All runs were still improving when they stopped. Scaling the corpus from Tiny
Shakespeare (~0.46M tokens) to TinyStories (~22M tokens) dropped the LSTM's
validation perplexity from 25.4 to **10.2** and produced genuinely coherent
stories. In the **param-matched** TinyStories comparison (~2.15M params each),
the transformer reaches **5.44** perplexity versus the LSTM's **10.2**. See
[`docs/`](docs/) for full write-ups, samples and curves.

## What's implemented ❓

- **Data pipeline** — corpus download/cache and streamed HuggingFace documents,
  byte-level BPE tokenizer, `<|endoftext|>` document boundaries, next-token
  windowing, batching.
- **Models** — a stateless **LSTM** baseline and a **GPT** decoder-only
  transformer (pre-LN, learned positions, GELU, tied embeddings), both behind a
  single `forward(input_ids) -> logits` contract.
- **Training harness** (architecture-agnostic) — AdamW, warmup + cosine LR
  schedule, gradient clipping, validation, checkpointing + **resume**,
  optional early stopping, TensorBoard logging, and sample-text generation
  during training.
- **Generation** — temperature / top-k / greedy sampling with early stopping at
  `<|endoftext|>`, shared by both models.
- **Reproducibility** — everything driven by YAML configs; a thin `scripts/`
  layer over a testable `src/llm_lab` library.

## Project structure 🏗️

```
LLM-Lab/
├── configs/experiments/     # one YAML per experiment (data + model + training)
├── src/llm_lab/             # the reusable library
│   ├── config.py            # YAML loading / flattening
│   ├── generation.py        # autoregressive sampling
│   ├── data/                # corpus, tokenizer, dataset, prepare
│   ├── model/               # base contract, lstm, attention, block, gpt, build
│   ├── training/            # optimizer + trainer
│   └── utils/               # logging (TensorBoard), device, tensorboard reader
├── scripts/                 # thin entrypoints
│   ├── prepare_data.py
│   ├── train.py
│   ├── plot_curves.py
│   └── inspect_tokens.py
├── tests/                   # mirrors src/llm_lab
├── docs/                    # dataset notes + experiment write-ups + figures
│   ├── datasets/            # one short note per corpus (+ sample documents)
│   ├── shakespeare/         # first baselines
│   ├── tinystories/         # TinyStories baselines
│   ├── scaling*/            # scaling experiments (size / data / compute)
│   └── fineweb_edu/         # corpus swap + tokenizer sweep + chinchilla
├── data/                    # raw + processed data         (gitignored)
└── artifacts/               # tokenizers, checkpoints, logs (gitignored)
```

## Datasets 📚

The corpora we train on — each has a short note with **sample documents** in
[`docs/datasets/`](docs/datasets/):

| Corpus | Shape | Used by |
|---|---|---|
| [Tiny Shakespeare](docs/datasets/tiny_shakespeare.md) | single document (~1.1M chars) | first baselines |
| [TinyStories](docs/datasets/tinystories.md) | short children's stories, `<\|endoftext\|>`-separated | baselines + scaling sweeps |
| [FineWeb-Edu](docs/datasets/fineweb_edu.md) | educational web pages (~1.3T tokens total; we stream a slice) | the "real knowledge" line |

## Quickstart 🏃

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
# 1. Install dependencies (creates .venv + uv.lock)
uv sync

# 2. Prepare data: train tokenizer, write train/val token tensors
uv run python scripts/prepare_data.py --config configs/experiments/gpt_baseline.yaml

# 3. Train (resumes automatically if a checkpoint exists)
uv run python scripts/train.py --config configs/experiments/gpt_baseline.yaml

# 4. Watch training (scalars, config, sample completions, hparams)
uv run tensorboard --logdir artifacts/runs
```

Switch architecture by changing `model.type` in the config (`lstm` or `gpt`);
everything else stays the same.

## Scripts 📜

| Script | Purpose |
|---|---|
| `scripts/prepare_data.py --config <yaml>` | Train tokenizer + write `data/processed/**/{train,val}.pt` |
| `scripts/train.py --config <yaml>` | Train a model; checkpoint, resume, log to TensorBoard |
| `scripts/generate.py --config <yaml> [--prompt <text> \| --interactive]` | Prompt a trained checkpoint (defaults to `best.pt`) and print the completion |
| `scripts/plot_curves.py --run-dir <dir>... --out <png>` | Plot one run's curves, or compare several runs in one figure |
| `scripts/inspect_tokens.py --processed-dir <dir>` | Show how a processed token stream is laid out (documents + EOS) |

## Configuration ⚙️

Each experiment is a single YAML file with four sections:

```yaml
experiment: baseline    # groups runs in TensorBoard, e.g. artifacts/runs/baseline
run_name: gpt           # the model variant within the experiment
seed: 0
device: auto            # auto | cpu | cuda | mps

data:
  corpus: tiny_shakespeare
  context_window: 64
  val_fraction: 0.1
  tokenizer:
    vocab_size: 1024
    output: artifacts/tokenizer/tiny_shakespeare_bpe.json

model:
  type: gpt             # lstm | gpt
  n_layer: 4
  n_head: 4
  n_embd: 128

training:
  batch_size: 32
  learning_rate: 3.0e-3
  warmup_steps: 100
  max_steps: 2000
  eval_interval: 200
  resume: true
  # optional; max_steps still defines the LR-schedule horizon
  # early_stopping:
  #   patience: 5
  #   min_delta: 0.0

sampling:
  prompt: "ROMEO: "
  max_new_tokens: 200
  temperature: 0.8
  top_k: 50
  # optional; stop early at this token, or null to disable early stopping
  # eos_token: "<|endoftext|>"
```

Available configs: [`configs/experiments/`](configs/experiments/).

## Tests 🧪

```bash
uv run pytest
```

The suite (73 tests) covers the tokenizer, dataset/windowing, data prep,
optimizer/schedule, trainer (including resume), generation, and both models
(including a causality check for attention and the GPT).

## Tech stack 📺

- **PyTorch** — models, training, tensors
- **Hugging Face Tokenizers** — byte-level BPE
- **Hugging Face Datasets** — dataset access (for larger corpora)
- **TensorBoard** — experiment tracking
- **PyYAML** — configs
- **tqdm** — training progress
- **matplotlib** — documentation plots
- **uv** — dependency management

## Roadmap 🚗

- [x] Data pipeline (corpus → tokenizer → windows → batches)
- [x] Config-driven, reproducible experiment setup
- [x] LSTM baseline + GPT transformer, shared harness
- [x] Checkpoint/resume, TensorBoard logging, sample generation
- [x] Scale to a larger corpus (TinyStories) with a larger-vocab tokenizer
- [x] Param-matched LSTM vs. transformer comparison on TinyStories (transformer
      wins: 5.44 vs. 10.2 val perplexity)
- [x] Scaling experiments (model size / data / compute)
- [x] Corpus swap to real web text (FineWeb-Edu)
- [x] Tokenizer vocab sweep (settled at **16k**)
- [x] Chinchilla compute-optimal run — 97.7M params, 1.95B tokens (1.089 bits/byte)
- [ ] Larger compute-optimal model (~350M) → instruction tuning / alignment

## Principles 📚

This project is deliberately incremental and scientific: small experiments,
explicit configs, reproducible artifacts, and understanding before optimizing.
See [`OPENCODE.md`](OPENCODE.md) for the full guiding principles.

## License 📄

Released under the [MIT License](LICENSE).
