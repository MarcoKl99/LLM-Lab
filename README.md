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
| [`lstm_baseline`](docs/shakespeare/lstm_baseline.md) | Tiny Shakespeare | LSTM | 1,577,984 | 3.255 | 25.92 |
| [`gpt_baseline`](docs/shakespeare/gpt_baseline.md) | Tiny Shakespeare | GPT (decoder-only) | 932,608 | 3.209 | 24.76 |
| [`lstm_tinystories`](docs/tinystories/lstm_tinystories.md) | TinyStories (~22M tokens) | LSTM | 2,169,856 | **2.466** | **11.78** |

All runs were still improving when they stopped. Scaling the corpus from Tiny
Shakespeare (~0.46M tokens) to TinyStories (~22M tokens) dropped the LSTM's
validation perplexity from 25.9 to **11.8** and produced genuinely coherent
stories. See [`docs/`](docs/) for full write-ups, samples and curves.

> The param-matched LSTM-vs-transformer comparison on TinyStories is in progress
> (the GPT run is pending).

## What's implemented ❓

- **Data pipeline** — corpus download/cache and streamed HuggingFace documents,
  byte-level BPE tokenizer, `<|endoftext|>` document boundaries, next-token
  windowing, batching.
- **Models** — a stateless **LSTM** baseline and a **GPT** decoder-only
  transformer (pre-LN, learned positions, GELU, tied embeddings), both behind a
  single `forward(input_ids) -> logits` contract.
- **Training harness** (architecture-agnostic) — AdamW, warmup + cosine LR
  schedule, gradient clipping, validation, checkpointing + **resume**,
  TensorBoard logging, and sample-text generation during training.
- **Generation** — temperature / top-k / greedy sampling, shared by both models.
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
├── docs/                    # experiment write-ups + curve figures
│   ├── shakespeare/
│   └── tinystories/
├── data/                    # raw + processed data         (gitignored)
└── artifacts/               # tokenizers, checkpoints, logs (gitignored)
```

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
| `scripts/plot_curves.py --run-dir <dir> --out <png>` | Plot loss / perplexity / LR curves for documentation |
| `scripts/inspect_tokens.py --processed-dir <dir>` | Show how a processed token stream is laid out (documents + EOS) |

## Configuration ⚙️

Each experiment is a single YAML file with four sections:

```yaml
run_name: gpt_baseline
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

sampling:
  prompt: "ROMEO: "
  max_new_tokens: 200
  temperature: 0.8
  top_k: 50
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
- [ ] Param-matched LSTM vs. transformer comparison (LSTM done, GPT pending)
- [ ] Scaling experiments (data / model / compute)
- [ ] Instruction-tuning / fine-tuning toward a useful assistant

## Principles 📚

This project is deliberately incremental and scientific: small experiments,
explicit configs, reproducible artifacts, and understanding before optimizing.
See [`OPENCODE.md`](OPENCODE.md) for the full guiding principles.

## License 📄

Released under the [MIT License](LICENSE).
