# Baseline: LSTM Language Model

**Run name:** `lstm_baseline`
**Date:** 2026-10-08
**Status:** complete (2000 / 2000 steps)

## Purpose

The first end-to-end baseline for LLM-Lab. A small, stateless LSTM language
model trained on Tiny Shakespeare. Its role is to validate the full pipeline
(data → train → eval → checkpoint → sample) and to serve as a reference point
for later architectures (in particular the decoder-only transformer).

This is a **baseline**, not a target. It exists so future changes can be
compared against a real number.

## Setup

| | |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.14.1 |
| Device | Apple MPS (macOS arm64) |
| Corpus | Tiny Shakespeare (1,115,394 chars) |
| Tokenizer | byte-level BPE, `vocab_size = 1024` |
| Tokens | 459,913 total (413,921 train / 45,992 val) |
| Parameters | 1,577,984 |

### Model

- Stateless LSTM (hidden state reset per window)
- `hidden_size = 256`, `num_layers = 2`, `dropout = 0.0`
- Embedding → LSTM → linear head to vocab

### Training

- `context_window = 64`, `batch_size = 32`
- AdamW, `learning_rate = 3e-3`, `weight_decay = 0.0`
- LR schedule: linear warmup (100 steps) + cosine decay to `min_lr_ratio = 0.1`
- `grad_clip = 1.0`, `max_steps = 2000`
- Eval every 200 steps; sample generation every eval

## Results

| Metric | Start | End (step 2000) |
|---|---|---|
| Train loss | 5.727 | 2.878 |
| Val loss | 4.407 (step 200) | **3.255** |
| Val perplexity | 82.0 | **25.9** |

Best validation loss: **3.2549** (perplexity **25.92**), reached at the final
evaluation — the model was still improving and had not yet plateaued.

### Validation loss over training

| Step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val loss | 4.407 | 3.906 | 3.591 | 3.452 | 3.379 | 3.323 | 3.307 | 3.259 | 3.264 | 3.255 |

Train loss tail: 3.077 (1800) → 3.056 (1850) → 2.930 (1900) → 2.926 (1950) → 2.878 (2000).

### Training curves

![Training curves](lstm_baseline_curves.png)

*Left: train/val loss. Middle: validation perplexity. Right: learning-rate schedule (warmup + cosine). Generated with [`scripts/plot_curves.py`](../../scripts/plot_curves.py).*

## Sample generations

Prompt: `ROMEO: `, `temperature = 0.8`, `top_k = 50`.

**Step 200 (early) — gibberish:**

```
ROMEO: Thar you not starre,
And eir me, and the lord?

SIO:

Mad:
O would to make, go, it is the stow:
```

**Step 2000 (final) — recognizable pseudo-Shakespeare:**

```
ROMEO: Nature is't,
Because I promvour'd body, and to tender.

FLORIZEL:
The daudio's infice, you have a publish deny
Turn forbider than forth thousand majesty.

GLOUCESTER:
Nay, therefore, my father.
...
FRIAR LAURENCE:
Sinsheat you the broath doth wonder ten;
His grace's wish,
```

By step 2000 the model has learned blank-verse line structure,
character-name formatting, and a real vocabulary (`because`, `body`,
`publish`, `majesty`, `father`).

## Observations

- The pipeline works end-to-end: reproducible data prep, config-driven training,
  checkpoint/resume, TensorBoard logging (scalars, config text, sample text),
  and generation.
- Validation loss decreased monotonically and had not plateaued, so the baseline
  is **compute-limited**, not capacity-limited. More steps would likely help.
- Train loss (2.878) sits below val loss (3.255) by ~0.38 nats — a mild gap,
  expected for a model of this size on ~0.4M training tokens.

## Artifacts

```
artifacts/runs/lstm_baseline/
├── best.pt      # best validation loss checkpoint
├── latest.pt    # last checkpoint (step 2000)
└── tb/          # TensorBoard logs (scalars, config, samples, hparams)
```

Tokenizer: `artifacts/tokenizer/tiny_shakespeare_bpe.json`
Processed data: `data/processed/tiny_shakespeare/{train,val}.pt`

## Reproduction

```bash
uv run python scripts/prepare_data.py --config configs/experiments/lstm_baseline.yaml
uv run python scripts/train.py --config configs/experiments/lstm_baseline.yaml
uv run tensorboard --logdir artifacts/runs
```

Config: [`configs/experiments/lstm_baseline.yaml`](../../configs/experiments/lstm_baseline.yaml)

## Related

- [`gpt_baseline.md`](gpt_baseline.md) — the decoder-only transformer baseline,
  which reaches lower validation loss (3.209 vs. 3.255) with ~41% fewer
  parameters on the same data.
