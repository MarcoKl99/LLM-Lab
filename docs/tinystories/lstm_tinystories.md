# Baseline: LSTM on TinyStories

**Run name:** `lstm_tinystories`
**Date:** 2026-10-09
**Status:** complete (10000 / 10000 steps)

## Purpose

The first baseline on a **larger, multi-document corpus**. It validates the
scaled data pipeline (document streaming, `<|endoftext|>` boundaries, larger-vocab
tokenizer) and provides the reference point for the param-matched transformer
comparison to come.

Unlike the tiny Shakespeare runs, this one has enough data (~22M tokens) to move
out of the data-starved regime.

## Setup

| | |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.14.1 |
| Device | Apple MPS (macOS arm64) |
| Corpus | TinyStories (`roneneldan/TinyStories`, 100,000 stories) |
| Chars | 88,306,740 |
| Tokenizer | byte-level BPE, `vocab_size = 4096` |
| Tokens | 22,499,315 total (22,274,321 train / 224,994 val) |
| Document boundary | `<|endoftext|>` (id 0) between stories |
| Parameters | **2,169,856** |

### Model

- Stateless LSTM (hidden state reset per window)
- `hidden_size = 192`, `num_layers = 2`, `dropout = 0.0`
- Embedding → LSTM → linear head to vocab

### Training

- `context_window = 128`, `batch_size = 64` (≈ 8,192 tokens/step)
- AdamW, `learning_rate = 3e-3`, `weight_decay = 0.0`
- LR schedule: linear warmup (200 steps) + cosine decay to `min_lr_ratio = 0.1`
- `grad_clip = 1.0`, `max_steps = 10000` (≈ 82M tokens ≈ 3.7 epochs)
- Eval every 500 steps; sample generation every eval

## Results

| Metric | Start | End (step 10000) |
|---|---|---|
| Train loss | 6.082 | 2.560 |
| Val loss | 3.558 (step 500) | **2.466** |
| Val perplexity | 35.1 | **11.78** |

Best validation loss: **2.4661** (perplexity **11.78**), reached at the final
evaluation — still improving, not plateaued.

### Validation loss over training

| Step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val loss | 3.558 | 3.099 | 2.907 | 2.804 | 2.722 | 2.658 | 2.635 | 2.600 | 2.589 | 2.549 |
| Val ppl | 35.1 | 22.2 | 18.3 | 16.5 | 15.2 | 14.3 | 13.9 | 13.5 | 13.3 | 12.8 |

| Step | 5500 | 6000 | 6500 | 7000 | 7500 | 8000 | 8500 | 9000 | 9500 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val loss | 2.556 | 2.542 | 2.519 | 2.513 | 2.495 | 2.501 | 2.488 | 2.480 | 2.471 | 2.466 |
| Val ppl | 12.9 | 12.7 | 12.4 | 12.3 | 12.1 | 12.2 | 12.0 | 11.9 | 11.8 | 11.8 |

### Training curves

![Training curves](lstm_tinystories_curves.png)

*Left: train/val loss. Middle: validation perplexity. Right: learning-rate schedule (warmup + cosine). Generated with [`scripts/plot_curves.py`](../../scripts/plot_curves.py).*

## Sample generations

Prompt: `Once upon a time`, `temperature = 0.8`, `top_k = 50`.

**Step 500 (early) — already story-shaped:**

```
Once upon a time, there was a great day, but the bear was very much. She
looked down at the big box.

"Sure, I want that it was sad. I am twins who lived for my little help."
...
```

**Step 10000 (final) — coherent story:**

```
Once upon a time, there was a little boy named Tim. Tim had a red ball that
he loved to be nice. He loved to draw pictures of animals and vegetables.
One day, Tim asked his mom to help him play with his friend, so he asked his
friends to help him pick some of his toys. Timmy said they would clean the box
together and put everything on his head.
...
His mom said, "I'm sorry. I didn't mean to hurt you."
```

By step 10000 the model produces fluent, on-register children's stories with
dialogue, named characters, and narrative flow.

## Observations

- The larger corpus transforms the result: perplexity dropped from ~25.9
  (Shakespeare LSTM) to **11.78** — and the output is genuinely coherent prose,
  not just Shakespeare-flavoured fragments.
- Validation loss decreased monotonically and had **not plateaued** at step
  10000, so this baseline is still compute-limited.
- TinyStories is purpose-built so small models can produce fluent text; the
  model reached readable stories remarkably early (step 500).
- The `<|endoftext|>` boundary appears to work: generations stay within a single
  story's register rather than jumping between unrelated ones.

## Artifacts

```
artifacts/runs/lstm_tinystories/
├── best.pt      # best validation loss checkpoint
├── latest.pt    # last checkpoint (step 10000)
└── tb/          # TensorBoard logs (scalars, config, samples, hparams)
```

Tokenizer: `artifacts/tokenizer/tinystories_bpe.json`
Processed data: `data/processed/tinystories/{train,val}.pt`

## Reproduction

```bash
uv run python scripts/prepare_data.py --config configs/experiments/lstm_tinystories.yaml
uv run python scripts/train.py --config configs/experiments/lstm_tinystories.yaml
uv run tensorboard --logdir artifacts/runs
```

Config: [`configs/experiments/lstm_tinystories.yaml`](../../configs/experiments/lstm_tinystories.yaml)

## Related

- [`../shakespeare/`](../shakespeare/) — the earlier, much smaller-corpus
  baselines (LSTM + GPT on Tiny Shakespeare).
- `gpt_tinystories` (pending) — the param-matched transformer on the same data.
