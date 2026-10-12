# Colab: AMP validation

**Experiment:** `colab/amp` (run `colab/amp/gpt_tinystories`)
**Date:** 2026-10-12
**Status:** complete (20,000 steps)

## Purpose

Verify that **Automatic Mixed Precision (AMP)** on a CUDA GPU (a free Colab T4):
(1) gives **the same** result as our local FP32 training, and (2) is **faster**.

## Setup

Same config as the local TinyStories GPT reference (`tinystories/gpt`) — a 2.1M
GPT, 20k steps — with mixed precision on. The only differences are the device and
the precision:

| | Local reference | Colab AMP |
|---|---|---|
| Device | Apple M4 Pro | NVIDIA T4 (Colab free) |
| Precision | FP32 | **FP16 (AMP)** |
| Config | `gpt_tinystories` | `amp/gpt_tinystories` (`amp: true`) |

On a T4 (Turing, compute capability 7.5) `amp_dtype: auto` selects **fp16** — it
has fp16 tensor cores but **not** bf16.

## Results

| | Best val loss | Best ppl | Final train loss | Throughput |
|---|---|---|---|---|
| Local M4 (FP32) | 1.6939 | 5.44 | 1.852 | ~176k tok/s |
| **Colab T4 (AMP, FP16)** | **1.6935** | **5.44** | 1.833 | **~299k tok/s** |

- **Same result:** best val loss differs by **0.0004** (1.6935 vs 1.6939) — pure
  noise. AMP is a numerical change, not a modelling one.
- **Faster:** ~**1.7×** the M4 Pro's throughput on this (small) model. The gap
  should widen for larger models, where tensor-core utilization is higher.

## The gotcha we hit (and fixed)

`torch.cuda.is_bf16_supported()` **defaults to `including_emulation=True`**, so it
returns **`True` on a T4** — even though Turing has no native bf16 tensor cores
(emulated bf16 falls back to fp32-ish math and gives **no** speedup). Our
`amp_dtype: auto` therefore wrongly picked bf16 on the T4 and showed no gain.

**Fix:** exclude emulation —
`torch.cuda.is_bf16_supported(including_emulation=False)` — which is `True` only
for native bf16 (Ampere+, `compute capability ≥ 8.0`). Now `auto` → **fp16 on
T4**, **bf16 on A100/H100**.

## Conclusion

AMP is **validated**: identical loss, real speedup, and CUDA-only (a no-op on
MPS/CPU). It's the key enabler for the cloud training runs.

## Reproduction

```bash
# Colab (T4):
python scripts/train.py --config configs/experiments/amp/gpt_tinystories.yaml   # amp: true
# or, without editing a config:
python scripts/train.py --config configs/experiments/gpt_tinystories.yaml --amp
```

Config: [`configs/experiments/amp/gpt_tinystories.yaml`](../../configs/experiments/amp/gpt_tinystories.yaml)
