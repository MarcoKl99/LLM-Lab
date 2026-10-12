# Experiments — Recap

A short, scannable summary of every experiment so far and the **one-line lesson**
from each. Detailed write-ups live in [`docs/`](docs/). Newest at the bottom.

---

## 1. Baselines — Tiny Shakespeare

**Question:** does the full pipeline work end-to-end, and how do the two
architectures compare? · [details](docs/shakespeare/lstm_baseline.md)

| Run | Model | Params | Val loss | Val ppl |
|---|---|---|---|---|
| `baseline/lstm` | LSTM | 1,577,984 | 3.233 | 25.36 |
| `baseline/gpt` | GPT | 932,608 | 3.216 | 24.93 |

**Lesson:** the GPT beats the LSTM with ~41% fewer parameters — but Tiny
Shakespeare (~0.46M tokens) is far too small to conclude anything about
architecture.

---

## 2. Baselines — TinyStories (~22M tokens)

**Question:** how much does a bigger, multi-document corpus help? ·
[details](docs/tinystories/lstm_tinystories.md)

| Run | Model | Params | Val loss | Val ppl |
|---|---|---|---|---|
| `tinystories/lstm` | LSTM | 2,169,856 | 2.324 | 10.21 |
| `tinystories/gpt` | GPT | 2,145,984 | 1.694 | 5.44 |

**Lesson:** scaling the corpus (~0.46M → 22M tokens) improves quality
dramatically (LSTM ppl 25.9 → 10.2) and yields coherent text. Param-matched, the
transformer reaches **~half** the LSTM's perplexity.

---

## 3. Scaling model I — model size

**Question:** how does loss change with model size at a fixed data/compute budget
(TinyStories, 20k steps)? · [details](docs/scaling/model_size_sweep.md)

| Run | Params | LR | Best val loss |
|---|---|---|---|
| `s` | 2.15M | 3e-3 | 1.699 |
| `m` | 4.24M | 3e-3 | 1.570 |
| `l` | 12.27M | 3e-3 | 1.547 |
| `xl` | 27.4M | 3e-3 | 2.742 (**diverged**) |
| `xl_lr1e-3` | 27.4M | 1e-3 | **1.478** |

**Lessons:**
- Bigger → better, but with **diminishing returns** (suggesting data-limited).
- The largest model **diverged at `lr = 3e-3`**; dropping to **`1e-3`** fixed it
  and gave the best result → **larger models need a smaller LR.**

---

## 4. Scaling data

**Question:** was experiment 3 actually data-limited? Fix the model (XL @ 1e-3)
and compute (20k steps); vary only the unique-data budget `D`. ·
[details](docs/scaling_data/data_scaling.md)

| Run | `D` | Epochs | Train loss | Best val loss | Val ppl |
|---|---|---|---|---|---|
| `d22m` | 22M | 7.4 | 1.258 | 2.104 | 8.20 |
| `d55m` | 55M | 3.0 | 1.420 | 1.969 | 7.16 |
| `d110m` | 110M | 1.5 | 1.514 | **1.925** | **6.86** |
| `d164m` | 164M | 1.0 | 1.612 | 1.936 | 6.93 |

**Lessons:**
- **Small data → lower train loss** (memorization); **large data → lower val
  loss** (generalization).
- `d22m` **overfits**: val loss bottoms out, then *rises*, while train keeps
  falling.
- The benefit **saturates around ~1.5 epochs** (~110M tokens) → beyond that we are
  **compute-limited**, not data-limited.

---

## 5. Scaling compute

**Question:** at the now-adequate data budget (`D = 164M`), does more compute
(steps) help? Vary steps only. · [details](docs/scaling_compute/compute_sweep.md)

| Run | Steps | Epochs | Train loss | Best val loss | Val ppl |
|---|---|---|---|---|---|
| `d164m_10k` | 10k | 0.5 | 1.682 | 2.038 | 7.68 |
| `d164m` (20k) | 20k | 1.0 | 1.612 | 1.936 | 6.93 |
| `d164m_40k` | 40k | 2.0 | 1.534 | 1.865 | 6.46 |
| `d164m_80k` | 80k | 4.0 | 1.424 | **1.811** | **6.11** |

**Lessons:**
- More compute → better, **monotonically**, but with **strongly diminishing
  returns** — a power law `loss ≈ 3.41 · steps^(−0.057)` (10× compute buys only
  ~12% lower loss). A **soft** plateau, not a hard wall.
- **No overfitting** even at 4 epochs on 164M tokens (unlike `d22m` at 7.4).

---

## 6. Scaling model II — bigger models · **TODO (running)**

**Question:** at the adequate data budget (`D = 164M`, 20k steps), does a **larger
model** help? Fix data + compute; scale width/depth (aspect ratio 64, head_dim 64)
and scale LR down with width.

| Run | `n_layer`/`n_head`/`n_embd` | Params | LR |
|---|---|---|---|
| `xl` (reference = `scaling_data/d164m`) | 8/8/512 | 27.4M | 1e-3 |
| `xl2` | 10/10/640 | 51.9M | 8e-4 |
| `xl3` | 12/12/768 | 88.3M | 6e-4 |

**Status:** results pending. **Hypothesis:** the bigger models are
**data/compute-starved** at this budget (~2–3 tokens/param vs Chinchilla's ~20),
so we may see diminishing — or even negative — returns. Either outcome is
informative.

---

## Where this points

The three axes — **model size, data, compute** — all show **diminishing returns**.
The natural next step is finding the **compute-optimal balance** between them
(scale data *and* compute *with* the model), and eventually moving beyond
pre-training (instruction tuning, alignment).

---

## References

The scaling thread rests on two papers:

- **Kaplan et al., 2020** — *Scaling Laws for Neural Language Models* (OpenAI),
  [arXiv:2001.08361](https://arxiv.org/abs/2001.08361). The original N/D/C power
  laws; implied growing model size much faster than data.
- **Hoffmann et al., 2022** — *Training Compute-Optimal Large Language Models*
  (DeepMind), [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). The
  **"Chinchilla"** paper (named after their ~70B model). It corrected Kaplan: for
  a compute-optimal run, **model size and training tokens should scale in equal
  proportions — ~20 tokens per parameter**. Their 70B model (1.4T tokens) beat
  Gopher-280B, GPT-3-175B and MT-NLG-530B at equal compute, because those were
  *undertrained*.

This is the lens for experiments 3–6: moving one axis at a time flattens out
(non-optimal); the goal is to scale data **and** compute **with** the model. Note
that modern models often train *past* Chinchilla (hundreds–thousands of
tokens/param) because **inference** cost dominates — Chinchilla only optimizes
*training* compute.
