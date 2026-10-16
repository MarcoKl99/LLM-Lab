# Models — Overview

Every model architecture used in the project, with its size and parameter count.
Counts are computed directly from the configs (`build_model` + `tokenizer.vocab_size`).

All GPT variants share **one architecture**: decoder-only, pre-LN blocks, learned
absolute position embeddings, GELU MLP (4× expansion), tied input/output
embeddings, GPT-2-style init. Only `n_layer` / `n_head` / `n_embd` differ.

---

## LSTM

A stateless LSTM baseline (hidden state reset per window): `Embedding → LSTM →
Linear head`. Kept as the non-attention reference point.

| Name | hidden_size | num_layers | Params | Corpus |
|---|---|---|---|---|
| `baseline/lstm` | 256 | 2 | 1,577,984 | Tiny Shakespeare |
| `tinystories/lstm` | 192 | 2 | 2,169,856 | TinyStories |

> **Why the wider model has *fewer* params?** Vocabulary, not width, dominates
> here. `baseline/lstm` uses the Tiny Shakespeare tokenizer (`vocab = 1024`);
> `tinystories/lstm` uses TinyStories (`vocab = 4096`). The `embedding` and
> `lm_head` scale with `vocab × hidden`, while the recurrent core scales with
> `hidden²`. Going 256 → 192 hidden shrinks the core by ~0.46M, but 1024 → 4096
> vocab grows the embedding + head by ~1.05M — net **+0.59M**, so the narrower
> model ends up larger. Concretely, the embedding + head are ≈73% of
> `tinystories/lstm` vs ≈33% of `baseline/lstm`. (This is why the GPT variants
> below list their vocab, and why tying embeddings is a big win at large vocab.)

---

## GPT variants

| Name | n_embd | n_layers | n_heads | head_dim | Params |
|---|---|---|---|---|---|
| `baseline/gpt` | 128 | 4 | 4 | 32 | 932,608 |
| `tinystories/gpt` | 192 | 3 | 6 | 32 | 2,145,984 |
| `s` | 192 | 3 | 6 | 32 | 2,145,984 |
| `m` | 256 | 4 | 8 | 32 | 4,240,896 |
| `l` | 384 | 6 | 8 | 48 | 12,269,568 |
| `xl` | 512 | 8 | 8 | 64 | 27,382,784 |
| `xl2` | 640 | 10 | 10 | 64 | 51,939,840 |
| `xl3` | 768 | 12 | 12 | 64 | 88,300,032 |
| `xl3` @ **ctx 1024** (vocab 16k) | 768 | 12 | 12 | 64 | **98,425,344** |

### Notes

- **Same architecture, different names:** `s` and `tinystories/gpt` are identical;
  `xl` and the LR-ablation run `xl_lr1e-3` are identical (only the learning rate
  differs). Only unique architectures are listed once above.
- **Corpus / vocab:** `baseline/gpt` uses the Tiny Shakespeare tokenizer
  (`vocab = 1024`, `context = 64`); all others use the TinyStories tokenizer
  (`vocab = 4096`, `context = 128`).
- **Params depend on vocab *and* context:** the table lists the *vocab-4096,
  ctx-128* sizes. On the FineWeb-Edu tokenizers the same `xl3` architecture grows
  to **97.7M** (vocab 16k, used by the Chinchilla run) and **110.3M** (vocab 32k) —
  see the [tokenizer sweep](docs/fineweb_edu/tokenizer/vocab_sweep.md). The learned
  absolute position table also scales with the window, so the **ctx-1024** variant
  of `xl3` (vocab 16k) is **98,425,344** — see the
  [context experiment](docs/context/xl3_ctx1024.md).
- **Scaling shape:** the sweeps keep `n_embd / n_layer ≈ 64` and, for the larger
  models, `head_dim = 64`.
- **Next target:** a ~350M GPT, Chinchilla-optimal (~7B tokens), to be trained on
  rented GPUs. Exact width/depth (and vocab) to be fixed once we choose the
  corpus/tokenizer.
