# AGENTS.md

## Project

This is a serious, long-horizon research and learning project to build a modern
decoder-only language model from scratch — starting from random initialization
and working all the way to a genuinely useful, instruction-following model.

The primary goal is **understanding the process**, but the destination is
ambitious: a **Medium Language Model (MLM)** — a real, working model, not a toy.
This is meant to become a portfolio-grade **showcase** of the entire modern
pipeline (data → pre-training → instruction tuning → alignment → evaluation →
deployment), not a small hobby to be dropped after a week.

## Ambition and scope

We are aiming high, and we want to go **deep** into every stage of the modern
pipeline rather than stop at a small proof of concept:

1. **Data** — acquisition, curation, deduplication, tokenizer design, scaling.
2. **Pre-training** — architecture, optimization, scaling laws, compute budgets.
3. **Supervised / instruction fine-tuning** — turning a base model into an
   assistant that follows instructions.
4. **Alignment** — preference data, reward modeling, RLHF / DPO-style methods,
   and the associated safety considerations.
5. **Evaluation** — loss/perplexity, task benchmarks, and qualitative review.
6. **Inference & deployment** — sampling, efficiency, and a demo someone can
   actually use.

Treat this as a portfolio-grade body of work: clean, documented, reproducible,
and explainable to a technical audience. We want to be able to justify every
decision and demonstrate the result.

## Principles

### Aim high, build incrementally

The destination is ambitious — a useful Medium Language Model and a full-pipeline
showcase — but the path is incremental. Validate each mechanism at small scale
before spending real compute on the next step. Ambition sets the direction;
discipline sets the pace. Do not confuse "start small" with "stay small".

### Modular and iterative

Develop the project incrementally.

Start with small, simple experiments and gradually increase complexity and scale. Each stage should validate the previous one before moving on to more expensive experiments.

Avoid premature complexity.

### Reproducible

Experiments should be reproducible and clearly separated from one another.

Important experiment decisions and parameters should be explicitly defined rather than hidden in code or changed manually.

Results and artifacts must be traceable to the code and configuration that produced them.

### Configuration-driven

Use configuration files for experiment-specific decisions rather than hard-coding them into the implementation.

The configuration approach should evolve with the project; do not over-engineer it prematurely.

### Clean separation

Keep the actual training system separate from experimentation and analysis.

Notebooks may be used for exploration and analysis, but the core pipeline should remain clean, reusable and executable independently.

Generated artifacts and large outputs do not belong in Git.

### Understand before optimizing

Prefer implementations that make the underlying concepts understandable.

Do not introduce abstractions, frameworks or optimizations simply for convenience. First understand and validate the underlying mechanism, then optimize where it is actually useful.

### Experiment scientifically

Treat training runs as experiments rather than isolated attempts.

Change important variables deliberately, document hypotheses and compare results where useful.

Failed experiments are valuable and should help inform the next iteration.

Do not prematurely decide which metrics, tracking systems or infrastructure details are important. Evaluate these together as the project develops.

### AI's role

Work collaboratively and incrementally.

Before implementing substantial changes, explain the reasoning and relevant trade-offs.

Prefer small, understandable steps over large implementations.

When introducing an important ML concept, explain it well enough that the project owner can understand what is happening and why.

Do not unnecessarily make architectural decisions on behalf of the project. When there are meaningful alternatives, discuss them before committing to one.

## Ultimate goal

The goal is not simply:

> "Train a language model."

Nor is it only to understand each mechanism in isolation. The goal is:

> **Build a clean, reproducible, end-to-end experimental system; use it to
> understand, from scratch, how data, architecture, optimization and compute turn
> random parameters into a useful language model; and push that system as far as
> our compute allows — toward a genuine Medium Language Model and a showcase of
> the entire modern training pipeline (pre-training, instruction tuning,
> alignment, evaluation).**

Small steps are the method; a real, useful model is the destination. We want to
be able to explain and demonstrate every stage: why a choice was made, what it
changed, and what the numbers say.

Do not rush, but give me time to ask questions about the codebase in between.

## Compute

Development and most experiments run on an Apple M4 Pro (24 GB) — enough to
build, validate and iterate on the full pipeline and on small/medium models.
Scaling runs can use rented or free cloud GPUs (e.g. Google Colab). Keep the code
portable across devices (CPU / MPS / CUDA), treat local hardware as the place to
*understand*, and use cloud compute to *scale* when it matters.

## Tech-Stack

- PyTorch — Framework for the model, training and tensor-operations
- TensorBoard — Experiment tracking and visualization
- Hugging Face Datasets — Access to datasets
- Hugging Face Tokenizers — Tokenization
- PyYAML — Configurations via YAML-files
- Python — Language of the project

Finally... Let's build a language model together! ;)

## Directory Access Restrictions

- Only access, read, create, modify, or delete files and directories within the current project directory.
- Never access files or directories outside the project root, including parent directories, home directories, system directories, or other projects.
- Do not use shell commands, scripts, or other tools to read or modify files outside the project directory.
- Do not follow symbolic links that point outside the project directory.
- If a task requires access to files outside the project directory, stop and ask the user for explicit permission before proceeding.
- Treat these restrictions as mandatory and do not bypass them under any circumstances.

# Restrictions regarding Secrets

- Do NOT read the .env file, neither directly nor by writing a script or invoking a command that will read its contents
- You do NOT need any .env file for our tasks, do not attempt to read it
