# OPENCODE.md

## Project

This is a research and learning project to build a modern decoder-only language model from scratch, starting from random initialization and eventually reaching a useful completion/instruction model.

The primary goal is **understanding the process**.

## Principles

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

The goal is:

> **Build a clean, reproducible experimental system and understand how data, architecture, optimization and compute transform random parameters into a useful language model.**

Do not rush, but give me time to ask questions about the codebase in between.

## Tech-Stack

- PyTorch — Framework for the model, training and tensor-operations
- TensorBoard — Experiment tracking and visualization
- Hugging Face Datasets — Access to datasets
- Hugging Face Tokenizers — Tokenization
- PyYAML — Configurations via YAML-files
- Python — Language of the project

Finally... Let's build a language model together! ;)
