"""Prompt a trained model loaded from its checkpoint.

Usage:
    # one-shot
    uv run python scripts/generate.py --config configs/experiments/gpt_tinystories.yaml \
        --prompt "Once upon a time"

    # interactive
    uv run python scripts/generate.py --config configs/experiments/gpt_tinystories.yaml --interactive

The model architecture, tokenizer and context window come from ``--config``; the
checkpoint defaults to ``best.pt`` in the run directory
(``checkpoint_dir / experiment / run_name``). Sampling settings default to the
config's ``sampling`` block and can be overridden on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_lab.config import load_config
from llm_lab.generation import generate
from llm_lab.inference import load_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text. If omitted, falls back to sampling.prompt or --interactive.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best', 'latest', or an explicit path to a .pt file.",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="Override the run directory."
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read prompts from stdin in a loop (blank line or Ctrl-D to quit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sampling = config.get("sampling", {})

    device = args.device or config.get("device", "auto")
    model, tokenizer, device = load_for_inference(
        config, checkpoint=args.checkpoint, run_dir=args.run_dir, device=device
    )

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else sampling.get("max_new_tokens", 200)
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else sampling.get("temperature", 0.8)
    )
    top_k = args.top_k if args.top_k is not None else sampling.get("top_k", 50)
    context_window = config["data"]["context_window"]
    eos_token = sampling.get("eos_token", "<|endoftext|>")

    def respond(prompt: str) -> str:
        return generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            context_window=context_window,
            temperature=temperature,
            top_k=top_k,
            device=device,
            eos_token=eos_token,
        )

    print(
        f"model: {config['experiment']}/{config['run_name']}  "
        f"checkpoint: {args.checkpoint}  device: {device}  "
        f"params: {sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"sampling: max_new_tokens={max_new_tokens} temperature={temperature} "
        f"top_k={top_k} eos_token={eos_token!r}"
    )

    if args.interactive:
        while True:
            try:
                prompt = input("\nprompt> ")
            except EOFError:
                print()
                break
            if not prompt.strip():
                continue
            print(respond(prompt))
        return

    prompt = args.prompt if args.prompt is not None else sampling.get("prompt")
    if prompt is None:
        raise SystemExit(
            "no prompt given: pass --prompt, or --interactive, "
            "or set sampling.prompt in the config"
        )
    print()
    print(respond(prompt))


if __name__ == "__main__":
    main()
