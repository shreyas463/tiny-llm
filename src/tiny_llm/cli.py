"""CLI entrypoints for prepare / tokenizer / train / eval / generate."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tiny_llm.config import load_config


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        "-c",
        default="configs/smoke.yaml",
        help="Path to YAML config (default: configs/smoke.yaml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")


def prepare_data(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and prepare TinyStories text")
    _add_config_arg(parser)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from tiny_llm.data import prepare_tinystories

    cfg = load_config(args.config)
    paths = prepare_tinystories(cfg)
    print(f"Prepared: train={paths['train']} val={paths['val']}")


def train_tokenizer(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    _add_config_arg(parser)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from tiny_llm.tokenizer import train_from_config

    cfg = load_config(args.config)
    tok = train_from_config(cfg)
    print(f"Tokenizer vocab_size={tok.get_vocab_size()} -> {cfg['tokenizer']['artifact_dir']}")


def train(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train tiny GPT")
    _add_config_arg(parser)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from tiny_llm.train import train_from_config

    summary = train_from_config(config_path=args.config)
    print(
        f"Done. best_val_ppl={summary['best_val_ppl']:.2f} "
        f"checkpoint={summary['checkpoint']}"
    )


def evaluate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate val perplexity")
    _add_config_arg(parser)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from tiny_llm.evaluate import evaluate_perplexity

    result = evaluate_perplexity(config_path=args.config, checkpoint=args.checkpoint)
    print(f"val_loss={result['val_loss']:.4f} val_ppl={result['val_perplexity']:.2f}")


def generate(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate text samples")
    _add_config_arg(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from tiny_llm.generate import generate_samples

    result = generate_samples(
        config_path=args.config,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        num_samples=args.num_samples,
    )
    for i, s in enumerate(result["samples"], 1):
        print(f"\n=== Sample {i} ===\n{s['completion']}\n")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "Usage: python -m tiny_llm.cli <prepare|tokenizer|train|eval|generate> [options]\n"
            "  --config configs/smoke.yaml   CPU smoke path\n"
            "  --config configs/tiny.yaml    GPU training path"
        )
        return

    cmd, rest = argv[0], argv[1:]
    dispatch = {
        "prepare": prepare_data,
        "tokenizer": train_tokenizer,
        "train": train,
        "eval": evaluate,
        "evaluate": evaluate,
        "generate": generate,
    }
    if cmd not in dispatch:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(2)
    # Ensure repo root is cwd-friendly for relative artifact paths.
    if not Path("configs").exists() and Path(__file__).resolve().parents[2].joinpath("configs").exists():
        import os

        os.chdir(Path(__file__).resolve().parents[2])
    dispatch[cmd](rest)


if __name__ == "__main__":
    main()
