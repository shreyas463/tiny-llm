# tiny-llm

A tiny GPT-style language model trained **from scratch** on [TinyStories](https://arxiv.org/abs/2305.07759): tokenizer → model → train → eval → samples.

> Status: scaffolding. Pipeline modules land in subsequent commits.

## Why this repo

Interview / portfolio artifact that shows the full LM training loop without a product UI:

1. Download a public corpus
2. Train a BPE tokenizer
3. Implement a decoder-only transformer
4. Train with checkpoints + loss logging
5. Report held-out perplexity and generation samples

## Quick start (CPU smoke)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m tiny_llm.cli prepare --config configs/smoke.yaml
python -m tiny_llm.cli tokenizer --config configs/smoke.yaml
python -m tiny_llm.cli train --config configs/smoke.yaml
python -m tiny_llm.cli eval --config configs/smoke.yaml
python -m tiny_llm.cli generate --config configs/smoke.yaml
```

GPU training uses `configs/tiny.yaml` (documented once training lands).

## Layout

```
configs/          # smoke (CPU) and tiny (GPU) YAML configs
src/tiny_llm/     # data, tokenizer, model, train, eval, generate
artifacts/        # tokenizer, plots, sample generations (small demos)
checkpoints/      # gitignored model weights
data/             # gitignored raw/processed text
```

## License

MIT
