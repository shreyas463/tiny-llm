# tiny-llm

A tiny **GPT-style language model trained from scratch** on [TinyStories](https://arxiv.org/abs/2305.07759): data → BPE tokenizer → decoder-only transformer → train → val perplexity → samples.

No product UI. Pure ML train → eval → writeup — the interview artifact for “I can train an LM end-to-end.”

## What you get

| Piece | Details |
| --- | --- |
| Data | TinyStoriesV2-GPT4 (public HF dump), capped train/val text |
| Tokenizer | Byte-level BPE (`tokenizers`), vocab configurable (1k smoke / 8k default) |
| Model | Decoder-only transformer (causal attn + MLP), weight-tied LM head |
| Train | AdamW, cosine LR w/ warmup, grad clip, checkpoints, JSONL loss log |
| Eval | Held-out cross-entropy → **perplexity**, generation script, loss curve plot |

## Architecture

```
token + position embed
  → N × (LayerNorm → causal MHA → residual → LayerNorm → MLP → residual)
  → LayerNorm → tied LM head
```

Config-driven sizes (`configs/*.yaml`):

| Config | Params (approx) | Layers | d_model | Heads | Context | Vocab | Target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke.yaml` | **~0.17M** | 2 | 64 | 2 | 64 | 1,024 | CPU CI / laptop smoke |
| `tiny.yaml` | **~10–15M** | 6 | 384 | 6 | 256 | 8,000 | Consumer GPU / Colab |

## Smoke results (CPU, this repo)

Reproduced with `configs/smoke.yaml` (50 steps, ~200k train chars):

| Metric | Value |
| --- | --- |
| Device | CPU |
| Parameters | 168,576 |
| Steps | 50 |
| Val loss | ~6.56 |
| Val perplexity | ~706 |
| Wall time | ~1s |

Loss curve:

![smoke loss curve](artifacts/demo/loss_curve.png)

Generation after 50 smoke steps is still gibberish (expected — undertrained tiny model). Qualitative samples after a real `tiny.yaml` GPU run belong in `artifacts/demo/samples.md`.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# CPU smoke (end-to-end in under a minute)
python3 -m tiny_llm.cli prepare   --config configs/smoke.yaml
python3 -m tiny_llm.cli tokenizer --config configs/smoke.yaml
python3 -m tiny_llm.cli train     --config configs/smoke.yaml
python3 -m tiny_llm.cli eval      --config configs/smoke.yaml
python3 -m tiny_llm.cli generate  --config configs/smoke.yaml
```

### GPU training (recommended for real metrics)

```bash
# Needs a CUDA GPU; uses configs/tiny.yaml (~10–15M params, 5k steps)
python3 -m tiny_llm.cli prepare   --config configs/tiny.yaml
python3 -m tiny_llm.cli tokenizer --config configs/tiny.yaml
python3 -m tiny_llm.cli train     --config configs/tiny.yaml
python3 -m tiny_llm.cli eval      --config configs/tiny.yaml
python3 -m tiny_llm.cli generate  --config configs/tiny.yaml --prompt "Once upon a time"
```

Tips:

- Raise `data.max_train_chars` / `train.max_steps` in `tiny.yaml` for a longer run.
- Checkpoints land in `checkpoints/` (gitignored). Small demo plots/metrics live under `artifacts/demo/`.
- `device: auto` picks CUDA when available; override with `device: cpu` or `cuda`.

## Layout

```
configs/               smoke (CPU) + tiny (GPU) YAML
src/tiny_llm/
  data.py              TinyStories download + train/val prep
  tokenizer.py         BPE train / encode / decode
  model.py             GPT decoder-only transformer
  dataset.py           token cache + random batch loader
  train.py             training loop + checkpoints
  evaluate.py          val perplexity
  generate.py          sampling
  plot.py              loss curves
  cli.py               prepare | tokenizer | train | eval | generate
artifacts/
  tokenizer/           committed smoke tokenizer
  demo/                loss curve, eval metrics, sample dump
tests/                 unit smoke tests (no network)
```

## Interview talking points

1. **Tokenizer from scratch** — not `tiktoken` / HF AutoTokenizer; trained BPE on the corpus.
2. **Model from scratch** — causal self-attention, residuals, LayerNorm, weight tying; size is a YAML knob.
3. **Train hygiene** — seed control, warmup + cosine LR, grad clip, best/last checkpoints, logged curves.
4. **Honest metrics** — report held-out perplexity + samples; smoke numbers are for pipeline proof, GPU `tiny` for quality.
5. **Reproducibility** — one config file drives data caps, vocab, architecture, and train schedule.

## License

MIT
