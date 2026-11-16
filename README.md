# tiny-llm

**I trained a GPT from scratch** — tokenizer, model, train loop, eval, samples — on [TinyStories](https://arxiv.org/abs/2305.07759). No API wrappers. No product UI. Just the ML loop you can reproduce.

## Headline result

| | Baseline CPU (`tiny-cpu`, 5k) | **This run (`tiny-cpu-long`, 18k)** |
| --- | --- | --- |
| Params | 13.8M | **13.8M** (same architecture) |
| Train chars | ~49M | **~98M** |
| Steps | 5,000 | **18,000** |
| Hardware | 4-core CPU, no GPU | **4-core CPU, no GPU** |
| Wall clock | ~58 min | **~7.4 hours** |
| **Best val perplexity** | ≈ 13.4 | **≈ 6.59** |
| Held-out eval ppl | ≈ 14.2 | **≈ 6.93** |

**~2× lower perplexity** by doubling data and extending the schedule (milder dropout, longer warmup, deeper cosine floor) — not by changing the architecture.

![tiny-cpu-long loss curve](artifacts/demo/loss_curve.png)

### Sample (temperature 0.8)

> Once upon a time, there was a little girl named Sue. She was very patient when she always put her toys in the box. One day, Sue saw her friend, Tom, playing with a ball. …  
> Sue said, "Let's play the ball!" Tom liked the idea, so he said, "Okay, but be careful." …  
> Then, something unexpected happened. The ball began to talk! …

Full dump: [`artifacts/demo/samples.md`](artifacts/demo/samples.md). Stories now have characters, dialogue, and short arcs (still imperfect grammar — honest for ~7 ppl / 13.8M / CPU).

## Why this repo

For interviews / social proof you want to show:

1. **Tokenizer from scratch** — trained byte-level BPE on the corpus (not `tiktoken`)
2. **Model from scratch** — causal Transformer, residuals, LayerNorm, weight tying
3. **Train hygiene** — AdamW, warmup + cosine, grad clip, best/last checkpoints, logged curves, resume
4. **Honest metrics** — held-out perplexity + qualitative samples, with a clear before/after
5. **Reproduce in one config** — data caps, vocab, architecture, schedule in YAML

## Architecture

```
token + position embed
  → N × (LayerNorm → causal MHA → residual → LayerNorm → MLP → residual)
  → LayerNorm → tied LM head
```

| Config | Params | Layers / d / heads | Ctx | Vocab | Role |
| --- | --- | --- | --- | --- | --- |
| `smoke.yaml` | ~0.17M | 2 / 64 / 2 | 64 | 1k | Pipeline CI |
| `tiny-cpu.yaml` | 13.8M | 6 / 384 / 6 | 256 | 8k | Baseline CPU (~13.4 ppl) |
| **`tiny-cpu-long.yaml`** | **13.8M** | **6 / 384 / 6** | **256** | **8k** | **Best CPU result (~6.59 ppl)** |
| `tiny.yaml` | 13.8M | 6 / 384 / 6 | 256 | 8k | Same model, GPU path |
| `medium-cpu.yaml` | ~31.5M | 8 / 512 / 8 | 256 | 12k | Capacity jump (CPU; see below) |
| `medium.yaml` | ~31.5M | 8 / 512 / 8 | **384** | 12k | Same capacity, **GPU** path |

### Hyperparams (`tiny-cpu-long`)

| Knob | Value |
| --- | --- |
| Optimizer | AdamW (β₁=0.9, β₂=0.95), weight decay 0.1 |
| Peak LR | 3×10⁻⁴ · warmup 500 · cosine to **5%** floor |
| Batch | micro 8 × accum 4 → **effective 32** |
| Dropout | **0.05** (was 0.1 on the 5k run) |
| Grad clip | 1.0 |
| Weight tying | embedding ↔ LM head |

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Best CPU result (~7–8 hours on 4 cores)
python3 -m tiny_llm.cli prepare   --config configs/tiny-cpu-long.yaml
python3 -m tiny_llm.cli tokenizer --config configs/tiny-cpu-long.yaml
python3 -m tiny_llm.cli train     --config configs/tiny-cpu-long.yaml
# resume after interrupt:  ... train --config configs/tiny-cpu-long.yaml --resume
python3 -m tiny_llm.cli eval      --config configs/tiny-cpu-long.yaml
python3 -m tiny_llm.cli generate  --config configs/tiny-cpu-long.yaml
```

**GPU (faster iteration):** use `configs/tiny.yaml` (bf16, larger micro-batch).  
**Smoke:** `configs/smoke.yaml` for a sub-minute end-to-end check.

Tips: checkpoints are gitignored (`checkpoints/`). Demo plots/metrics/samples live under `artifacts/demo/`. Token caches fingerprint text + tokenizer so recipes don’t silently reuse stale IDs.

## Layout

```
configs/          smoke · tiny-cpu · tiny-cpu-long · tiny
src/tiny_llm/     data, tokenizer, model, train, eval, generate, CLI
artifacts/
  tokenizer/      8k BPE from the long run
  demo/           loss curve, metrics, samples
tests/            unit tests (no GPU required)
```

## Medium capacity experiment (honest)

We tried a real architecture jump on the same 4-core CPU (no GPU available):

| | **tiny-cpu-long (winner)** | medium-cpu (this experiment) |
| --- | --- | --- |
| Params | **13.8M** | ~31.5M (8L / 512d / 8H / V12k) |
| Steps / train chars | 18k / ~98M | 25k / ~148M |
| Wall clock | ~7.4 h | ~8.4 h (plus earlier segments) |
| **Best val ppl** | **≈ 6.59** | ≈ **10.16** |

**Did not beat the baseline.** The larger model is under-trained in tokens/wall-hour on CPU (smaller micro-batch). Samples still form stories but are weaker than tiny-cpu-long. Artifacts: [`artifacts/demo/medium/`](artifacts/demo/medium/).

**To actually unlock the capacity jump:** run `configs/medium.yaml` on a CUDA GPU (bf16, larger batch, ctx 384).

```bash
python3 -m tiny_llm.cli prepare   --config configs/medium.yaml
python3 -m tiny_llm.cli tokenizer --config configs/medium.yaml
python3 -m tiny_llm.cli train     --config configs/medium.yaml
python3 -m tiny_llm.cli eval      --config configs/medium.yaml
python3 -m tiny_llm.cli generate  --config configs/medium.yaml
```

## What’s next (intentional, not hope-and-rerun)

- **GPU `medium.yaml`** — fair test of the 32M recipe (this is the real remaining quality jump)
- **GPU `tiny.yaml`** — faster iteration on the proven 13.8M recipe
- **Data** — uncapped TinyStories train dump when download/time allow

## License

MIT
