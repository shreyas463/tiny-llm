# Demo artifacts

**Headline (best metrics):** `configs/tiny-cpu-long.yaml` — 13.8M / **val ppl ≈ 6.59**

| File | Meaning |
| --- | --- |
| `loss_curve.png` | Train/val loss over 18,000 steps |
| `loss.jsonl` | Raw logged losses |
| `eval_metrics.json` | Held-out loss + perplexity |
| `train_summary.json` | Params, hardware, hyperparams, wall time |
| `samples.json` / `samples.md` | Generation dump |
| `medium/` | Capacity-jump experiment (~31.5M); best ppl ≈ 10.16 — did **not** beat 6.59 on CPU |

Checkpoints under `checkpoints/` are gitignored.
