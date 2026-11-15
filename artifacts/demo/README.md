# Demo artifacts

Results from **`configs/tiny-cpu-long.yaml`** (stronger CPU run):

| File | Meaning |
| --- | --- |
| `loss_curve.png` | Train/val loss over 18,000 steps |
| `loss.jsonl` | Raw logged losses |
| `eval_metrics.json` | Held-out loss + perplexity |
| `train_summary.json` | Params, hardware, hyperparams, wall time |
| `samples.json` / `samples.md` | Generation dump |

**Hardware:** 4-core CPU, no GPU · **wall clock:** ~7.4 h · **best val ppl:** ≈ 6.59 (vs ≈ 13.4 at 5k steps)

Checkpoints (`checkpoints/tiny-cpu-long/*.pt`) are gitignored.
