# Demo artifacts

Primary results from the **tiny-cpu** training run (`configs/tiny-cpu.yaml`):

| File | Meaning |
| --- | --- |
| `loss_curve.png` | Train/val loss over 5,000 steps |
| `loss.jsonl` | Raw logged losses |
| `eval_metrics.json` | Held-out loss + perplexity |
| `train_summary.json` | Run summary (params, device, wall time, hyperparams) |
| `samples.json` / `samples.md` | Generation dump |

**Hardware:** 4-core CPU, no GPU · **wall clock:** ~58 min · **best val ppl:** ~13.4

Checkpoints (`checkpoints/tiny-cpu/*.pt`, ~158MB) are gitignored — regenerate with the reproduce commands in the root README.
