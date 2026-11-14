# Demo artifacts

Small outputs from the CPU smoke run (`configs/smoke.yaml`):

| File | Meaning |
| --- | --- |
| `loss_curve.png` | Train/val loss over 50 steps |
| `loss.jsonl` | Raw logged losses |
| `eval_metrics.json` | Held-out loss + perplexity |
| `train_summary.json` | Run summary (params, device, wall time) |
| `samples.json` / `samples.md` | Generation dump (undertrained; pipeline proof) |

Checkpoints are gitignored under `checkpoints/`. Replace these files after a GPU `tiny.yaml` run for the interview-quality writeup.
