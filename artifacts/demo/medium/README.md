# Medium-cpu experiment artifacts

Capacity jump attempt vs `tiny-cpu-long` (13.8M / ppl ≈ 6.59).

| | tiny-cpu-long (baseline) | **medium-cpu (this)** |
| --- | --- | --- |
| Params | 13.8M | **~31.5M** |
| Arch | 6L / 384d / 6H / V8k | **8L / 512d / 8H / V12k** |
| Best val ppl | **≈ 6.59** | ≈ **10.16** |
| Hardware | 4-core CPU | 4-core CPU (no GPU) |

**Honest conclusion:** on this CPU, the larger model saw fewer tokens per wall-hour and did **not** beat the smaller well-trained baseline. Use `configs/medium.yaml` on GPU for a fair capacity run.
