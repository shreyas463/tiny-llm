"""Held-out perplexity evaluation."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from tiny_llm.config import ensure_dir, load_config
from tiny_llm.dataset import RandomBatchLoader, tokenize_corpus_to_ids
from tiny_llm.util import load_checkpoint, resolve_device, set_seed

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    cfg: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if cfg is None:
        cfg = load_config(config_path)  # type: ignore[arg-type]

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    tok_dir = cfg["tokenizer"]["artifact_dir"]

    ckpt_path = Path(checkpoint) if checkpoint else Path(train_cfg["checkpoint_dir"]) / "best.pt"
    if not ckpt_path.exists():
        alt = Path(train_cfg["checkpoint_dir"]) / "last.pt"
        if alt.exists():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    model, ckpt = load_checkpoint(ckpt_path, device)
    model.eval()

    processed = Path(data_cfg["processed_dir"])
    val_txt = processed / data_cfg.get("val_file", "val.txt")
    val_ids = tokenize_corpus_to_ids(
        val_txt, tok_dir, cache_path=processed / "val_ids.npy"
    )

    block_size = model.config.block_size
    batch_size = int(train_cfg["batch_size"])
    eval_iters = int(train_cfg.get("eval_iters", 50))
    loader = RandomBatchLoader(val_ids, block_size, batch_size, device)

    losses: list[float] = []
    for _ in range(eval_iters):
        x, y = loader.get_batch()
        _, loss = model(x, y)
        losses.append(float(loss.item()))

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(min(mean_loss, 20))
    result = {
        "checkpoint": str(ckpt_path),
        "step": ckpt.get("step"),
        "val_loss": mean_loss,
        "val_perplexity": ppl,
        "eval_iters": eval_iters,
        "device": str(device),
        "params": model.num_parameters(non_embedding=False),
    }

    out_dir = ensure_dir("artifacts/demo")
    out_path = out_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Val loss=%.4f perplexity=%.2f -> %s", mean_loss, ppl, out_path)
    return result
