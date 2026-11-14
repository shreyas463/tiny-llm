"""Shared helpers: device, seed, dtype, checkpoint IO."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny_llm.config import ensure_dir
from tiny_llm.model import GPT, GPTConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    name = (dtype_name or "float32").lower()
    if name in {"bfloat16", "bf16"}:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        logger.warning("bfloat16 unsupported on %s; using float32", device)
        return torch.float32
    if name in {"float16", "fp16"}:
        if device.type == "cuda":
            return torch.float16
        logger.warning("float16 on CPU is slow/unstable; using float32")
        return torch.float32
    return torch.float32


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    best_val_loss: float | None = None,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "model_config": model.config.__dict__,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint -> %s", path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[GPT, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = GPTConfig(**ckpt["model_config"])
    model = GPT(mcfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
