"""Loss curve plotting from JSONL training logs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tiny_llm.config import ensure_dir

logger = logging.getLogger(__name__)


def plot_loss_curves(log_path: Path, out_path: Path) -> Path | None:
    """Plot train/val loss from loss.jsonl. Returns path or None if matplotlib missing/empty."""
    if not log_path.exists():
        logger.warning("No loss log at %s", log_path)
        return None

    steps_train, train_losses = [], []
    steps_val, val_losses = [], []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "val_loss" in rec:
                steps_val.append(rec["step"])
                val_losses.append(rec["val_loss"])
                if "train_loss" in rec:
                    steps_train.append(rec["step"])
                    train_losses.append(rec["train_loss"])
            elif "train_loss" in rec:
                steps_train.append(rec["step"])
                train_losses.append(rec["train_loss"])

    if not steps_train and not steps_val:
        logger.warning("Empty loss log; skipping plot")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return None

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if steps_train:
        ax.plot(steps_train, train_losses, label="train", alpha=0.85, linewidth=1.5)
    if steps_val:
        ax.plot(steps_val, val_losses, label="val", alpha=0.9, linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("tiny-llm training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote loss curve -> %s", out_path)
    return out_path
