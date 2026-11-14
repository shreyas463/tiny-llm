"""Training loop with checkpoints, loss logging, and LR warmup."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from tiny_llm.config import ensure_dir, load_config
from tiny_llm.dataset import RandomBatchLoader, tokenize_corpus_to_ids
from tiny_llm.model import build_model_from_config
from tiny_llm.plot import plot_loss_curves
from tiny_llm.tokenizer import load_tokenizer_meta
from tiny_llm.util import (
    append_jsonl,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger(__name__)


def _lr_at_step(step: int, warmup: int, max_steps: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return lr * 0.1
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (0.1 + 0.9 * coeff)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    loaders: dict[str, RandomBatchLoader],
    eval_iters: int,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    use_autocast = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    for split, loader in loaders.items():
        losses = []
        for _ in range(eval_iters):
            x, y = loader.get_batch()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train_from_config(cfg: dict[str, Any] | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    if cfg is None:
        if config_path is None:
            raise ValueError("Provide cfg or config_path")
        cfg = load_config(config_path)

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    tok_cfg = cfg["tokenizer"]
    amp_dtype = resolve_dtype(str(train_cfg.get("dtype", "float32")), device)

    processed = Path(data_cfg["processed_dir"])
    train_txt = processed / data_cfg.get("train_file", "train.txt")
    val_txt = processed / data_cfg.get("val_file", "val.txt")
    if not train_txt.exists() or not val_txt.exists():
        raise FileNotFoundError("Missing processed text; run prepare first.")

    tok_dir = tok_cfg["artifact_dir"]
    meta = load_tokenizer_meta(tok_dir)
    vocab_size = int(meta["vocab_size"])

    train_ids = tokenize_corpus_to_ids(
        train_txt, tok_dir, cache_path=processed / "train_ids.npy"
    )
    val_ids = tokenize_corpus_to_ids(
        val_txt, tok_dir, cache_path=processed / "val_ids.npy"
    )

    block_size = int(cfg["model"]["block_size"])
    batch_size = int(train_cfg["batch_size"])
    loaders = {
        "train": RandomBatchLoader(train_ids, block_size, batch_size, device),
        "val": RandomBatchLoader(val_ids, block_size, batch_size, device),
    }

    model = build_model_from_config(cfg, vocab_size=vocab_size).to(device)
    n_params = model.num_parameters(non_embedding=False)
    logger.info(
        "Model params=%.2fM device=%s dtype=%s block=%d vocab=%d",
        n_params / 1e6,
        device,
        amp_dtype,
        block_size,
        vocab_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )

    run_dir = ensure_dir(train_cfg["run_dir"])
    ckpt_dir = ensure_dir(train_cfg["checkpoint_dir"])
    log_path = run_dir / "loss.jsonl"
    if log_path.exists():
        log_path.unlink()

    max_steps = int(train_cfg["max_steps"])
    warmup = int(train_cfg.get("warmup_steps", 0))
    accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    eval_interval = int(train_cfg.get("eval_interval", 100))
    eval_iters = int(train_cfg.get("eval_iters", 20))
    log_interval = int(train_cfg.get("log_interval", 10))
    base_lr = float(train_cfg["learning_rate"])

    use_autocast = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_autocast and amp_dtype == torch.float16))

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    model.train()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(range(max_steps), desc="train", disable=max_steps < 5)
    for step in pbar:
        lr = _lr_at_step(step, warmup, max_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = loaders["train"].get_batch()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
            _, loss = model(x, y)
            loss = loss / accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = float(loss.item()) * accum
        if step % log_interval == 0 or step == max_steps - 1:
            rec = {"step": step, "train_loss": train_loss, "lr": lr}
            append_jsonl(log_path, rec)
            history.append(rec)
            pbar.set_postfix(loss=f"{train_loss:.3f}", lr=f"{lr:.2e}")

        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model, loaders, eval_iters, amp_dtype, device)
            val_loss = losses["val"]
            rec = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": val_loss,
                "val_ppl": math.exp(min(val_loss, 20)),
                "lr": lr,
            }
            append_jsonl(log_path, rec)
            history.append(rec)
            logger.info(
                "step %d train=%.3f val=%.3f ppl=%.2f",
                step,
                losses["train"],
                val_loss,
                rec["val_ppl"],
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    cfg,
                    best_val_loss=best_val,
                )

    save_checkpoint(
        ckpt_dir / "last.pt",
        model,
        optimizer,
        max_steps - 1,
        cfg,
        best_val_loss=best_val,
    )

    plot_path = Path("artifacts/demo") / "loss_curve.png"
    plot_loss_curves(log_path, plot_path)

    elapsed = time.time() - t0
    summary = {
        "steps": max_steps,
        "best_val_loss": best_val,
        "best_val_ppl": math.exp(min(best_val, 20)),
        "params": n_params,
        "device": str(device),
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_dir / "best.pt"),
        "loss_log": str(log_path),
        "plot": str(plot_path),
    }
    (run_dir / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Training done: %s", summary)
    return summary
