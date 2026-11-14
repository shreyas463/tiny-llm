"""Text generation from a trained checkpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from tiny_llm.config import ensure_dir, load_config
from tiny_llm.tokenizer import BOS, EOS, decode, encode, load_tokenizer
from tiny_llm.util import load_checkpoint, resolve_device, set_seed

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_samples(
    cfg: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    prompt: str | None = None,
    num_samples: int = 3,
) -> dict[str, Any]:
    if cfg is None:
        cfg = load_config(config_path)  # type: ignore[arg-type]

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    gen_cfg = cfg.get("generate", {})
    train_cfg = cfg["train"]
    tok_dir = cfg["tokenizer"]["artifact_dir"]

    ckpt_path = Path(checkpoint) if checkpoint else Path(train_cfg["checkpoint_dir"]) / "best.pt"
    if not ckpt_path.exists():
        alt = Path(train_cfg["checkpoint_dir"]) / "last.pt"
        if not alt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        ckpt_path = alt

    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()
    tokenizer = load_tokenizer(tok_dir)
    eos_id = tokenizer.token_to_id(EOS)
    bos_id = tokenizer.token_to_id(BOS)

    prompt_text = prompt if prompt is not None else str(gen_cfg.get("prompt", "Once upon a time"))
    max_new = int(gen_cfg.get("max_new_tokens", 64))
    temperature = float(gen_cfg.get("temperature", 0.8))
    top_k = gen_cfg.get("top_k", 40)
    top_k = int(top_k) if top_k is not None else None

    samples: list[dict[str, str]] = []
    for i in range(num_samples):
        # Manual BOS + prompt tokens (skip auto EOS from post-processor).
        ids = [bos_id] + encode(tokenizer, prompt_text, add_special_tokens=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )
        text = decode(tokenizer, out[0].tolist(), skip_special_tokens=True)
        samples.append({"prompt": prompt_text, "completion": text})
        logger.info("Sample %d:\n%s\n", i + 1, text)

    result = {
        "checkpoint": str(ckpt_path),
        "prompt": prompt_text,
        "temperature": temperature,
        "top_k": top_k,
        "samples": samples,
    }
    out_dir = ensure_dir("artifacts/demo")
    out_path = out_dir / "samples.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Also a human-readable dump for the README / interview story.
    md_path = out_dir / "samples.md"
    lines = [f"# Generation samples\n\nPrompt: `{prompt_text}`\n"]
    for i, s in enumerate(samples, 1):
        lines.append(f"## Sample {i}\n\n{s['completion']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote samples -> %s", out_path)
    return result
