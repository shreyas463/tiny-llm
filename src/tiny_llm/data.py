"""TinyStories download and train/val text preparation."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import requests

from tiny_llm.config import ensure_dir

logger = logging.getLogger(__name__)

# Public TinyStories text dumps (Karpathy / community mirrors of roneneldan/TinyStories).
# validation is small (~20MB unpacked) — ideal for CPU smoke.
TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "validation": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}

# Fallback: smaller HF dataset export if the GPT4 dumps are unavailable.
TINYSTORIES_FALLBACK_ZIP = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
)


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached download: %s", dest)
        return dest

    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    return dest


def _read_text_capped(path: Path, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return path.read_text(encoding="utf-8", errors="replace")
    # Read enough bytes for max_chars (UTF-8 worst case ~4 bytes/char).
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def _split_stories(text: str) -> list[str]:
    """Split TinyStories on blank lines into individual stories."""
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def prepare_tinystories(cfg: dict[str, Any]) -> dict[str, Path]:
    """Download TinyStories and write capped train/val text files.

    Returns paths to the written train and val files.
    """
    data_cfg = cfg["data"]
    raw_dir = ensure_dir(data_cfg["raw_dir"])
    processed_dir = ensure_dir(data_cfg["processed_dir"])

    source_split = data_cfg.get("source_split", "validation")
    url = TINYSTORIES_URLS.get(source_split)
    if url is None:
        raise ValueError(f"Unknown source_split={source_split!r}; expected one of {list(TINYSTORIES_URLS)}")

    raw_path = raw_dir / f"tinystories_{source_split}.txt"
    try:
        _download_file(url, raw_path)
    except requests.RequestException as exc:
        logger.warning("Primary download failed (%s); trying synthetic fallback corpus.", exc)
        raw_path = _write_synthetic_corpus(raw_dir / "tinystories_synthetic.txt")

    # For smoke: one small file → carve train/val. For full: train file + separate val download.
    max_train = int(data_cfg.get("max_train_chars") or 0) or None
    max_val = int(data_cfg.get("max_val_chars") or 0) or None

    if source_split == "validation" or data_cfg.get("carve_from_single_file", source_split != "train"):
        text = _read_text_capped(raw_path, None)
        stories = _split_stories(text)
        if len(stories) < 2:
            # Flat text without blank-line separators — split by character budget.
            total_budget = (max_train or len(text)) + (max_val or 0)
            text = text[:total_budget] if total_budget else text
            cut = max_train or int(0.9 * len(text))
            train_text = text[:cut]
            val_text = text[cut : cut + (max_val or len(text) - cut)]
        else:
            n_val = max(1, int(0.1 * len(stories)))
            val_stories = stories[-n_val:]
            train_stories = stories[:-n_val]
            train_text = "\n\n".join(train_stories)
            val_text = "\n\n".join(val_stories)
            if max_train:
                train_text = train_text[:max_train]
            if max_val:
                val_text = val_text[:max_val]
    else:
        train_text = _read_text_capped(raw_path, max_train)
        val_url = TINYSTORIES_URLS["validation"]
        val_raw = raw_dir / "tinystories_validation.txt"
        try:
            _download_file(val_url, val_raw)
            val_text = _read_text_capped(val_raw, max_val)
        except requests.RequestException:
            logger.warning("Val download failed; carving 5%% from train text.")
            cut = int(0.95 * len(train_text))
            val_text = train_text[cut:]
            train_text = train_text[:cut]
            if max_val:
                val_text = val_text[:max_val]

    train_path = processed_dir / data_cfg.get("train_file", "train.txt")
    val_path = processed_dir / data_cfg.get("val_file", "val.txt")
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    logger.info(
        "Wrote train=%s (%d chars), val=%s (%d chars)",
        train_path,
        len(train_text),
        val_path,
        len(val_text),
    )
    return {"train": train_path, "val": val_path, "raw": raw_path}


def _write_synthetic_corpus(dest: Path) -> Path:
    """Offline fallback so the pipeline stays runnable without network."""
    ensure_dir(dest.parent)
    stories = []
    templates = [
        "Once upon a time there was a {adj} {noun}. It loved to {verb} every day. One day it found a friend and they were happy.",
        "Lily was a {adj} girl who liked to {verb}. She saw a {noun} in the garden and smiled. They played until the sun went down.",
        "Tom had a {adj} {noun}. He wanted to {verb} with it. After a while he learned something new and felt proud.",
        "In a small village lived a {adj} {noun}. People came to watch it {verb}. Everyone cheered and went home joyful.",
        "A {adj} cat wanted to {verb} near the {noun}. It tried again and again. Finally it succeeded and purred softly.",
    ]
    nouns = ["dog", "bird", "dragon", "robot", "fox", "bear", "rabbit", "star"]
    adjs = ["little", "brave", "curious", "kind", "clever", "happy", "quiet", "swift"]
    verbs = ["play", "explore", "sing", "draw", "run", "read", "dance", "build"]
    for i in range(400):
        t = templates[i % len(templates)]
        stories.append(
            t.format(
                adj=adjs[i % len(adjs)],
                noun=nouns[i % len(nouns)],
                verb=verbs[i % len(verbs)],
            )
        )
    dest.write_text("\n\n".join(stories), encoding="utf-8")
    logger.info("Wrote synthetic corpus to %s (%d stories)", dest, len(stories))
    return dest


def iter_text_files(paths: list[Path]):
    for path in paths:
        yield path.read_text(encoding="utf-8", errors="replace")
