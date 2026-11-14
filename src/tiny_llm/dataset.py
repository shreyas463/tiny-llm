"""Tokenized dataset and batch sampling for LM training."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_llm.data import split_stories
from tiny_llm.tokenizer import encode, load_tokenizer

logger = logging.getLogger(__name__)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _cache_fingerprint(text_path: Path, tokenizer_dir: Path) -> dict[str, str]:
    tok_path = Path(tokenizer_dir) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    return {
        "text_path": str(text_path.resolve()),
        "text_sha256": _sha256_file(text_path),
        "text_size": str(text_path.stat().st_size),
        "tokenizer_path": str(tok_path.resolve()),
        "tokenizer_sha256": _sha256_file(tok_path),
    }


def _cache_meta_path(cache_path: Path) -> Path:
    return Path(str(cache_path)[: -len(cache_path.suffix)] + ".meta.json")


def _cache_is_valid(cache_path: Path, text_path: Path, tokenizer_dir: Path) -> bool:
    meta_path = _cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return False
    try:
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = _cache_fingerprint(text_path, Path(tokenizer_dir))
    return saved == expected


def tokenize_corpus_to_ids(
    text_path: Path,
    tokenizer_dir: str | Path,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Encode a text file to a flat uint16/uint32 id array (cached).

    Cache hits require matching SHA-256 fingerprints of both the source text and
    ``tokenizer.json``. Changing either (e.g. smoke → tiny prep/tokenizer)
    forces regeneration so GPU runs never reuse stale smoke token IDs.
    """
    tokenizer_dir = Path(tokenizer_dir)
    if cache_path is not None and _cache_is_valid(cache_path, text_path, tokenizer_dir):
        logger.info("Loading cached tokens from %s", cache_path)
        return np.load(cache_path)
    if cache_path is not None and cache_path.exists():
        logger.info(
            "Token cache stale or mismatched (text/tokenizer changed); regenerating %s",
            cache_path,
        )

    tokenizer = load_tokenizer(tokenizer_dir)
    text = text_path.read_text(encoding="utf-8", errors="replace")
    # Split on <|endoftext|> or blank lines so each story gets BOS/EOS.
    stories = split_stories(text)
    if len(stories) < 2:
        ids = encode(tokenizer, text, add_special_tokens=True)
    else:
        ids = []
        for story in stories:
            ids.extend(encode(tokenizer, story, add_special_tokens=True))

    vocab = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab < 65535 else np.uint32
    arr = np.array(ids, dtype=dtype)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
        meta_path = _cache_meta_path(cache_path)
        meta_path.write_text(
            json.dumps(_cache_fingerprint(text_path, tokenizer_dir), indent=2),
            encoding="utf-8",
        )
        logger.info("Cached %d tokens -> %s", len(arr), cache_path)
    return arr


class TokenDataset(Dataset):
    """Sliding windows over a flat token stream."""

    def __init__(self, tokens: np.ndarray, block_size: int) -> None:
        if len(tokens) < block_size + 1:
            raise ValueError(
                f"Need at least block_size+1 tokens; got {len(tokens)} with block_size={block_size}"
            )
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class RandomBatchLoader:
    """Lightweight random-batch sampler (nanoGPT-style) without DataLoader overhead."""

    def __init__(
        self,
        tokens: np.ndarray,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.n = len(tokens) - block_size
        if self.n <= 0:
            raise ValueError("Token stream too short for block_size")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(self.n, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(
                    self.tokens[i : i + self.block_size].astype(np.int64)
                )
                for i in ix.tolist()
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.tokens[i + 1 : i + 1 + self.block_size].astype(np.int64)
                )
                for i in ix.tolist()
            ]
        )
        return x.to(self.device), y.to(self.device)
