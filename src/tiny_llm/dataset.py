"""Tokenized dataset and batch sampling for LM training."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_llm.tokenizer import encode, load_tokenizer

logger = logging.getLogger(__name__)


def tokenize_corpus_to_ids(
    text_path: Path,
    tokenizer_dir: str | Path,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Encode a text file to a flat uint16/uint32 id array (cached)."""
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached tokens from %s", cache_path)
        return np.load(cache_path)

    tokenizer = load_tokenizer(tokenizer_dir)
    text = text_path.read_text(encoding="utf-8", errors="replace")
    # Encode story-by-story when blank-line separated to keep BOS/EOS boundaries.
    stories = [s.strip() for s in text.split("\n\n") if s.strip()]
    if len(stories) < 2:
        ids = encode(tokenizer, text, add_special_tokens=True)
    else:
        ids: list[int] = []
        for story in stories:
            ids.extend(encode(tokenizer, story, add_special_tokens=True))

    vocab = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab < 65535 else np.uint32
    arr = np.array(ids, dtype=dtype)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
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
