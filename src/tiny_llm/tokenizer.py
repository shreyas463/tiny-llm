"""Train and load a BPE tokenizer with HuggingFace tokenizers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from tiny_llm.config import ensure_dir

logger = logging.getLogger(__name__)

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"


def train_bpe_tokenizer(
    train_path: Path,
    artifact_dir: str | Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on a text file and save artifacts."""
    special_tokens = special_tokens or [PAD, UNK, BOS, EOS]
    artifact_dir = ensure_dir(artifact_dir)

    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    logger.info("Training BPE tokenizer (vocab_size=%d) on %s", vocab_size, train_path)
    tokenizer.train([str(train_path)], trainer)

    bos_id = tokenizer.token_to_id(BOS)
    eos_id = tokenizer.token_to_id(EOS)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS} $A {EOS}",
        special_tokens=[(BOS, bos_id), (EOS, eos_id)],
    )

    tok_path = artifact_dir / "tokenizer.json"
    tokenizer.save(str(tok_path))

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {
            "pad": tokenizer.token_to_id(PAD),
            "unk": tokenizer.token_to_id(UNK),
            "bos": bos_id,
            "eos": eos_id,
        },
        "train_file": str(train_path),
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved tokenizer to %s (vocab=%d)", tok_path, meta["vocab_size"])
    return tokenizer


def load_tokenizer(artifact_dir: str | Path) -> Tokenizer:
    artifact_dir = Path(artifact_dir)
    tok_path = artifact_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}; run tokenizer training first.")
    return Tokenizer.from_file(str(tok_path))


def load_tokenizer_meta(artifact_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(artifact_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Tokenizer meta not found at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def encode(tokenizer: Tokenizer, text: str, add_special_tokens: bool = True) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def decode(tokenizer: Tokenizer, ids: list[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def train_from_config(cfg: dict[str, Any]) -> Tokenizer:
    data_cfg = cfg["data"]
    tok_cfg = cfg["tokenizer"]
    train_path = Path(data_cfg["processed_dir"]) / data_cfg.get("train_file", "train.txt")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}; run prepare first.")
    return train_bpe_tokenizer(
        train_path=train_path,
        artifact_dir=tok_cfg["artifact_dir"],
        vocab_size=int(tok_cfg["vocab_size"]),
        min_frequency=int(tok_cfg.get("min_frequency", 2)),
        special_tokens=list(tok_cfg.get("special_tokens", [PAD, UNK, BOS, EOS])),
    )
