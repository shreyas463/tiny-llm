"""Tests for token-cache invalidation and story-boundary splitting."""

from pathlib import Path

import numpy as np

from tiny_llm.data import normalize_stories_text, split_stories
from tiny_llm.dataset import tokenize_corpus_to_ids
from tiny_llm.tokenizer import train_bpe_tokenizer


def test_split_stories_endoftext():
    text = "Story one.\n\nMore.\n<|endoftext|>\nStory two.<|endoftext|>"
    stories = split_stories(text)
    assert stories == ["Story one.\n\nMore.", "Story two."]
    assert "<|endoftext|>" not in normalize_stories_text(text)


def test_token_cache_invalidates_on_text_or_tokenizer_change(tmp_path: Path):
    text_path = tmp_path / "train.txt"
    text_path.write_text("Once upon a time there was a cat.\n\nIt liked to play.", encoding="utf-8")
    tok_dir = tmp_path / "tok"
    train_bpe_tokenizer(text_path, tok_dir, vocab_size=200, min_frequency=1)
    cache = tmp_path / "train_ids.npy"

    ids1 = tokenize_corpus_to_ids(text_path, tok_dir, cache_path=cache)
    assert cache.exists()
    assert cache.with_name("train_ids.meta.json").exists() or Path(
        str(cache)[: -len(".npy")] + ".meta.json"
    ).exists()

    # Cache hit
    ids_hit = tokenize_corpus_to_ids(text_path, tok_dir, cache_path=cache)
    assert np.array_equal(ids1, ids_hit)

    # Text change → regenerate
    text_path.write_text(
        "Once upon a time there was a dog.\n\nIt liked to run far away today.",
        encoding="utf-8",
    )
    ids2 = tokenize_corpus_to_ids(text_path, tok_dir, cache_path=cache)
    assert not np.array_equal(ids1, ids2)

    # Tokenizer change → regenerate (new vocab size)
    tok_dir2 = tmp_path / "tok2"
    train_bpe_tokenizer(text_path, tok_dir2, vocab_size=250, min_frequency=1)
    ids3 = tokenize_corpus_to_ids(text_path, tok_dir2, cache_path=cache)
    # Different tokenizer → different encoding in general; at least cache meta updated
    meta = Path(str(cache)[: -len(".npy")] + ".meta.json")
    assert "tok2" in meta.read_text(encoding="utf-8")
    assert len(ids3) > 0
