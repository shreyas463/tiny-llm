"""Smoke unit tests that don't need downloaded data."""

import torch

from tiny_llm.model import GPT, GPTConfig


def test_gpt_forward_and_generate():
    cfg = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    out = model.generate(x[:1, :4], max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape[1] == 4 + 5


def test_param_count_positive():
    cfg = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    assert model.num_parameters() > 0
