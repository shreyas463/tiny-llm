"""LR schedule helpers."""

from tiny_llm.train import _lr_at_step


def test_warmup_and_cosine_floor():
    lr = 3e-4
    assert abs(_lr_at_step(0, warmup=100, max_steps=1000, lr=lr) - lr / 100) < 1e-12
    assert abs(_lr_at_step(99, warmup=100, max_steps=1000, lr=lr) - lr) < 1e-12
    # End of schedule ≈ min_lr_ratio * lr
    end = _lr_at_step(999, warmup=100, max_steps=1000, lr=lr, min_lr_ratio=0.05)
    assert abs(end - lr * 0.05) < 1e-8
    assert _lr_at_step(1000, warmup=100, max_steps=1000, lr=lr, min_lr_ratio=0.05) == lr * 0.05
