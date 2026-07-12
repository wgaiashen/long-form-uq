"""Grounding tests for src/luq/weighting.py — the shared token-weighting primitives.

Pins the invariants both tracks rely on: the pool broadcasts, every regulariser is 0 exactly at the
uniform floor and positive off it, the transforms behave at their limits, and the floor-reduction check
works. Pure CPU, milliseconds.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import weighting as wt  # noqa: E402


def test_pool_broadcasts_scalar_and_feature():
    w = torch.tensor([1.0, 1.0, 1.0])           # uniform avg-1
    x1 = torch.tensor([2.0, 4.0, 6.0])          # [T] NLLs
    assert torch.allclose(wt.pool(x1, w), x1.sum())          # Σ 1·x = sum
    xd = torch.arange(9.0).reshape(3, 3)         # [T, d] hidden states
    assert torch.allclose(wt.pool(xd, w), xd.sum(dim=0))     # broadcast over feature dim


def test_normalize_avg1_sums_to_n():
    raw = torch.tensor([0.3, -1.0, 2.0, 0.5])
    w = wt.normalize_avg1(raw)
    assert torch.allclose(w.sum(), torch.tensor(4.0), atol=1e-5)   # averages to 1
    assert (w > 0).all()


def test_regularisers_zero_at_uniform_positive_off():
    uni = torch.ones(5)
    peaky = wt.normalize_avg1(torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0]))
    for name, fn in wt.REGULARISERS.items():
        assert abs(fn(uni).item()) < 1e-6, f"{name} not ~0 at uniform"
        assert fn(peaky).item() > 0, f"{name} not >0 for a peaky w"


def test_shrink_is_msd_from_one():
    w = torch.tensor([0.0, 2.0, 1.0, 1.0])       # deviations 1,1,0,0 -> mean sq = 0.5
    assert torch.allclose(wt.shrink_to_uniform(w), torch.tensor(0.5))


def test_smooth_raw_flattens_and_preserves_length():
    raw = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0])
    sm = wt.smooth_raw(raw, 3)
    assert sm.shape == raw.shape                              # length preserved
    assert sm.var() < raw.var()                               # variance reduced (spike spread out)
    const = torch.full((6,), 2.0)
    assert torch.allclose(wt.smooth_raw(const, 3), const)     # constant is a fixed point


def test_beta_sharpen_limits():
    s = torch.tensor([0.1, 0.2, 0.7])
    assert torch.allclose(wt.beta_sharpen(s, 0.0), torch.ones(3))   # beta=0 -> uniform floor
    w1 = wt.beta_sharpen(s, 1.0)
    assert torch.allclose(w1.sum(), torch.tensor(3.0), atol=1e-5)   # avg-1
    # larger beta concentrates more mass on the largest source score
    w2 = wt.beta_sharpen(s, 3.0)
    assert w2.max() > w1.max()


def test_reduces_to_uniform():
    assert wt.reduces_to_uniform(torch.ones(4))
    assert not wt.reduces_to_uniform(torch.tensor([0.5, 1.5, 1.0, 1.0]))
