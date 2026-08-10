"""B6 mandatory controls for the adaptive Lehmer implementation (W8).

B6.1 fixed-beta numerical identity vs THE existing scorer (sharpening_family.score_lehmer);
B6.2 endpoint sanity (beta=0 == mean NLL; the high-beta end converges toward max NLL);
B6.3 constant-gate control (zero weights + the beta=1 bias == fixed Lehmer beta=1);
plus the torch/log-space twin identity and the train-stats-only standardiser.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "checks"))

from luq.adaptive_lehmer import (AdaptiveLehmerGate, BETA_MAX, NLL_FLOOR, ShapeStandardiser,
                                 constant_beta_bias, lehmer_score_np, lehmer_score_torch,
                                 nll_shape_vector)
from sharpening_family import score_lehmer  # the registered scorer — the identity target

RNG = np.random.RandomState(0)
FIXED_BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SAMPLES = [RNG.exponential(1.5, size=t) + 1e-6 for t in (1, 2, 5, 37, 128, 384)]


@pytest.mark.parametrize("beta", FIXED_BETAS)
def test_b61_fixed_beta_identity(beta):
    """The new log-space implementation must reproduce score_lehmer on a deterministic sample."""
    for nll in SAMPLES:
        want = score_lehmer(nll, beta)
        got = lehmer_score_np(nll, beta)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (beta, len(nll))


def test_b61_torch_matches_np():
    for nll in SAMPLES:
        log_l = torch.log(torch.clamp(torch.from_numpy(nll), min=NLL_FLOOR))
        for beta in FIXED_BETAS:
            got = float(lehmer_score_torch(log_l, torch.tensor(beta, dtype=torch.float64)))
            assert got == pytest.approx(lehmer_score_np(nll, beta), rel=1e-6)


def test_b62_endpoints():
    for nll in SAMPLES:
        assert lehmer_score_np(nll, 0.0) == pytest.approx(float(nll.mean()), rel=1e-9)
        assert lehmer_score_np(nll, np.inf) == pytest.approx(float(nll.max()), rel=0)
        # high finite beta converges TOWARD max (not required to equal it)
        gap16 = abs(lehmer_score_np(nll, 16.0) - nll.max())
        gap2 = abs(lehmer_score_np(nll, 2.0) - nll.max())
        assert gap16 <= gap2 + 1e-12


def test_b62_zero_nll_floor_changes_no_ranking():
    """Prob-1.0 tokens (nll = 0) are floored; the floor must not move any score materially."""
    nll = np.array([0.0, 0.5, 2.0, 0.0, 1.0])
    for beta in FIXED_BETAS:
        want = score_lehmer(np.maximum(nll, NLL_FLOOR), beta)
        got = lehmer_score_np(nll, beta)
        assert got == pytest.approx(want, rel=1e-9)


def test_b63_constant_gate_reproduces_fixed_beta1():
    """Zero gate weights + the beta=1 bias: every mode must score EXACTLY fixed Lehmer beta=1."""
    d = 8
    h = torch.randn(4, d, dtype=torch.float32)
    x = torch.randn(4, len(nll_shape_vector(SAMPLES[3]))).float()
    for mode in ("global", "nllshape", "hs", "hybrid"):
        gate = AdaptiveLehmerGate(mode, d_hidden=d)
        with torch.no_grad():
            gate.bias.fill_(constant_beta_bias(1.0))      # weights are zero-initialised already
        beta = gate.beta(h if mode in ("hs", "hybrid") else None,
                         x if mode in ("nllshape", "hybrid") else None, n=4)
        assert beta.shape[0] == 4
        assert torch.allclose(beta, torch.ones_like(beta), atol=1e-6)
        for nll in SAMPLES:
            got = lehmer_score_np(nll, float(beta[0]))
            assert got == pytest.approx(score_lehmer(nll, 1.0), rel=1e-5)


def test_b63_beta_bounds():
    """beta = BETA_MAX * sigmoid(g) is strictly inside (0, BETA_MAX)."""
    gate = AdaptiveLehmerGate("global")
    with torch.no_grad():
        gate.bias.fill_(50.0)
    assert float(gate.beta(None, None)) < BETA_MAX + 1e-6
    with torch.no_grad():
        gate.bias.fill_(-50.0)
    assert float(gate.beta(None, None)) >= 0.0


def test_standardiser_uses_train_stats_only():
    Xtr = RNG.normal(3.0, 2.0, size=(100, 10)).astype(np.float32)
    Xte = RNG.normal(-5.0, 7.0, size=(50, 10)).astype(np.float32)
    s = ShapeStandardiser(Xtr)
    Ztr = s(Xtr)
    assert np.allclose(Ztr.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(Ztr.std(axis=0), 1.0, atol=1e-4)
    # test rows transformed with TRAIN stats — their mean must NOT be ~0
    assert not np.allclose(s(Xte).mean(axis=0), 0.0, atol=0.5)


def test_shape_vector_edges():
    v1 = nll_shape_vector(np.array([0.7]))                # T = 1: defined, no NaN
    assert np.isfinite(v1).all() and v1[0] == pytest.approx(0.0)  # log_T = 0
    v0 = nll_shape_vector(np.zeros(10))                   # all-certain generation
    assert np.isfinite(v0).all()
    v = nll_shape_vector(SAMPLES[4])
    assert v.shape == (10,) and np.isfinite(v).all()
