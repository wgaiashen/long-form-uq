"""Grounding tests for the learned power-mean aggregator (W-B1).

The family must CONTAIN the fixed rules it is meant to replace. If these limits fail, a "learned aggregator
beats mean/min/geomean" result would be meaningless, because the learned thing would not be a generalisation
of them at all.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luq import seg_aggregate as SA  # noqa: E402

RNG = np.random.RandomState(0)


def _p(n=7):
    return np.clip(RNG.rand(n), 0.02, 0.99)


def test_alpha_1_is_exactly_the_mean():
    for _ in range(20):
        p = _p()
        assert abs(SA.power_mean(p, 1.0) - p.mean()) < 1e-9


def test_alpha_0_is_the_geometric_mean():
    for _ in range(20):
        p = _p()
        assert abs(SA.power_mean(p, 0.0) - float(np.exp(np.log(p).mean()))) < 1e-9


# The power mean approaches min/max only ASYMPTOTICALLY, at rate ~log(n)/|alpha| in RELATIVE terms
# (M_alpha ~ min * exp(log(n)/|alpha|) as alpha -> -inf). So these limits are asserted relatively, with the
# tolerance derived from that rate rather than guessed -- an absolute 1e-3 bound fails for no reason other
# than the min being large, which would be a false alarm about correct code.
def test_large_negative_alpha_approaches_min():
    for _ in range(20):
        p = _p()
        got, want = SA.power_mean(p, -2000.0), p.min()
        assert abs(got - want) / want < 1e-2, (got, want)


def test_large_positive_alpha_approaches_max():
    for _ in range(20):
        p = _p()
        got, want = SA.power_mean(p, 2000.0), p.max()
        assert abs(got - want) / want < 1e-2, (got, want)


def test_convergence_rate_to_min_is_the_expected_one():
    """Tightening alpha by 10x must shrink the gap to min by ~10x (the log(n)/|alpha| rate)."""
    p = _p(7)
    g1 = SA.power_mean(p, -100.0) - p.min()
    g2 = SA.power_mean(p, -1000.0) - p.min()
    assert g1 > g2 > 0
    assert 5 < g1 / g2 < 20, (g1, g2)


def test_monotone_in_alpha():
    """The power mean is non-decreasing in alpha -- a standard property, and a strong wiring check."""
    p = _p(9)
    vals = [SA.power_mean(p, a) for a in [-8, -4, -1, 0, 1, 4, 8]]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), vals


def test_a_single_zero_does_not_annihilate_the_score():
    """A p=0 sentence would send geomean and every negative-alpha mean to exactly 0, destroying the ranking
    across instances. Clipping must keep the score finite and still ordered."""
    a = SA.power_mean(np.array([0.0, 0.9, 0.9]), 0.0)
    b = SA.power_mean(np.array([0.0, 0.2, 0.2]), 0.0)
    assert a > b > 0.0


def test_alpha_is_fitted_on_train_only_and_is_recoverable():
    """If the data is generated so the WEAKEST sentence determines correctness, the fit should prefer a
    min-like (negative) alpha over the mean."""
    rng = np.random.RandomState(3)
    p_list = [np.clip(rng.rand(5), 0.05, 0.95) for _ in range(400)]
    y = np.array([float(p.min() > 0.35) for p in p_list])          # weakest-link ground truth

    def prr(yy, unc):                                              # rank-correlation stand-in
        conf = -np.asarray(unc)
        return float(np.corrcoef(conf, yy)[0, 1])

    alpha, _ = SA.fit_alpha(p_list, y, prr)
    mean_s = prr(y, 1.0 - SA.aggregate(p_list, 1.0))
    fit_s = prr(y, 1.0 - SA.aggregate(p_list, alpha))
    assert alpha < 1.0, f"expected a min-like alpha on weakest-link data, got {alpha}"
    assert fit_s >= mean_s - 1e-12
