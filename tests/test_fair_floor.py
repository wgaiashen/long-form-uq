"""Guard tests for luq.msp.fair_floor — 15 drivers now depend on it.

The bug this exists to prevent: a floor that is chosen conveniently rather than honestly. Comparing a
supervised method against `msp_sum` alone overstated every margin, because msp_sum is not
length-normalised and is the WEAKEST of the three standard floors on all 9 of our datasets.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luq import msp, results  # noqa: E402


def _recs(rng, n=200):
    return [{"token_logprobs": list(-np.abs(rng.randn(rng.randint(5, 40))))} for _ in range(n)]


def test_picks_the_strictly_best_floor():
    rng = np.random.RandomState(0)
    recs = _recs(rng)
    y = rng.rand(len(recs))
    vec, name = msp.fair_floor(recs, y, results.prr)
    ind = {a: results.prr(y, np.array([msp.msp_uncertainty(r["token_logprobs"], a) for r in recs]))
           for a in msp.FLOOR_AGGREGATES}
    assert name == max(ind, key=ind.get)
    assert abs(results.prr(y, vec) - max(ind.values())) < 1e-12


def test_never_weaker_than_bare_msp_sum():
    """The whole point: switching to the fair floor can only RAISE the bar, never lower it."""
    rng = np.random.RandomState(1)
    for _ in range(5):
        recs = _recs(rng)
        y = rng.rand(len(recs))
        vec, _ = msp.fair_floor(recs, y, results.prr)
        bare = results.prr(y, np.array([msp.msp_uncertainty(r["token_logprobs"], "sum") for r in recs]))
        assert results.prr(y, vec) >= bare - 1e-12


def test_returns_the_winners_vector_not_a_recomputation():
    """The returned vector must BE the winning floor's, so a paired bootstrap tests the real bar.
    (The original incomplete fix updated the reported PRR but left the bootstrap on the old vector.)"""
    rng = np.random.RandomState(2)
    recs = _recs(rng)
    y = rng.rand(len(recs))
    vec, name = msp.fair_floor(recs, y, results.prr)
    expected = np.array([msp.msp_uncertainty(r["token_logprobs"], name) for r in recs])
    assert np.allclose(vec, expected)
