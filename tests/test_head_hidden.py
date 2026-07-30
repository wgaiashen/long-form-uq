"""Grounding tests for the `head_hidden` knob on AttnPool (the 2x2 head axis, head_aggregation_2x2.py).

The gating invariant is the LIMIT CASE, exactly as for S6: adding `head_hidden` must not perturb the
incumbent linear-head pooler (arms A/B), and a `head_hidden` tuple must build the SAME MLP that SAPLMA
uses (luq.probe.train_probe_mlp), so the head axis is a pure head swap under one training recipe.

  - head_hidden=None byte-identical : train_attn() == train_attn(head_hidden=None) AND _make_head(None)
                                      draws RNG identically to a bare nn.Linear(d,1) (draw order unchanged).
  - None head is Linear             : the primary head is nn.Linear(d,1), shape (1,d).
  - tuple builds the SAPLMA stack   : head_hidden=(256,128,64) -> Linear->ReLU->...->Linear(.,1) with the
                                      SAME layer shapes probe.train_probe_mlp builds.
  - MLP head params are decayed      : train_attn's param-group split routes every MLP head param to the
                                      wd group; only the query (q / q_rest) stays un-decayed.
  - MLP head trains + scores         : train_attn(head_hidden=...) runs and returns finite uncertainties.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from attn_pool import AttnPool, _make_head, train_attn, WEIGHT_DECAY  # noqa: E402
from aggregation_table import attn_unc, conf_meanpool                 # noqa: E402

D, T, B = 16, 6, 5
TOL = 1e-6


def _states(seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(T, D).astype(np.float32) for _ in range(B)]


def test_make_head_none_rng_identical():
    """_make_head(d, None) must consume RNG exactly like a bare nn.Linear(d,1) -- so the None path is
    byte-identical to the pre-`head_hidden` pooler (same draw order => same init everywhere downstream)."""
    torch.manual_seed(0); a = _make_head(D, None)
    torch.manual_seed(0); b = nn.Linear(D, 1)
    assert isinstance(a, nn.Linear) and a.weight.shape == (1, D)
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)


def test_head_hidden_none_unchanged():
    """Passing head_hidden=None must equal the default call at the same seed -- the new param perturbs nothing."""
    states = _states(); y = np.array([0., 1., 0., 1., 0.]); tr = [0, 1, 2, 3, 4]
    a = train_attn(states, y, tr, "cpu", seed=0)                    # default (no head_hidden)
    b = train_attn(states, y, tr, "cpu", seed=0, head_hidden=None)  # explicit None
    ua = attn_unc(a, states, tr, "cpu"); ub = attn_unc(b, states, tr, "cpu")
    assert np.allclose(ua, ub, atol=TOL)


def test_head_hidden_builds_saplma_arch():
    """head_hidden=(256,128,64) builds exactly probe.train_probe_mlp's stack: Linear->ReLU->..->Linear(.,1)."""
    hidden = (256, 128, 64)
    head = _make_head(D, hidden)
    assert isinstance(head, nn.Sequential)
    # Reference: the same construction loop luq.probe.train_probe_mlp uses.
    ref, prev = [], D
    for h in hidden:
        ref += [("linear", prev, h), ("relu",)]
        prev = h
    ref += [("linear", prev, 1)]
    got = []
    for m in head:
        if isinstance(m, nn.Linear):
            got.append(("linear", m.in_features, m.out_features))
        elif isinstance(m, nn.ReLU):
            got.append(("relu",))
        else:
            got.append((type(m).__name__,))
    assert got == ref


def test_mlp_head_params_all_decayed():
    """The param-group split (name not starting 'q' -> wd group) must place EVERY MLP head param in the
    decayed group and leave only the query un-decayed -- so the 2x2 MLP cells use the locked recipe
    (wd=1e-2 on the head, wd_query=0)."""
    m = AttnPool(D, head_hidden=(256, 128, 64))
    head_names = [n for n, _ in m.named_parameters() if not n.startswith("q")]
    query_names = [n for n, _ in m.named_parameters() if n.startswith("q")]
    assert query_names == ["q"]                       # only the query is un-decayed
    assert all(n.startswith("head") for n in head_names)
    assert len(head_names) == 8                        # 4 Linear layers x (weight + bias)
    assert WEIGHT_DECAY == 1e-2                        # the locked head weight-decay


def test_train_attn_mlp_runs_and_scores():
    """train_attn with an MLP head runs end-to-end and returns finite, in-range uncertainties."""
    states = _states(3); y = np.array([0., 1., 0., 1., 1.]); tr = [0, 1, 2, 3, 4]
    m = train_attn(states, y, tr, "cpu", seed=0, temperature=1.0, head_hidden=(8, 4), epochs=3)
    assert isinstance(m.head, nn.Sequential)
    u = np.asarray(attn_unc(m, states, tr, "cpu"), float)
    assert u.shape == (B,) and np.isfinite(u).all() and (u >= 0).all() and (u <= 1).all()


def test_meanpool_mlp_5ep_equals_saplma():
    """The 2x2 clean gate: a FROZEN-query pooler with the SAPLMA MLP head at 5 epochs / wd=0 must
    reproduce SAPLMA (conf_meanpool -> train_probe_mlp) -- uniform softmax over real tokens == mean-pool,
    the MLP init RNG draws first in both, same generator-seeded batches, same Adam(wd=0). If this drifts,
    the MLP-head path is not the SAPLMA head and the whole bridge is invalid."""
    states = [np.random.RandomState(k).randn(T, D).astype(np.float32) for k in range(24)]
    y = np.array([float(k % 2) for k in range(24)])
    tr = list(range(16)); te = list(range(16, 24)); seed = 1
    Xmean = np.stack([s.mean(axis=0) for s in states])
    saplma_unc = 1.0 - conf_meanpool(Xmean, tr, te, y, seed)                       # the reference
    m = train_attn(states, y, tr, "cpu", seed=seed, freeze_query=True,
                   head_hidden=(256, 128, 64), epochs=5, weight_decay=0.0)          # the 2x2 5ep/wd0 cell
    mlp_unc = np.asarray(attn_unc(m, states, te, "cpu"), float)
    assert np.allclose(saplma_unc, mlp_unc, atol=1e-4), f"max|Δ|={np.abs(saplma_unc - mlp_unc).max():.2e}"
