"""Grounding tests for S6 multi-head attention (Joe idea 2). The gating check is the LIMIT CASE: the
generalisation must not change the incumbent single-head pooler (arm A), and the multi-head code path must
reduce to a single head when its heads are made identical.

  - K=1 default unchanged        : train_attn(n_query=1,n_head=1) == train_attn() (the single-head fast path
                                   is byte-identical; adding the n_query/n_head params perturbs nothing).
  - MH path reduces to single    : an MH pooler with all queries/heads REPLICATED from a single head gives
                                   ensemble predictions identical to that single head, <1e-6 (proves the MH
                                   einsum/stack/mean-of-sigmoids is a correct generalisation).
  - shapes                        : MH -> logits (B,K), attention (B,T,K); ABLATION (1 query, K heads) ->
                                   logits (B,K), attention (B,T,1).
  - symmetry-broken               : the extra queries init NON-zero, so the heads CAN diverge (else all-zero
                                   queries share a gradient and collapse by construction).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from attn_pool import AttnPool, pad_batch, train_attn                       # noqa: E402
from aggregation_table import attn_unc                                      # noqa: E402

D, T, B = 8, 6, 5
TOL = 1e-6


def _states(seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(T, D).astype(np.float32) for _ in range(B)]


def test_k1_default_unchanged():
    """n_query=n_head=1 must equal the plain single-head pooler (same seed) — the params perturb nothing."""
    states = _states(); y = np.array([0., 1., 0., 1., 0.])
    tr = [0, 1, 2, 3, 4]
    a = train_attn(states, y, tr, "cpu", seed=0)                       # default
    b = train_attn(states, y, tr, "cpu", seed=0, n_query=1, n_head=1)  # explicit K=1
    ua = attn_unc(a, states, tr, "cpu"); ub = attn_unc(b, states, tr, "cpu")
    assert np.allclose(ua, ub, atol=TOL)


def test_mh_path_reduces_to_single():
    """The MH path with REPLICATED queries+heads == the single-head pooler (ensemble of identical = itself)."""
    states = _states(2)
    torch.manual_seed(0); single = AttnPool(D, n_query=1, n_head=1)
    torch.manual_seed(0); mh = AttnPool(D, n_query=2, n_head=2)        # same seed -> primary q/head match
    with torch.no_grad():
        single.q.copy_(torch.randn(D) * 0.1)                          # a non-uniform query (exercise softmax)
        mh.q.copy_(single.q); mh.q_rest[0].copy_(single.q)            # both queries identical
        for h in (mh.head, mh.heads_rest[0]):                        # both heads identical to single's head
            h.weight.copy_(single.head.weight); h.bias.copy_(single.head.bias)
    X, mask, pos = pad_batch(states, "cpu")
    with torch.no_grad():
        ls, a_s = single(X, mask, pos)          # (B,), (B,T)
        lm, a_m = mh(X, mask, pos)              # (B,2), (B,T,2)
    p_single = torch.sigmoid(ls)
    p_mh = torch.sigmoid(lm).mean(dim=1)        # ensemble = mean-of-sigmoids
    assert p_single.shape == (B,) and lm.shape == (B, 2) and a_m.shape == (B, T, 2)
    assert torch.allclose(p_single, p_mh, atol=TOL)


def test_shapes_mh_and_ablation():
    states = _states(3)
    X, mask, pos = pad_batch(states, "cpu")
    mh = AttnPool(D, n_query=4, n_head=4)
    abl = AttnPool(D, n_query=1, n_head=4)      # ABLATION: one attention, 4 heads
    with torch.no_grad():
        lm, am = mh(X, mask, pos); la, aa = abl(X, mask, pos)
    assert lm.shape == (B, 4) and am.shape == (B, T, 4)   # MH: 4 attention distributions
    assert la.shape == (B, 4) and aa.shape == (B, T, 1)   # ABLATION: ONE attention, 4 heads


def test_extra_queries_break_symmetry():
    """Extra queries must init distinct (non-zero) so the heads can diverge — else collapse by construction."""
    torch.manual_seed(0); mh = AttnPool(D, n_query=4, n_head=4)
    q_all = torch.cat([mh.q.unsqueeze(0), mh.q_rest], 0)   # (4, D)
    # the 3 extra queries are not all-zero and not all-identical to each other
    assert mh.q_rest.abs().sum() > 0
    assert not torch.allclose(mh.q_rest[0], mh.q_rest[1], atol=TOL)
