"""Grounding tests for the W4 hierarchical (two-level) pooler.

The project convention is that every aggregation variant carries a NUMERIC correctness check that would
catch a silent bug where the method stops being what it claims (the same role `constant == plain MSP` plays
for weighted-MSP). For a two-level pooler the natural checks are its two degenerate limits, both of which
must reproduce the FLAT attention pooler exactly:

    all tokens in ONE segment      -> level 2 is a softmax over one item -> flat pooler
    every token its OWN segment    -> level 1 is a softmax over one item -> flat pooler

If either fails, the tokens<->segments wiring (the scatter indices) is wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq.hier_pool import HierPool, pad_batch_seg, segment_softmax  # noqa: E402
from attn_pool import AttnPool, pad_batch  # noqa: E402

D, T, B = 8, 6, 3
TOL = 1e-6


def _states(seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(T, D).astype(np.float32) for _ in range(B)]


def _flat_logit(states, q, head_w, head_b):
    """Reference: the FLAT attention pooler with a given query + head."""
    flat = AttnPool(D)
    with torch.no_grad():
        flat.q.copy_(q); flat.head.weight.copy_(head_w); flat.head.bias.copy_(head_b)
    X, mask, pos = pad_batch(states, "cpu")
    with torch.no_grad():
        logit, _ = flat(X, mask, pos)
    return logit


def _hier_logit(states, seg_list, q_tok, q_seg, head_w, head_b):
    h = HierPool(D)
    with torch.no_grad():
        h.q_tok.copy_(q_tok); h.q_seg.copy_(q_seg)
        h.head.weight.copy_(head_w); h.head.bias.copy_(head_b)
    X, mask, seg, n_seg = pad_batch_seg(states, seg_list, "cpu")
    with torch.no_grad():
        logit, _, _ = h(X, mask, seg, n_seg)
    return logit


def test_one_segment_reduces_to_flat_pooler():
    """All tokens in a single segment: level-1 attention IS the flat attention, level 2 is a no-op."""
    states = _states(0)
    q = torch.randn(D); hw = torch.randn(1, D); hb = torch.randn(1)
    seg = [np.zeros(T, dtype=np.int64) for _ in range(B)]          # one segment
    # q_seg is irrelevant here (a softmax over one segment is 1.0 whatever the score)
    got = _hier_logit(states, seg, q, torch.randn(D), hw, hb)
    want = _flat_logit(states, q, hw, hb)
    assert torch.max(torch.abs(got - want)).item() < TOL, (got, want)


def test_each_token_own_segment_reduces_to_flat_pooler():
    """Every token its own segment: level-1 is a no-op, level-2 attention IS the flat attention."""
    states = _states(1)
    q = torch.randn(D); hw = torch.randn(1, D); hb = torch.randn(1)
    seg = [np.arange(T, dtype=np.int64) for _ in range(B)]          # T segments of 1 token
    # q_tok is irrelevant here (a softmax over one token is 1.0 whatever the score)
    got = _hier_logit(states, seg, torch.randn(D), q, hw, hb)
    want = _flat_logit(states, q, hw, hb)
    assert torch.max(torch.abs(got - want)).item() < TOL, (got, want)


def test_segment_softmax_sums_to_one_within_each_segment():
    """The level-1 weights must form a distribution PER SEGMENT (not per response)."""
    rng = np.random.RandomState(2)
    scores = torch.tensor(rng.randn(2, 6), dtype=torch.float32)
    mask = torch.ones(2, 6)
    mask[1, 4:] = 0.0                                              # example 1 is padded to length 4
    seg = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 0, 0]], dtype=torch.long)
    a, _ = segment_softmax(scores, mask, seg, n_seg=3)
    for b in range(2):
        for s in range(3):
            sel = (seg[b] == s) & (mask[b] > 0)
            if sel.any():
                assert abs(a[b][sel].sum().item() - 1.0) < 1e-5
    # padded positions must carry zero weight
    assert a[1, 4:].abs().max().item() < 1e-9


def test_padding_does_not_change_the_score():
    """A batch containing a SHORT example alongside long ones must score that example identically to
    scoring it alone -- i.e. the padding is genuinely masked out of both levels."""
    rng = np.random.RandomState(3)
    short = rng.randn(3, D).astype(np.float32)
    long_ = rng.randn(9, D).astype(np.float32)
    q_tok, q_seg = torch.randn(D), torch.randn(D)
    hw, hb = torch.randn(1, D), torch.randn(1)
    seg_short = [np.array([0, 0, 1], dtype=np.int64)]
    seg_long = [np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)]
    alone = _hier_logit([short], seg_short, q_tok, q_seg, hw, hb)
    together = _hier_logit([short, long_], seg_short + seg_long, q_tok, q_seg, hw, hb)
    assert abs(alone[0].item() - together[0].item()) < 1e-5
