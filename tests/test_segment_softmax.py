"""Grounding tests for the sentence-softmax wMSP weighting (segment_mode='softmax').

It must satisfy two EXACT reduction limits, or it is not the method it claims:
  * ONE sentence (whole response) -> plain MSP (all weights 1).
  * every token its OWN sentence  -> plain per-token wMSP (softmax over tokens x n).
Plus the length-invariance property that motivates it: a sentence's total contribution does not depend on
its token count.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luq.weighted_msp import _segment_softmax_weights, _weights_from_raw  # noqa: E402


def test_one_sentence_reduces_to_plain_msp():
    raw = torch.randn(9)
    sid = torch.zeros(9, dtype=torch.long)            # all one segment
    w = _segment_softmax_weights(raw, sid)
    assert torch.allclose(w, torch.ones(9), atol=1e-6), w


def test_every_token_own_sentence_reduces_to_token_wmsp():
    raw = torch.randn(7)
    sid = torch.arange(7, dtype=torch.long)           # each token its own segment
    got = _segment_softmax_weights(raw, sid)
    want = _weights_from_raw(raw, "normalised")       # softmax(raw) * n
    assert torch.allclose(got, want, atol=1e-6), (got, want)


def test_weights_average_one():
    raw = torch.randn(10)
    sid = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
    w = _segment_softmax_weights(raw, sid)
    assert abs(float(w.sum()) - 10.0) < 1e-5           # sum = n => average 1


def test_sentence_total_is_length_invariant():
    """The whole point: two sentences with the SAME learned score contribute the SAME total weight even if
    one is 3x longer. (Under the flat token-softmax path the long one would get ~3x the mass.)"""
    raw = torch.zeros(8)                               # equal score everywhere
    sid = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long)  # sentence 0 has 6 tokens, 1 has 2
    w = _segment_softmax_weights(raw, sid)
    tot0 = float(w[sid == 0].sum()); tot1 = float(w[sid == 1].sum())
    assert abs(tot0 - tot1) < 1e-5, (tot0, tot1)       # equal totals despite 6 vs 2 tokens


def test_keep_mask_excludes_special_tokens():
    """With a keep-mask, excluded (special) tokens get EXACTLY zero weight, and the kept weights still
    average 1 over the KEPT tokens -- matching the flat path's exclude-special behaviour."""
    raw = torch.randn(8)
    sid = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    keep = torch.tensor([1, 1, 0, 1, 1, 1, 1, 0], dtype=torch.float32)   # 2 specials excluded
    w = _segment_softmax_weights(raw, sid, keep=keep)
    assert torch.all(w[keep == 0] == 0.0), w                            # specials -> 0
    assert abs(float(w.sum()) - float(keep.sum())) < 1e-5               # sum = n_kept


def test_keep_none_matches_all_ones_keep():
    raw = torch.randn(9)
    sid = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    a = _segment_softmax_weights(raw, sid, keep=None)
    b = _segment_softmax_weights(raw, sid, keep=torch.ones(9))
    assert torch.allclose(a, b, atol=1e-6)
