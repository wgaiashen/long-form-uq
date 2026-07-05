"""Unit tests for the weighted-MSP method (src/luq/weighted_msp.py).

These pin the two things that must not silently break: the soft-rank loss components (ported from
Joe's msp_probe_uq.py) and the grounding property that `constant` mode reduces to plain MSP.
Pure-CPU, no cache, no GPU -- runs in milliseconds.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import weighted_msp as wm  # noqa: E402


def test_true_rank_matches_argsort_order():
    # Higher incorrectness -> higher rank (1-indexed).
    inc = torch.tensor([0.2, 0.9, 0.5])
    ranks = wm._true_rank(inc)
    # 0.2 is smallest -> rank 1; 0.5 -> rank 2; 0.9 -> rank 3.
    assert ranks.tolist() == [1.0, 3.0, 2.0]


def test_soft_rank_is_monotonic_in_q():
    # A strictly increasing q should give strictly increasing soft ranks.
    q = torch.tensor([-2.0, 0.0, 1.0, 5.0])
    sr = wm._soft_rank(q)
    assert torch.all(sr[1:] > sr[:-1]), sr
    # Soft ranks live in [1, n]; the largest element is close to n, the smallest close to 1.
    assert sr.min() >= 1.0 and sr.max() <= len(q)


def test_soft_rank_symmetry_center():
    # Two equal values sit at the same soft rank, symmetric about the middle.
    q = torch.tensor([0.0, 0.0])
    sr = wm._soft_rank(q)
    assert torch.allclose(sr, torch.tensor([1.5, 1.5]))


def test_weights_from_raw_modes():
    raw = torch.tensor([1.0, 2.0, 3.0])
    # constant -> all ones (plain MSP).
    assert torch.allclose(wm._weights_from_raw(raw, "constant"), torch.ones(3))
    # normalised -> softmax * n, so the weights average to 1 (sum to n).
    w = wm._weights_from_raw(raw, "normalised")
    assert torch.allclose(w.sum(), torch.tensor(3.0))
    # unconstrained -> passthrough.
    assert torch.allclose(wm._weights_from_raw(raw, "unconstrained"), raw)


def test_constant_seq_q_equals_msp():
    # For a synthetic record, constant-mode q must equal plain MSP (sum) and, length-normalised,
    # the perplexity (mean NLL). This mirrors constant_equals_msp_maxdiff on real data.
    rng = np.random.RandomState(0)
    logprobs = (-rng.rand(17)).tolist()  # 17 generated tokens, negative logprobs
    record = {"token_logprobs": logprobs}
    nll = torch.from_numpy(wm.per_token_nll(record))
    raw = torch.zeros_like(nll)  # ignored in constant mode

    q_sum = wm._seq_q(raw, nll, "constant", length_normalise=False).item()
    q_mean = wm._seq_q(raw, nll, "constant", length_normalise=True).item()

    from luq import msp
    assert abs(q_sum - msp.msp_uncertainty(logprobs, "sum")) < 1e-6
    assert abs(q_mean - msp.msp_uncertainty(logprobs, "perplexity")) < 1e-6


def test_answer_states_drops_row0():
    # The per-token window is [last_prompt_token] + gen_tokens (G+1 rows); answer_states drops row 0
    # so the G rows align 1:1 with the G NLL values.
    state = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)  # 5 rows (G+1), hidden 3
    ans = wm.answer_states(state)
    assert ans.shape == (4, 3)
    assert np.array_equal(ans, state[1:])
