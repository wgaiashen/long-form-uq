"""CPU unit tests for the pipeline's pure logic (run with `pytest`).

No GPU, no cache: these pin the maths the reported numbers depend on, so a refactor can't
silently change a result. The GPU verification checks (lookback vs the authors' code,
greedy-generation determinism) stay as manual scripts in scripts/checks/ because they need
the model loaded — see README.md.
"""
import numpy as np

from luq import msp, results
from luq.labels.llm_judge import parse_score


def test_perplexity_is_mean_negative_log_likelihood():
    # The "perplexity" aggregate == -mean(logprob), which is exactly lm-polygraph's Perplexity
    # estimator (`-np.mean(ll)`; uhead inherits it; Joe's "Perplexity" baseline). Confirmed equal
    # to the installed source by inspection; pinned here so the equivalence can't drift. (Name
    # follows lm-polygraph: it returns mean NLL, not exp(mean NLL) — PRR ranking is identical.)
    lp = [-0.1, -2.0, -0.5, -1.2]
    assert np.isclose(msp.msp_uncertainty(lp, "perplexity"), -np.mean(lp))


def test_msp_aggregates():
    lp = [-0.1, -2.0, -0.5]
    p = np.exp(lp)
    assert np.isclose(msp.msp_uncertainty(lp, "mean"), 1 - p.mean())
    assert np.isclose(msp.msp_uncertainty(lp, "min"), 1 - p.min())
    assert np.isclose(msp.msp_uncertainty(lp, "sum"), -np.sum(lp))
    assert np.isclose(msp.msp_uncertainty(lp, "perplexity"), -np.mean(lp))


def test_prr_oracle_is_one_and_anti_oracle_is_negative():
    rng = np.random.default_rng(0)
    correctness = rng.random(200)
    # Oracle uncertainty = -correctness (reject the truly-worst first) -> PRR == 1.
    assert np.isclose(results.prr(correctness, -correctness), 1.0, atol=1e-9)
    # Uncertainty == correctness (reject the BEST first) -> worse than random -> PRR < 0.
    assert results.prr(correctness, correctness) < 0


def test_prr_perfect_binary_separation():
    # Two cleanly separated groups, uncertainty aligned with wrongness -> PRR == 1.
    correctness = np.array([1.0, 1.0, 0.0, 0.0])
    uncertainty = np.array([0.0, 0.1, 0.9, 1.0])
    assert np.isclose(results.prr(correctness, uncertainty), 1.0, atol=1e-9)


def test_parse_score_clean_replies():
    assert parse_score("0.8") == 0.8
    assert parse_score(" 1.0\n") == 1.0
    assert parse_score("0") == 0.0
    assert parse_score("nonsense") is None
    assert parse_score(None) is None


def test_parse_score_known_fragility():
    # Documented limitation: a leading list-number is read as the score. This is why the
    # local judge wants a model that emits a bare number (the GPT-5 path uses strict parsing).
    assert parse_score("1. the answer is correct") == 1.0
