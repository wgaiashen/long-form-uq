"""MSP: an unsupervised token-probability uncertainty score. Comes free.

We already capture per-token logprobs in the Tier-1 record at generation time, so
MSP costs nothing extra. Sequence confidence is an aggregate of the per-token
probabilities; uncertainty = 1 - confidence (higher = more uncertain).
"""
import numpy as np


def msp_uncertainty(token_logprobs, aggregate: str = "mean") -> float:
    """token_logprobs: list of log p(chosen token). Returns uncertainty.

    "mean": 1 - mean token probability (smooth, whole-sequence view).
    "min" : 1 - least-confident token's probability (weakest-link view).
    "sum" : minus the sum of logprobs = -log p(sequence). This is what
            lm-polygraph calls MaximumSequenceProbability (and what Joe's Hidden
            Failures reports as "MSP"), so use it when comparing against those
            numbers. Not length-normalised, and not bounded to [0, 1] like the
            others (fine for PRR, which only uses the ranking).
    "perplexity" : mean per-token negative log-likelihood = (1/L) * sum(-log p),
            the LENGTH-NORMALISED counterpart to "sum". This is exactly
            lm-polygraph's `Perplexity` estimator (`-np.mean(ll)`; uhead inherits
            it) and the "Perplexity" baseline in Joe's Hidden Failures, so the
            name matches them. NOTE the name follows that convention rather than
            the textbook definition: lm-polygraph's "Perplexity" returns mean NLL
            (log-perplexity), NOT exp(mean NLL) — there is no exp. Under PRR only
            the ordering matters, and exp is monotonic, so mean-NLL vs exp(mean-NLL)
            give identical rankings. Higher = more uncertain; like "sum", not
            bounded to [0, 1]. (It is NOT lm-polygraph's full-distribution token
            entropy, which needs the per-step softmax the Tier-1 cache omits.)
    Pick one, keep it fixed, and record which you used.
    """
    lp = np.asarray(token_logprobs, dtype=float)
    if aggregate == "sum":
        return float(-lp.sum())
    if aggregate == "perplexity":
        return float(-lp.mean())
    probs = np.exp(lp)
    conf = probs.mean() if aggregate == "mean" else probs.min()
    return float(1.0 - conf)


# ---- the FAIR unsupervised floor -------------------------------------------------------------------
FLOOR_AGGREGATES = ("sum", "perplexity", "min")


def fair_floor(records_te, y_te, prr_fn, aggregates=FLOOR_AGGREGATES):
    """The honest unsupervised bar for a cell: the BEST-scoring of the standard MSP floors.

    WHY THIS EXISTS (2026-07-22). Drivers used to hard-code the floor to `sum`, which is NOT
    length-normalised. On several datasets a different aggregate is far stronger -- pubmed_qa
    sum=+0.202 vs min=+0.371, ASQA sum=+0.148 vs perplexity=+0.316 -- so a "beats the floor" claim
    measured against `sum` alone can be more than twice the honest margin. That is exactly the
    artefact that produced, and then killed, the cnn headline (STOCKTAKE PART VI/XI/XII). A supervised
    method should have to beat the best thing you can get for free, not the most convenient one.

    Returns (vector, name): the per-example uncertainty vector of the winning floor, and which it was.
    Report the NAME alongside any margin -- which floor wins is itself informative (it says whether the
    dataset's signal is length-driven, whole-sequence, or weakest-link).

    `prr_fn` is injected (normally `luq.results.prr`) to avoid a circular import.
    """
    cands = {a: np.asarray([msp_uncertainty(r["token_logprobs"], a) for r in records_te], dtype=float)
             for a in aggregates}
    best = max(cands, key=lambda a: prr_fn(y_te, cands[a]))
    return cands[best], best
