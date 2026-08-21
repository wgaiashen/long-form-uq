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
            lm-polygraph calls MaximumSequenceProbability (and what the Hidden
            Failures reports as "MSP"), so use it when comparing against those
            numbers. Not length-normalised, and not bounded to [0, 1] like the
            others (fine for PRR, which only uses the ranking).
    "perplexity" : mean per-token negative log-likelihood = (1/L) * sum(-log p),
            the LENGTH-NORMALISED counterpart to "sum". This is exactly
            lm-polygraph's `Perplexity` estimator (`-np.mean(ll)`; uhead inherits
            it) and the "Perplexity" baseline in the Hidden Failures, so the
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

# ---- the PRE-REGISTERED primary floor (2026-07-24 meeting decision) ---------------------------------
# Max-of-three was rejected ("gives the baseline three shots; one may look good by chance"). Instead we
# FIX ONE aggregate in advance and use it as the bar on EVERY dataset. `min` is chosen because it is the
# strongest averaged ACROSS datasets (cross-dataset mean PRR: min ~0.28 > perplexity ~0.21 > sum ~0.20),
# and because pre-committing to one aggregate with no per-dataset hindsight is the deployment-honest choice.
# Report the three-variant row per dataset regardless, and where a DIFFERENT variant is the strongest free
# score on a dataset (cnn/samsum -> perplexity) DUAL-REPORT against it too (see the project record).
PRIMARY_FLOOR_AGG = "min"


def all_floors(records_te, prr_fn=None, y_te=None, aggregates=FLOOR_AGGREGATES):
    """Every floor variant's per-example uncertainty vector, keyed by aggregate name (`msp_<agg>`).

    This is what feeds the per-dataset THREE-VARIANT floor table (sum / perplexity / min), which we always
    show. If `prr_fn` and `y_te` are given, also returns a dict of each variant's PRR. Read-only over the
    cached logprobs -- deterministic, CPU-only.
    """
    cands = {f"msp_{a}": np.asarray([msp_uncertainty(r["token_logprobs"], a) for r in records_te],
                                    dtype=float)
             for a in aggregates}
    if prr_fn is not None and y_te is not None:
        prrs = {name: prr_fn(y_te, vec) for name, vec in cands.items()}
        return cands, prrs
    return cands


def primary_floor(records_te, agg: str = PRIMARY_FLOOR_AGG):
    """The PRE-REGISTERED primary unsupervised bar, FIXED across all datasets (default = msp_min).

    Use THIS for every 'beats the floor' verdict going forward (replaces the rejected max-of-three
    `fair_floor` for the verdict). `fair_floor` is retained only where the max-of-three view is explicitly
    wanted, and `all_floors` for the per-dataset three-variant table.

    Returns (vector, name) e.g. (..., "msp_min") -- carry the name so the bar is unambiguous in the CSV.
    """
    vec = np.asarray([msp_uncertainty(r["token_logprobs"], agg) for r in records_te], dtype=float)
    return vec, f"msp_{agg}"


def fair_floor(records_te, y_te, prr_fn, aggregates=FLOOR_AGGREGATES):
    """The honest unsupervised bar for a cell: the BEST-scoring of the standard MSP floors.

    WHY THIS EXISTS (2026-07-22). Drivers used to hard-code the floor to `sum`, which is NOT
    length-normalised. On several datasets a different aggregate is far stronger -- pubmed_qa
    sum=+0.202 vs min=+0.371, ASQA sum=+0.148 vs perplexity=+0.316 -- so a "beats the floor" claim
    measured against `sum` alone can be more than twice the honest margin. That is exactly the
    artefact that produced, and then killed, the cnn headline (the project record). A supervised
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
