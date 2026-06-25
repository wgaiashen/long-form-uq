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
