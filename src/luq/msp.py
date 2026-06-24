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
            lm-polygraph calls MaximumSequenceProbability, so use it when
            comparing against published lm-polygraph numbers. Not length-
            normalised, and not bounded to [0, 1] like the others (fine for
            PRR, which only uses the ranking).
    "nll" : mean per-token negative log-likelihood = (1/L) * sum(-log p),
            the LENGTH-NORMALISED counterpart to "sum" (= "sum" / L). This is
            the unsupervised baseline Joe wants: "sum" is length-confounded
            (it grows with the number of tokens), and dividing by length
            removes that. Joe refers to this as the "entropy" baseline
            (lm-polygraph's loose naming); precisely it is the chosen-token
            perplexity in log form (perplexity = exp of this), NOT
            lm-polygraph's full-distribution token entropy, which would need
            the per-step softmax the Tier-1 cache does not store. Under PRR
            only the ordering matters, so the exp-vs-log form and the absolute
            scale are irrelevant. Higher = more uncertain; like "sum", not
            bounded to [0, 1].
    Pick one, keep it fixed, and record which you used.
    """
    lp = np.asarray(token_logprobs, dtype=float)
    if aggregate == "sum":
        return float(-lp.sum())
    if aggregate == "nll":
        return float(-lp.mean())
    probs = np.exp(lp)
    conf = probs.mean() if aggregate == "mean" else probs.min()
    return float(1.0 - conf)
