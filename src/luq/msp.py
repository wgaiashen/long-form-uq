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
    Pick one, keep it fixed, and record which you used.
    """
    lp = np.asarray(token_logprobs, dtype=float)
    if aggregate == "sum":
        return float(-lp.sum())
    probs = np.exp(lp)
    conf = probs.mean() if aggregate == "mean" else probs.min()
    return float(1.0 - conf)
