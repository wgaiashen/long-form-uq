"""MSP: an unsupervised token-probability uncertainty score. Comes free.

We already capture per-token logprobs in the Tier-1 record at generation time, so
MSP costs nothing extra. Sequence confidence is an aggregate of the per-token
probabilities; uncertainty = 1 - confidence (higher = more uncertain).
"""
import numpy as np


def msp_uncertainty(token_logprobs, aggregate: str = "mean") -> float:
    """token_logprobs: list of log p(chosen token). Returns uncertainty.

    "mean": mean token probability (smooth, whole-sequence view).
    "min" : least-confident token (closest to the original 'maximum softmax
            probability' idea applied to the worst step).
    Pick one, keep it fixed, and record which you used.
    """
    probs = np.exp(np.asarray(token_logprobs, dtype=float))
    conf = probs.mean() if aggregate == "mean" else probs.min()
    return float(1.0 - conf)
