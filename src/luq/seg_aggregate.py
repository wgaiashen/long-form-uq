"""W-B1 — a LEARNED aggregator over per-sentence P(correct).

THE PROBLEM. Decompose-and-aggregate scores each sentence of a long answer with the probe, then has to
combine the per-sentence probabilities into ONE instance score. Today we use fixed rules
(`aggregators.aggregate_conf`: mean / min / geomean), and each encodes a strong, unargued assumption:

    mean     every claim counts equally -> a single fabrication is DILUTED by surrounding true sentences
    min      the response is only as good as its worst claim -> OVER-SENSITIVE to one noisy sentence
    geomean  the soft-conjunction under independence -> between the two, still fixed

The framing doc's actual contribution is to *learn* the aggregation rather than pick one by hand.

THE DESIGN. Rather than an unconstrained network (p >> n here; and an arbitrary set-function would be
unfalsifiable), we learn ONE interpretable parameter: the exponent of a POWER MEAN.

    M_alpha(p) = ( (1/n) * sum_i p_i^alpha ) ^ (1/alpha)          (alpha != 0)
    M_0(p)     = geometric mean                                    (the alpha -> 0 limit)

This single family CONTAINS every fixed rule we currently use:

    alpha -> -inf : min          (pure weakest-link)
    alpha  = 0    : geomean      (soft conjunction / independence)
    alpha  = 1    : mean         (full dilution)
    alpha -> +inf : max

So learning `alpha` asks the data where on the dilution<->weakest-link axis a task actually sits, instead of
us asserting it. The learned value is directly interpretable and reportable per dataset: **alpha < 1 means
one bad claim should dominate; alpha > 1 means the response survives isolated errors.** That interpretation
is the deliverable, not just the PRR.

GROUNDING (unit-tested in tests/test_seg_aggregate.py): at alpha=1 this must reproduce the `mean` aggregator
EXACTLY, at alpha->0 the geomean, and at very negative alpha the min. If those limits fail, the family is
miswired and any "learned aggregator wins" result would be meaningless.

Numerics: probabilities are clipped away from 0 before exponentiation (a single p=0 would send the geomean
and every negative-alpha mean to 0 and destroy the gradient), and the power mean is computed in log space
via logsumexp for stability with large |alpha|.
"""

import numpy as np


EPS = 1e-6


def power_mean(p, alpha):
    """Power mean of probabilities `p` with exponent `alpha`. Handles the alpha->0 (geometric) limit.

    Computed in log space: M = exp( (1/alpha) * log( mean( exp(alpha * log p) ) ) ), with the max factored
    out, so large |alpha| does not overflow.
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    if p.size == 0:
        return 0.5
    lp = np.log(p)
    if abs(alpha) < 1e-8:                      # alpha -> 0 : geometric mean
        return float(np.exp(lp.mean()))
    z = alpha * lp
    m = z.max()
    return float(np.exp((m + np.log(np.mean(np.exp(z - m)))) / alpha))


def aggregate(p_list, alpha):
    """Apply the power mean to a list of per-sentence probability arrays -> one score per instance."""
    return np.array([power_mean(p, alpha) for p in p_list], dtype=float)


def fit_alpha(p_list_tr, y_tr, prr_fn, grid=None):
    """Choose `alpha` on the TRAINING instances only, by grid search on the training objective.

    A grid (not gradient descent) is deliberate: alpha is ONE parameter, the objective is cheap, and a grid
    cannot get stuck in a local optimum or silently fail to converge -- which for a single interpretable
    parameter matters more than elegance. The grid spans min-like to max-like behaviour.

    IMPORTANT: fitted on TRAIN ONLY. Selecting alpha on the test set would be choosing the aggregator on the
    thing we then report, which is exactly the trap the project's aggregation work is meant to avoid.
    """
    if grid is None:
        grid = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    best, best_s = 1.0, -np.inf
    for a in grid:
        s = prr_fn(y_tr, 1.0 - aggregate(p_list_tr, a))     # uncertainty = 1 - aggregated confidence
        if s > best_s:
            best, best_s = a, s
    return best, best_s


def learned_aggregate(p_list_tr, y_tr, p_list_te, prr_fn):
    """Fit alpha on train, apply to test. Returns (test uncertainty vector, fitted alpha).

    The returned alpha is the interpretable payload: report it per dataset alongside the PRR.
    """
    alpha, _ = fit_alpha(p_list_tr, y_tr, prr_fn)
    return 1.0 - aggregate(p_list_te, alpha), alpha
