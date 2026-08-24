"""Token-level Mahalanobis distance and the published hybrids built on it.

This is a port of the reference implementation shipped with the Hidden Failures code base
(`satmd_baseline/token_mahalanobis_distance.py`, `relative_token_mahalanobis_distance.py`,
`average_token_mahalanobis_distance.py`, `huq_msp_lrtmd.py`, and `run_hbo.py`), not a
reimplementation from the paper text. Each function names the file it came from so the two can be
diffed. Where this code departs from the reference it says so in capitals at the point of departure.

THE ONE STRUCTURAL DEVIATION, STATED ONCE HERE
----------------------------------------------
The reference fits a Mahalanobis distance at EVERY hidden layer and then learns a ridge regression
over the resulting per-layer distances. That is where the "supervised aggregation" in the method name
lives. This project caches per-token hidden states at ONE layer per model, the middle layer, and
storing every layer for eight datasets would need roughly 500 GB per model, so the port runs at the
middle layer only.

Consequences, which must be carried into any write-up:
  * The distance-based scores are MIDDLE-LAYER ADAPTATIONS of the published methods, not
    reproductions of them.
  * With a single layer the ridge regression has one input feature and a positivity constraint, so
    its output is a monotone increasing affine function of that feature, and its ranking -- hence its
    PRR -- is identical to the mean distance itself, unless the fitted coefficient is clipped to
    zero, in which case the score is constant and carries no signal. `satmd` therefore reports the
    fitted coefficient so this is visible rather than inferred.
  * The hybrid back-off is NOT affected: its reference recipe selects a single layer index (the
    middle one) out of the saved stack, so at the middle layer it is exact.

WINDOW
------
Every per-token cache in this project stores the last prompt position plus the generated tokens. All
distances here are computed over that window, for training rows, evaluation rows and the background
corpus alike, so the three are always measured on the same footing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import rankdata

# The reference tries these jitters in order and takes the first that makes the covariance positive
# semi-definite (lm_polygraph_lite/estimators/mahalanobis_distance.py).
JITTERS = [10 ** exp for exp in range(-15, 0, 1)]

# The reference's default token filter for the SATMD family (scripts/table1_satmd.sh: --metric_thr 0.3).
DEFAULT_METRIC_THR = 0.3

# The reference's fixed dev split for fitting the meta-regressor
# (average_token_mahalanobis_distance.py: train_test_split(..., test_size=0.5, random_state=42)).
DEV_SIZE = 0.5
DEV_RANDOM_STATE = 42


class BoundedStatsCache(dict):
    """A fitted-statistics cache with a hard entry limit.

    Each entry holds an inverse covariance of about 67 MB at 4096 dimensions, so an unbounded cache
    would grow to tens of gigabytes over a full grid. Oldest entries are evicted first, which suits
    the caller's access pattern: cells sharing a training pool are computed consecutively.
    """

    def __init__(self, max_entries=12):
        super().__init__()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def __contains__(self, k):
        hit = super().__contains__(k)
        self.hits += int(hit)
        self.misses += int(not hit)
        return hit

    def __setitem__(self, k, v):
        while len(self) >= self.max_entries:
            del self[next(iter(self))]
        super().__setitem__(k, v)


@dataclass
class MDStats:
    """A fitted centroid and inverse covariance, plus what it was fitted on."""
    centroid: np.ndarray        # (D,)
    sigma_inv: np.ndarray       # (D, D)
    n_tokens: int
    n_rows: int
    jitter: float
    key: str


def _stack(rows):
    """Concatenate a list of (n_tokens_i, D) arrays into one (N, D) float32 array."""
    keep = [r for r in rows if r is not None and len(r) > 0]
    if not keep:
        raise ValueError("no tokens to fit on -- refusing to build a degenerate centroid")
    return np.ascontiguousarray(np.concatenate(keep, axis=0), dtype=np.float32)


def pool_key(row_ids, layer, metric_thr, kind):
    """A content hash of exactly which rows a statistic was fitted on.

    Cells in the grid share training pools -- every eval's single-source far-OOD rung draws the same
    source with the same cap and the same seed -- so a fit keyed on the row identities can be reused
    instead of repeated. The key is the row identities themselves, never a cell name, because two
    differently-named cells may or may not share a pool and only the rows decide.
    """
    payload = json.dumps({"rows": sorted(map(list, row_ids)), "layer": int(layer),
                          "thr": float(metric_thr), "kind": kind}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def compute_inv_covariance(centroid, features):
    """Port of `compute_inv_covariance`. Returns (sigma_inv float32 (D,D), jitter_eps).

    Kept faithful in three details that change the numbers: the covariance is torch's unbiased
    estimator over the token axis, the jitter is the FIRST value in the fixed ladder that makes every
    eigenvalue non-negative, and the inverse is taken in float64 and cast back to float32.
    """
    X = torch.from_numpy(features)                       # (N, D) float32
    cov = torch.cov(X.T)                                 # (D, D), unbiased, over N tokens
    jitter_eps = None
    eye = torch.eye(cov.shape[1], dtype=cov.dtype)
    for jitter_eps in JITTERS:
        upd = cov + jitter_eps * eye
        if bool((torch.linalg.eigh(upd).eigenvalues >= 0).all()):
            break
    cov = cov + jitter_eps * eye
    sigma_inv = torch.inverse(cov.to(torch.float64)).float()
    return sigma_inv.numpy(), float(jitter_eps)


def fit_md(rows, row_metrics=None, metric_thr=0.0, layer=None, row_ids=None, kind="fg", cache=None):
    """Fit centroid + inverse covariance on the tokens of `rows`. Port of TokenMahalanobisDistance.

    `row_metrics` is the sequence-level correctness of each row. The reference expands it to one value
    per token and keeps only tokens whose value is at or above `metric_thr`, and it applies that filter
    only when more than 10 TOKENS survive -- a token count, not a row count. Both are reproduced.
    """
    key = pool_key(row_ids, layer, metric_thr, kind) if row_ids is not None else None
    if cache is not None and key is not None and key in cache:
        return cache[key]

    use = rows
    n_rows = len(rows)
    if metric_thr > 0 and row_metrics is not None:
        m = np.asarray(row_metrics, dtype=float)
        per_token = np.concatenate([np.repeat(m[i], len(r)) for i, r in enumerate(rows)]) \
            if len(rows) else np.zeros(0)
        if int((per_token >= metric_thr).sum()) > 10:
            use = [r for i, r in enumerate(rows) if m[i] >= metric_thr]
            n_rows = len(use)

    feats = _stack(use)
    centroid = feats.mean(axis=0)
    sigma_inv, jitter = compute_inv_covariance(centroid, feats)
    stats = MDStats(centroid=centroid.astype(np.float64), sigma_inv=sigma_inv,
                    n_tokens=int(feats.shape[0]), n_rows=int(n_rows), jitter=jitter, key=key or "")
    if cache is not None and key is not None:
        cache[key] = stats
    return stats


def md_tokens(rows, stats: MDStats):
    """Per-token Mahalanobis distance for each row. Port of the single-centroid path.

    The reference takes the SQUARE ROOT of the quadratic form. That matters: the ridge regression and
    the hybrid back-off's percentile are both computed on these values, and only the back-off is
    invariant to a monotone transform.
    """
    mu = stats.centroid.astype(np.float32)
    S = stats.sigma_inv
    out = []
    for r in rows:
        if r is None or len(r) == 0:
            out.append(np.zeros(0, dtype=np.float64))
            continue
        d = np.asarray(r, dtype=np.float32) - mu
        q = np.einsum("ij,jk,ik->i", d, S, d, optimize=True)
        # A jittered covariance can still leave a marginally negative quadratic form at float32.
        # Clipping at zero before the square root is the only safe reading; it is recorded because a
        # silent nan would otherwise propagate into the mean.
        out.append(np.sqrt(np.clip(q, 0.0, None)).astype(np.float64))
    return out


def md_mean(rows, stats: MDStats):
    """Mean per-token distance per row -- the reference's `aggregation='mean'`."""
    return np.array([float(np.mean(d)) if len(d) else np.nan for d in md_tokens(rows, stats)])


def rmd_mean(rows, fg: MDStats, bg: MDStats):
    """Relative distance: per-token MD minus per-token background MD, then averaged.

    Port of RelativeTokenMahalanobisDistance, which subtracts at TOKEN level and only then aggregates.
    Subtracting the two row means would give the same number here, but the reference's order is kept
    so the code matches the source it claims to port.
    """
    a, b = md_tokens(rows, fg), md_tokens(rows, bg)
    out = []
    for da, db in zip(a, b):
        out.append(float(np.mean(da - db)) if len(da) else np.nan)
    return np.array(out)


# --------------------------------------------------------------------------------------------------
# The hybrid back-off. Port of run_hbo.py.
# --------------------------------------------------------------------------------------------------
def hbo(msp_scores, probe_scores, test_md, dev_md):
    """Rank-combine an unsupervised and a supervised score, weighted by how out-of-distribution the
    test example looks. Port of run_hbo.py, which is the released implementation of the paper's
    equations 2-4.

    `test_md` and `dev_md` are per-example mean distances at ONE layer -- the reference selects the
    middle layer index from its saved stack, which is the layer this project caches, so no adaptation
    is involved here.

    Two properties of the formula, recorded so they are not later mistaken for bugs: the supervised
    weight never exceeds 0.5, and it is exactly 0 for any example whose percentile exceeds 0.5, so the
    score collapses to the unsupervised one for the more shifted half of the test set.
    """
    dev = list(np.asarray(dev_md, dtype=float))
    pct = np.array([rankdata(dev + [float(x)])[-1] / (len(dev) + 1) for x in np.asarray(test_md, float)])
    w_unsup = np.minimum(1.0, pct + 0.5)
    w_sup = 1.0 - w_unsup
    score = w_unsup * rankdata(np.asarray(msp_scores, float)) + w_sup * rankdata(np.asarray(probe_scores, float))
    return score, pct, w_sup


# --------------------------------------------------------------------------------------------------
# HUQ. Port of huq_msp_lrtmd.py.
# --------------------------------------------------------------------------------------------------
def total_uncertainty_linear_step(epistemic, aleatoric, threshold_min=0.1, threshold_max=0.9, alpha=0.1):
    """Verbatim port of `total_uncertainty_linear_step`, including the second assignment, which can
    never fire because its condition is a subset of the first. That is a property of the published
    code and is preserved rather than tidied: removing it would change nothing numerically but would
    make this no longer a port."""
    aleatoric = np.asarray(aleatoric, dtype=float)
    epistemic = np.asarray(epistemic, dtype=float)
    n_preds = len(aleatoric)
    n_lowest = int(n_preds * threshold_min)
    n_max = int(n_preds * threshold_max)

    aleatoric_rank = rankdata(aleatoric)
    epistemic_rank = rankdata(epistemic)

    total_rank = (1 - alpha) * epistemic_rank + alpha * aleatoric_rank
    total_rank[epistemic_rank <= n_lowest] = rankdata(aleatoric[epistemic_rank <= n_lowest])
    sel = (aleatoric_rank > n_max) & (epistemic_rank <= n_lowest)
    total_rank[sel] = aleatoric_rank[sel]
    return total_rank


def grid_search_hp(epistemic, aleatoric, metrics, prr_fn,
                   t_min_min=0.0, t_min_max=0.3, t_max_min=0.8, t_max_max=1.0,
                   alpha_min=0.0, alpha_max=1.0):
    """Verbatim port of `grid_search_hp`, including the reference's own bug fix on the best-score
    update. The search runs on the DEV HALF of the training pool only; the caller must never pass it
    evaluation labels, and `md_hybrids.py` asserts that."""
    eps = 0.01
    t_min_best, t_max_best, alpha_best = 0, 1, 0
    best = prr_fn(metrics, epistemic)
    for t_min in np.arange(t_min_min, t_min_max + eps, 0.05):
        for t_max in np.arange(t_max_min, t_max_max + eps, 0.05):
            for alpha in np.arange(alpha_min, alpha_max + eps, 0.1):
                unc = total_uncertainty_linear_step(epistemic, aleatoric, t_min, t_max, alpha)
                new = prr_fn(metrics, unc)
                if new > best:
                    best, t_min_best, t_max_best, alpha_best = new, t_min, t_max, alpha
    return best, t_min_best, t_max_best, alpha_best
