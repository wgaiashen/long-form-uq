"""Stage 4: results CSV + PRR.

Every method writes one row per example into the same CSV schema that robust-uq-eval
reads:
    dataset, task, split, correctness (higher=better), <uncertainty col> (higher=worse)

robust-uq-eval is not installed yet, so prr() here is a local fallback so you are not
blocked. Switch to robust-uq-eval for the official numbers once it is available.
"""
import csv

import numpy as np


def write_csv(path, rows: list[dict], uncertainty_col: str) -> None:
    """rows: dicts with keys dataset, task, split, correctness, <uncertainty_col>."""
    fields = ["dataset", "task", "split", "correctness", uncertainty_col]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def prr(correctness, uncertainty) -> float:
    """Prediction-Rejection Ratio (a teaching implementation).

    Idea: rank examples by uncertainty, reject the most uncertain first, and watch the
    mean correctness of what remains. A good score makes it climb. We measure the area
    under that rejection curve, then normalise:

        PRR = (model_area - random_area) / (oracle_area - random_area)

    1 = as good as the oracle, 0 = no better than random, < 0 = worse than random.
    The oracle ranks by true correctness. Model and oracle are scored the same way so
    the comparison is fair.

    TODO(gaia): verify this against robust-uq-eval on one dataset before trusting it in
    the writeup.
    """
    correctness = np.asarray(correctness, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    n = len(correctness)

    def area(rank_by):
        # Keep the LEAST-uncertain prefix as we reject the most-uncertain end.
        order = np.argsort(rank_by)            # ascending: least "bad" first
        kept = correctness[order]
        mean_of_kept = np.cumsum(kept) / np.arange(1, n + 1)
        return mean_of_kept.mean()

    model_area = area(uncertainty)
    oracle_area = area(-correctness)           # perfect uncertainty = -correctness
    random_area = correctness.mean()
    return float((model_area - random_area) / (oracle_area - random_area + 1e-12))
