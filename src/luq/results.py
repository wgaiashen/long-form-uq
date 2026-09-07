"""Stage 4: results CSV + PRR.

Every method writes one row per example into the same CSV schema that robust-uq-eval
reads:
    dataset, task, split, correctness (higher=better), <uncertainty col> (higher=worse)

robust-uq-eval is not installed yet, so prr() here is a local fallback so you are not
blocked. Switch to robust-uq-eval for the official numbers once it is available.
"""
import csv

import numpy as np


def write_csv(path, rows: list[dict], uncertainty_cols, meta_cols=None) -> None:
    """rows: dicts with keys dataset, task, split, correctness, <uncertainty col(s)>.

    uncertainty_cols: one column name or a list of them, so several methods can
    share a CSV (same examples, same correctness, one column per method).

    meta_cols: optional extra columns inserted right after `correctness` (e.g.
    `label_field`/`label_model`). This exists to STAMP WHICH LABEL the `correctness`
    column actually holds — the judge vs AlignScore mix-up has bitten this project more
    than once (the per-example CSV `correctness` is whatever --label-field was passed,
    NOT necessarily the judge). See results/README_LABELS.md.
    """
    if isinstance(uncertainty_cols, str):
        uncertainty_cols = [uncertainty_cols]
    meta_cols = meta_cols or []
    fields = ["dataset", "task", "split", "correctness", *meta_cols, *uncertainty_cols]
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

    Verified against lm-polygraph's PredictionRejectionArea + oracle/random
    normalisation on sciq ID: agrees to <0.001 on five different scores (their
    random baseline is simulated over 1000 shuffles, ours is analytic, hence the
    tiny residual). Still cross-check robust-uq-eval once installed, since that
    is the project's official scorer.
    """
    correctness = np.asarray(correctness, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    n = len(correctness)

    # A NON-FINITE SCORE MUST NOT PRODUCE A NUMBER.
    # np.argsort puts NaN at the end and happily ranks the rest, so an ALL-NaN score vector used to
    # return the PRR of an arbitrary permutation -- a different plausible-looking value for every label
    # vector (-0.686, +0.013, -0.044 on three different inputs). That is exactly how asqa's wMSP cell
    # came to read -0.0439 across all 8 variants and 3 seeds: the scores were entirely NaN and this
    # function invented a ranking for them. Partial NaN is worse still, because it silently scores on a
    # subset while reporting as the full test set.
    # Returns NaN rather than raising so one bad method cannot kill a 40-cell ladder run: NaN lands in
    # the CSV as an EMPTY cell, which reads as "not measured". The warning is what makes it loud.
    n_bad = int((~np.isfinite(uncertainty)).sum())
    if n_bad:
        print(f"    prr(): {n_bad}/{n} uncertainty values are NaN/inf -> returning NaN, NOT a number. "
              "The cell will be BLANK (not measured). Fix the score, do not read the blank as a result.",
              flush=True)
        return float("nan")

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
