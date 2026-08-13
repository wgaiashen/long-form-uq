"""Shared paired-statistics helpers for the report verification pack.

Every test in the pack uses the same convention, which is the project's standing rule:
the DATASET is the unit of analysis (n = 8), seeds are averaged inside a cell before the
dataset enters, and OOD means the mean of the four OOD rungs.  Keeping that in one place
stops a second copy of the convention drifting away from the first.
"""

import numpy as np
from scipy import stats

OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
EVALS = ["asqa", "cnn_dailymail", "expertqa", "factscore",
         "med_quad", "pubmed_qa", "samsum", "xsum"]
BOOT_SEED = 20260814
N_BOOT = 10_000


def mean_ood(df, method, value_col="prr_mean", evals=EVALS, rungs=None):
    """One PRR per dataset: the mean over the four OOD rungs. Fails loudly on a gap."""
    rungs = rungs or OOD_RUNGS
    s = df[(df.method == method) & (df.rung.isin(rungs))]
    piv = s.pivot_table(index="eval", columns="rung", values=value_col)
    missing = [e for e in evals if e not in piv.index]
    if missing:
        raise ValueError(f"{method}: missing evals {missing}")
    gaps = piv.loc[evals].isna()
    if gaps.any().any():
        raise ValueError(f"{method}: missing rungs {gaps[gaps.any(axis=1)]}")
    return piv.loc[evals].mean(axis=1)


def paired(deltas, evals=EVALS, seed=BOOT_SEED, n_boot=N_BOOT):
    """Descriptive + inferential summary of a per-dataset delta vector.

    Wilcoxon is the exact two-sided signed-rank test (n = 8 is small enough that the exact
    null is used rather than the normal approximation).  The bootstrap resamples DATASETS,
    not cells, so the interval reflects the same unit of analysis as the test.
    """
    d = np.asarray(deltas, float)
    n = len(d)
    if np.allclose(d, 0):
        w_p = 1.0
    else:
        w_p = float(stats.wilcoxon(d, alternative="two-sided",
                                   zero_method="wilcox", mode="exact").pvalue)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # leave-one-dataset-out macro means: how much any single dataset carries the result
    lodo = {evals[i]: float(np.delete(d, i).mean()) for i in range(n)}
    return dict(
        per_dataset={evals[i]: float(d[i]) for i in range(n)},
        macro_mean=float(d.mean()), median=float(np.median(d)),
        n_positive=int((d > 0).sum()), n=n,
        wilcoxon_p_exact=w_p,
        boot_ci_lo=float(lo), boot_ci_hi=float(hi),
        boot_seed=seed, n_boot=n_boot,
        lodo_macro=lodo,
        lodo_min=float(min(lodo.values())), lodo_max=float(max(lodo.values())),
    )


def fmt_paired(name, r):
    lines = [f"{name}: macro {r['macro_mean']:+.4f}  median {r['median']:+.4f}  "
             f"{r['n_positive']}/{r['n']} positive  Wilcoxon p={r['wilcoxon_p_exact']:.4f}  "
             f"boot95% [{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}]",
             "  per dataset: " + "  ".join(f"{k}={v:+.4f}" for k, v in r["per_dataset"].items()),
             f"  LODO macro range [{r['lodo_min']:+.4f}, {r['lodo_max']:+.4f}]"]
    return "\n".join(lines)
