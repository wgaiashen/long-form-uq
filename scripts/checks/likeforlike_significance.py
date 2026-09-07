#!/usr/bin/env python
"""Dataset-level paired tests for the like-for-like probability-aggregation comparison.

The summary script reports macro PRR, which is a mean over eight datasets and carries no measure of
spread. A macro difference of a few thousandths is not evidence of anything, so any claim that the
learned weighting beats a training-free rule has to be made at the level of the unit of analysis,
which is the dataset (n = 8), not the cell.

For each model, each rung, and each fixed weighting rule, this reports the paired
difference in PRR across the eight evaluation datasets: the macro difference, how many datasets it wins,
an exact Wilcoxon signed-rank p-value with zeros dropped, and a bootstrap confidence interval. The fixed
rules are unsupervised and do not depend on the training pool, so a single PRR per dataset is compared
against the learned score at every rung.

Reads only `results/hybrids/likeforlike__<slug>.csv`; computes no PRR of its own.

    python scripts/checks/likeforlike_significance.py
    python scripts/checks/likeforlike_significance.py --models google/gemma-2-9b
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache  # noqa: E402

FIXED = ["sequence NLL (published MSP)", "mean token NLL", "minimum token probability",
         "TokenSAR", "answer-span mean NLL", "answer-span sequence NLL"]
OOD_RUNGS = ["LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
RUNGS = ["ID"] + OOD_RUNGS


def exact_wilcoxon(d):
    """The exact test with zeros dropped, matching the convention used elsewhere in this project."""
    d = np.asarray([x for x in d if x != 0.0], float)
    if len(d) == 0:
        return float("nan"), 0
    from scipy.stats import wilcoxon
    return float(wilcoxon(d, alternative="two-sided", method="exact").pvalue), len(d)


def boot_ci(d, b=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    m = np.array([np.mean(d[rng.randint(0, len(d), len(d))]) for _ in range(b)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(model):
    path = ROOT / "results" / "hybrids" / f"likeforlike__{cache._slug(model)}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["dataset"]] = float(r["prr"])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["meta-llama/Meta-Llama-3.1-8B", "Qwen/Qwen2.5-14B", "google/gemma-2-9b"])
    args = ap.parse_args()

    for model in args.models:
        by = load(model)
        if by is None:
            print(f"-- {model}: no like-for-like table on disk, skipping --")
            continue
        datasets = sorted(by["mean token NLL"])
        print(f"\n=== {model}  (n = {len(datasets)} datasets) ===")

        best = max(FIXED, key=lambda m: np.mean([by[m][d] for d in datasets]))
        print(f"strongest training-free rule on this model: {best} "
              f"(macro {np.mean([by[best][d] for d in datasets]):+.4f})\n")

        print(f"{'rung':16s}{'fixed rule':32s}{'macro d':>9s}{'wins':>7s}{'p':>9s}   95% CI")
        for rung in RUNGS:
            key = f"CAWSA lambda=2 [{rung}]"
            if key not in by:
                print(f"{rung:16s}MISSING")
                continue
            for fx in FIXED:
                d = np.array([by[key][ds] - by[fx][ds] for ds in datasets])
                p, n = exact_wilcoxon(d)
                lo, hi = boot_ci(d)
                mark = "  <-- strongest rule" if fx == best else ""
                print(f"{rung:16s}{fx:32s}{d.mean():+9.4f}{f'{int((d>0).sum())}/{len(d)}':>7s}"
                      f"{p:9.4f}   [{lo:+.3f}, {hi:+.3f}]{mark}")
            print()


if __name__ == "__main__":
    main()
