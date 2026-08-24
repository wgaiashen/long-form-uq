#!/usr/bin/env python
"""Report the probability-branch substitution against its comparators, per rung.

The analysis is exactly the one fixed in prereg/M8_cawsa_hbo_substitution.md section 4, and nothing
outside it is computed here. The unit is the DATASET, n = 8. Cell-level counts would be
pseudo-replication for the unsupervised scores, whose value does not depend on the training pool.

    python scripts/checks/cawsa_hbo_analysis.py --model meta-llama/Meta-Llama-3.1-8B
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache                                                           # noqa: E402

RUNGS = ["ID", "LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
ORDER = ["msp", "floor_min", "cawsa", "saplma", "hbo", "cawsa_hbo"]
LABEL = {"msp": "published MSP", "floor_min": "min token probability",
         "cawsa": "learned weighting", "saplma": "probe",
         "hbo": "back-off (published)", "cawsa_hbo": "back-off (substituted)"}
# Registered comparisons. The first is close to arithmetic on the far rungs and is labelled as such.
DELTAS = [("cawsa_hbo", "hbo"), ("cawsa_hbo", "cawsa"), ("cawsa_hbo", "saplma")]


def exact_wilcoxon(d):
    """scipy's exact test with zeros dropped, matching the convention used elsewhere here."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--csv", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    slug = cache._slug(args.model)
    src = ROOT / (args.csv or f"results/hybrids/pdl_cawsa_hbo__{slug}.csv")
    d = pd.read_csv(src)
    diag = pd.read_csv(src.with_name(src.stem + "__diagnostics.csv"))

    print(f"POPULATION {args.model}   source {src.relative_to(ROOT)}")
    cells = d[["eval", "rung"]].drop_duplicates()
    print(f"COVERAGE {len(cells)} cells, {d['eval'].nunique()} datasets x {d.rung.nunique()} rungs")
    missing = [m for m in ORDER if m not in set(d.method)]
    if missing:
        print(f"  METHODS ABSENT (left blank, never zero): {missing}")

    lines = []
    for rung in RUNGS:
        r = d[d.rung == rung]
        if r.empty:
            print(f"\n[{rung}] no cells -- NOT MEASURED"); continue
        piv = r.pivot_table(index="eval", columns="method", values="prr_mean")
        piv = piv.reindex(columns=[m for m in ORDER if m in piv.columns])
        g = diag[diag.rung == rung]
        print(f"\n{'=' * 100}\n[{rung}]  n = {len(piv)} datasets   "
              f"mean supervised weight {g.mean_w_sv.mean():.4f}   "
              f"fraction beyond the gate {g.frac_R_gt_half.mean():.3f}")
        print("\n  macro PRR")
        for m in piv.columns:
            v = piv[m].dropna()
            print(f"    {LABEL[m]:26s} {v.mean():+.4f}   (n = {len(v)})")
        print("\n  per dataset")
        print(piv.round(4).to_string())

        print("\n  registered paired comparisons, dataset as the unit")
        for a, b in DELTAS:
            if a not in piv.columns or b not in piv.columns:
                continue
            sub = piv[[a, b]].dropna()
            dd = (sub[a] - sub[b]).values
            if len(dd) == 0:
                continue
            p, n_nz = exact_wilcoxon(dd)
            lo, hi = boot_ci(dd)
            note = ""
            if a == "cawsa_hbo" and b == "hbo" and g.mean_w_sv.mean() < 1e-12:
                note = ("  <- the supervised weight is zero everywhere on this rung, so this is the "
                        "known difference between the two probability scores, not evidence")
            print(f"    {LABEL[a]} - {LABEL[b]:26s} {dd.mean():+.4f}  "
                  f"{int((dd > 0).sum())}/{len(dd)}  p = {p:.4f}  CI [{lo:+.4f}, {hi:+.4f}]{note}")
            lines.append(dict(model=args.model, rung=rung, comparison=f"{a}-{b}",
                              n=len(dd), mean_delta=round(dd.mean(), 6),
                              wins=int((dd > 0).sum()), p_exact=round(p, 6),
                              ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                              mean_w_sv=round(float(g.mean_w_sv.mean()), 6),
                              frac_beyond_gate=round(float(g.frac_R_gt_half.mean()), 6)))

    if lines:
        out = ROOT / (args.out or f"results/hybrids/cawsa_hbo_stats__{slug}.csv")
        pd.DataFrame(lines).to_csv(out, index=False)
        print(f"\nwrote {out.relative_to(ROOT)} ({len(lines)} rows)")


if __name__ == "__main__":
    main()
