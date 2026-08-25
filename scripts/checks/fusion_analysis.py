#!/usr/bin/env python
"""Report the source-calibrated fusion against its registered comparators.

The analysis fixed in prereg/M10_source_calibrated_cawsa_saplma_fusion.md section 4, and nothing
outside it. The unit is the DATASET, n = 8: cell counts would be pseudo-replication for scores that
do not depend on the training pool.

    python scripts/checks/fusion_analysis.py --model meta-llama/Meta-Llama-3.1-8B
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
ORDER = ["saplma", "cawsa", "cawsa_saplma_src50", "cawsa_saplma_cohort50"]
LABEL = {"saplma": "probe",
         "cawsa": "learned weighting",
         "cawsa_saplma_src50": "source-calibrated fusion (target-independent)",
         "cawsa_saplma_cohort50": "cohort-rank ensemble (target-cohort-dependent)"}
# The primary comparison is the first. The rest are registered as secondary.
DELTAS = [("cawsa_saplma_src50", "saplma"),
          ("cawsa_saplma_src50", "cawsa"),
          ("cawsa_saplma_src50", "cawsa_saplma_cohort50")]


def exact_wilcoxon(d):
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
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    slug = cache._slug(args.model)
    src = ROOT / f"results/hybrids/pdl_fusion__{slug}.csv"
    d = pd.read_csv(src)
    diag = pd.read_csv(src.with_name(src.stem + "__diagnostics.csv"))

    print(f"POPULATION {args.model}   source {src.relative_to(ROOT)}")
    print(f"COVERAGE {len(d[['eval','rung']].drop_duplicates())} cells, "
          f"{d['eval'].nunique()} datasets x {d.rung.nunique()} rungs")
    gmax = max(diag[f"gateE_{m}_max"].max() for m in ("cawsa", "saplma"))
    print(f"GATE E worst row-level |d| over every cell and seed: {gmax:.3e}  "
          f"(0 means the retrained models are the frozen ones)")

    lines = []
    for rung in RUNGS:
        r = d[d.rung == rung]
        if r.empty:
            print(f"\n[{rung}] NOT MEASURED"); continue
        piv = r.pivot_table(index="eval", columns="method", values="prr_mean")
        piv = piv.reindex(columns=[m for m in ORDER if m in piv.columns])
        g = diag[diag.rung == rung]
        print(f"\n{'=' * 100}\n[{rung}]  n = {len(piv)} datasets")
        print("\n  macro PRR")
        for m in piv.columns:
            v = piv[m].dropna()
            print(f"    {LABEL[m]:48s} {v.mean():+.4f}   (n = {len(v)})")
        # Descriptive only (amendment 1.4): a fixed half-and-half on the percentile scale is equal
        # weighting only if the two spreads are comparable. Reported, never acted on.
        print(f"\n  realised target-percentile spread (descriptive, never acted on): "
              f"learned weighting sd {g.q_cawsa_sd.mean():.4f}, probe sd {g.q_saplma_sd.mean():.4f}")
        print("\n  per dataset")
        print(piv.round(4).to_string())

        print("\n  registered comparisons, dataset as the unit")
        for i, (a, b) in enumerate(DELTAS):
            if a not in piv.columns or b not in piv.columns:
                continue
            sub = piv[[a, b]].dropna()
            dd = (sub[a] - sub[b]).values
            if not len(dd):
                continue
            p, _ = exact_wilcoxon(dd)
            lo, hi = boot_ci(dd)
            tag = "PRIMARY  " if i == 0 else "secondary"
            print(f"    [{tag}] {LABEL[a][:28]} - {LABEL[b][:28]:30s} "
                  f"{dd.mean():+.4f}  {int((dd > 0).sum())}/{len(dd)}  p = {p:.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]")
            lines.append(dict(model=args.model, rung=rung, comparison=f"{a}-{b}",
                              primary=int(i == 0), n=len(dd), mean_delta=round(dd.mean(), 6),
                              wins=int((dd > 0).sum()), p_exact=round(p, 6),
                              ci_lo=round(lo, 6), ci_hi=round(hi, 6)))

        # Descriptive: how much of the cohort-rank gain survives being made target-independent.
        if {"cawsa_saplma_src50", "cawsa_saplma_cohort50", "saplma"} <= set(piv.columns):
            num = float((piv["cawsa_saplma_src50"] - piv["saplma"]).mean())
            den = float((piv["cawsa_saplma_cohort50"] - piv["saplma"]).mean())
            if den > 0.005:
                print(f"    retained gain = {num / den:+.2f}   ({num:+.4f} / {den:+.4f})  "
                      f"DESCRIPTIVE, not a test")
            else:
                print(f"    retained gain not computed: the cohort-rank gain over the probe is "
                      f"{den:+.4f}, too small to divide by")

    if lines:
        out = ROOT / (args.out or f"results/hybrids/fusion_stats__{slug}.csv")
        pd.DataFrame(lines).to_csv(out, index=False)
        print(f"\nwrote {out.relative_to(ROOT)} ({len(lines)} rows)")


if __name__ == "__main__":
    main()
