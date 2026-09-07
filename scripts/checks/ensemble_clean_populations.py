#!/usr/bin/env python
"""The combination diagnostic, recomputed on the corrected-span populations that the main
comparison table reports.

WHY THIS FILE EXISTS. The earlier combination analysis was run before the answer-span correction,
on per-example score files that predate it. Its numbers therefore describe a population that no
longer appears anywhere in the report. This recomputes the same quantities, with the same combiner
and the same seed convention, from the certified corrected-span score files, so the combination
result and the main comparison table describe one population.

WHICH POPULATIONS. Every population listed in POPULATIONS below, which is every one whose
corrected-span per-example scores exist and have been certified. That is currently two of the three
the main comparison table covers. Qwen2.5-14B is absent because its only complete per-example score
directory is on the uncorrected answer span, provably so: the probability floors, which are
deterministic functions of the cached token log-probabilities and cannot be moved by any refit,
disagree with its scored ladder on every cell of the four datasets whose answer span was corrected,
while agreeing to 5e-05 on the other four. Rebuilding it needs a fresh ladder run, which was not
done. A population is added here only after certification, never because a directory exists.

WHAT IS AND IS NOT RECOMPUTED. Nothing is trained and nothing is refitted. Every component score is
read from the stored per-example vectors that the ladder wrote, and each component is checked cell by
cell against the consolidated master before any combination is formed. A population whose components
do not reproduce the master is refused rather than reported with a caveat.

THE COMBINER. Equal-weight average of within-cohort ranks, formed inside the seed loop and scored per
seed, with the per-seed values then averaged. This is the same convention the ladder uses. The rank
average depends on which responses share a test cohort, so it is a diagnostic of whether the two
signals carry different information, not a deployable per-response estimator.

    python scripts/checks/ensemble_clean_populations.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import results                                              # noqa: E402
import complementary_ensemble as ce                                  # noqa: E402

LONG = ce.LONG
RUNGS = ce.RUNGS
OOD = ce.OOD

# Implementation key in the stored files, and the name used in the report.
COMPONENTS = {"saplma": "SAPLMA", "wmsp_shrink2": "CAWSA lambda=2", "attention": "attention pooling"}

# The populations this may run on: the corrected-span score directory and the table the components
# are checked against. A population is listed here only after its score directory has been certified.
POPULATIONS = [
    ("meta-llama/Meta-Llama-3.1-8B", "meta-llama_Meta-Llama-3.1-8B",
     "results/perex_clean__meta-llama_Meta-Llama-3.1-8B"),
    ("google/gemma-2-9b", "google_gemma-2-9b",
     "results/perex_clean__google_gemma-2-9b"),
]

GATE_TOL = 1e-3     # the masters store four decimals, so this is the resolution floor
OUT = ROOT / "results/analysis/ensemble_clean"


def prr_seedmean(y, V):
    """Mean over seeds of the per-seed rejection ratio. Never the ratio of a seed-averaged vector."""
    return float(np.mean([results.prr(y, V[s]) for s in range(V.shape[0])]))


def prr_rankavg(y, A, B):
    """The combination, formed inside the seed loop so seed noise is not averaged away first."""
    return float(np.mean([results.prr(y, ce.rankavg(A[s], B[s])) for s in range(A.shape[0])]))


def gate(cells, master, slug):
    """Every component, every cell, against the consolidated master. Refuses on any excursion."""
    m = pd.read_csv(master).set_index(["method", "eval", "rung"])["prr_mean"]
    rows, worst = [], 0.0
    for key, name in COMPONENTS.items():
        for (d, rg), (meth, y) in sorted(cells.items()):
            got = prr_seedmean(y, meth[key])
            ref = float(m.loc[(key, d, rg)])
            rows.append((name, d, rg, got, ref, abs(got - ref)))
            worst = max(worst, abs(got - ref))
    g = pd.DataFrame(rows, columns=["component", "eval", "rung", "recomputed", "master", "abs_diff"])
    bad = g[g.abs_diff > GATE_TOL]
    print(f"  component gate: {len(g)} cells, max abs difference {worst:.3e}, tolerance {GATE_TOL}")
    for name in COMPONENTS.values():
        s = g[g.component == name]
        print(f"    {name:<20} {len(s):3d} cells  max {s.abs_diff.max():.3e}")
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit(f"component gate FAILED for {slug}: {len(bad)} cells beyond tolerance")
    print("  component gate: PASS")
    return g


def ladder(cells):
    """PRR per method per cell, for the five single methods and the two combinations."""
    out = []
    for (d, rg), (meth, y) in sorted(cells.items()):
        vals = {
            "SAPLMA": prr_seedmean(y, meth["saplma"]),
            "CAWSA lambda=2": prr_seedmean(y, meth["wmsp_shrink2"]),
            "attention pooling": prr_seedmean(y, meth["attention"]),
            "CAWSA lambda=2 + SAPLMA": prr_rankavg(y, meth["wmsp_shrink2"], meth["saplma"]),
            "SAPLMA + attention pooling": prr_rankavg(y, meth["saplma"], meth["attention"]),
        }
        for k, v in vals.items():
            out.append({"eval": d, "rung": rg, "method": k, "prr": v})
    return pd.DataFrame(out)


def wilcoxon_exact(d):
    """Exact two-sided signed-rank test. n is 8, so the exact distribution is enumerable."""
    from scipy.stats import wilcoxon
    d = np.asarray([x for x in d if x != 0.0], float)
    if d.size == 0:
        return float("nan"), 0
    return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox",
                          mode="exact").pvalue), int(d.size)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_ladders, all_targets, summary = [], [], []

    for model, slug, perex in POPULATIONS:
        print(f"\n{'=' * 92}\n{model}\n  scores  {perex}")
        master = ROOT / f"results/hybrids/pdl_consolidated_master__{slug}.csv"
        print(f"  table   {master.relative_to(ROOT)}")
        cells, missing = ce.load_cells(ROOT / perex, slug)
        print(f"  coverage {len(cells)} of 40 cells" + (f", missing {missing}" if missing else ""))
        if missing:
            raise SystemExit(f"{slug}: incomplete grid, refusing to report a partial ladder")

        g = gate(cells, master, slug)
        g.insert(0, "model", model)
        g.to_csv(OUT / f"component_gate__{slug}.csv", index=False)

        lad = ladder(cells)
        lad.insert(0, "model", model)
        all_ladders.append(lad)

        # Macro over the eight datasets, per setting.
        piv = lad.pivot_table(index="method", columns="rung", values="prr", aggfunc="mean")[RUNGS]
        piv["mean_ood"] = piv[OOD].mean(axis=1)
        base = piv.loc["SAPLMA", "mean_ood"]
        ctrl = piv.loc["SAPLMA + attention pooling", "mean_ood"]
        piv["delta_vs_saplma"] = piv["mean_ood"] - base
        piv["delta_vs_saplma_attention"] = piv["mean_ood"] - ctrl
        piv.insert(0, "model", model)
        summary.append(piv.reset_index())
        print("\n  macro rejection ratio over the eight datasets")
        print(piv.drop(columns="model").round(4).to_string())

        # Target level, for the combination against the probe alone.
        t = lad[lad.rung.isin(OOD)].pivot_table(index=["eval", "method"], values="prr", aggfunc="mean")
        t = t.reset_index().pivot(index="eval", columns="method", values="prr").loc[LONG]
        tt = pd.DataFrame({
            "model": model,
            "target": t.index,
            "cawsa_saplma_mean_ood": t["CAWSA lambda=2 + SAPLMA"].values,
            "saplma_mean_ood": t["SAPLMA"].values,
            "saplma_attention_mean_ood": t["SAPLMA + attention pooling"].values,
        })
        tt["delta_vs_saplma"] = tt.cawsa_saplma_mean_ood - tt.saplma_mean_ood
        tt["delta_vs_saplma_attention"] = tt.cawsa_saplma_mean_ood - tt.saplma_attention_mean_ood
        all_targets.append(tt)
        p, n = wilcoxon_exact(tt.delta_vs_saplma.values)
        print(f"\n  target level, combination minus probe: improved on "
              f"{int((tt.delta_vs_saplma > 0).sum())} of 8, mean {tt.delta_vs_saplma.mean():+.4f}, "
              f"exact two-sided p = {p:.4f} (n = {n})")
        print(tt.drop(columns="model").round(4).to_string(index=False))

    lad = pd.concat(all_ladders, ignore_index=True)
    lad.to_csv(OUT / "ensemble_clean_cells.csv", index=False)
    tgt = pd.concat(all_targets, ignore_index=True)
    tgt.to_csv(OUT / "ensemble_clean_targets.csv", index=False)
    sm = pd.concat(summary, ignore_index=True)
    sm.to_csv(OUT / "ensemble_clean_rungs.csv", index=False)

    # Equal weight per population, never per cell, so a population with more cells cannot dominate.
    print(f"\n{'=' * 92}\nequal weight per population, {len(POPULATIONS)} of them "
          f"({', '.join(m for m, _, _ in POPULATIONS)})")
    eq = sm.groupby("method")[RUNGS + ["mean_ood", "delta_vs_saplma",
                                       "delta_vs_saplma_attention"]].mean()
    print(eq.round(4).to_string())
    eq.to_csv(OUT / "ensemble_clean_equalmodel.csv")
    print(f"\nwrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
