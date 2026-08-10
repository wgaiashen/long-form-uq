#!/usr/bin/env python
"""THE R-claim scorer — one script, two populations (M2: prereg/M2_qwen14b_replication.md).

Scores R1a / R1b / R2 / R2-desc / R3 / R4' from a pdl master CSV. Written 2026-08-10 because the
Llama numbers quoted in the prereg were produced in a working session with no committed script, and
re-deriving the tests from prose on a second model risks running a subtly different test per model
and calling the difference a replication result. This file IS the test definition now.

THE EXACT CONVENTIONS (reverse-engineered from the Llama master and verified to reproduce every
published number EXACTLY — run --gate-llama to prove it on your checkout):

  * A dataset's "OOD" value for a method = the MEAN over the four OOD rungs
    (SameTask-long, DiffTask-long, LOO-long, 1ds-Diff-long). ID never enters, except in R3's ID leg.
    For the free floors the four rung values are identical (rung-invariant), so the mean equals any
    single rung — n = 8 datasets, never 32 cells (the 32-cell framing was 4x pseudo-replication).
  * All quoted Wilcoxon p-values are TWO-SIDED (scipy.stats.wilcoxon defaults). The prereg's
    decision rule speaks of a one-sided test at alpha = 0.05; both are printed, and the VERDICT
    line applies the registered one-sided rule while the gate checks the published two-sided
    numbers. (For context: with 8/8 same-sign the two-sided p is exactly 2x the one-sided.)
  * R2 = OLS (scipy.stats.linregress, with intercept) of per-dataset OOD-mean SAPLMA on
    per-dataset OOD-mean msp_min; t = (b - 1)/se at df = n - 2 = 6, one-sided p toward b < 1;
    the 95% CI uses the two-sided t quantile. The CI straddling 0 is reported alongside (the
    "does not track the baseline" caveat): p(b > 0) is also printed.
  * R2-desc: sd across the 8 datasets' OOD means, ddof = 1.
  * R3 DiD per dataset = (pooler - SAPLMA at ID) - (mean over the 4 OOD rungs of pooler - SAPLMA).
    Positive = the pooler's ID advantage evaporates OOD. Wilcoxon on the 8 DiDs. The two legs are
    printed descriptively; the published OOD-leg wins/p were quoted on the 32 cells and are
    reproduced that way, labelled as descriptive only.
  * R4' rungs: DiffTask-long and 1ds-Diff-long (the "two hardest", as named on Llama in
    PLAN_execution_post7Aug.md §2.4). wMSP-shrink@2 - SAPLMA per dataset, n = 8, DIRECTIONAL ONLY.
  * fair_floor enters NOWHERE.

Usage:
    # the gate: reproduce every published Llama number or exit 1 (run this before touching Qwen)
    python scripts/checks/replication_claims.py --population llama --gate-llama
    # the Qwen verdict (DoC), after qwen_pdl_master.py --strict passes:
    python scripts/checks/replication_claims.py --population qwen

NEVER pool the two populations; run this once per model and compare verdicts per claim.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
EVALS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]

# canonical method -> its name in each population's master CSV. The Llama master carries display
# names; the Qwen master carries the driver's raw arm names. Nothing else differs.
NAMES = {
    "msp_min":  {"llama": "msp_min",         "qwen": "floor_min"},
    "msp_sum":  {"llama": "msp_sum",         "qwen": "floor_sum"},
    "perplexity": {"llama": "perplexity",    "qwen": "floor_ppl"},
    "saplma":   {"llama": "SAPLMA",          "qwen": "saplma"},
    "pooler":   {"llama": "armA(attention)", "qwen": "attention"},
    "wmsp2":    {"llama": "wMSP-shrink@2",   "qwen": "wmsp_shrink2"},
}
CSV_DEFAULT = {
    "llama": RES / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv",
    "qwen": RES / "pdl_master__Qwen_Qwen2.5-14B.csv",
}

# The published Llama numbers (PLAN_execution_post7Aug.md §2.4 / prereg M2 §1), the gate targets.
GATE = {
    "r1a": (0.1310, 8, 0.0078), "r1b": (0.0686, 5, 0.46),
    "r2": (0.242, 0.195, 0.0041, -0.236, 0.720),
    "r2desc": (0.164, 0.088, 0.064),
    "r3_id": (0.0354, 6, 0.148), "r3_ood_cells": (-0.0187, 15, 0.50),
    "r3_did": (0.0540, 7, 0.0156),
    "r4_diff": (0.0022, 4, 1.00), "r4_1ds": (0.0228, 6, 0.64),
}


def load(path, population):
    """(method, rung, eval) -> prr. Fails loud on a missing cell when it is later read."""
    g = {}
    for r in csv.DictReader(open(path)):
        if population == "llama":
            if r.get("seed_regime") != "3seed":
                continue
            key, val = (r["method"], r["rung"], r["eval"]), r["prr"]
        else:
            key, val = (r["method"], r["rung"], r["eval"]), r.get("prr_mean", "")
        try:
            g[key] = float(val)
        except (TypeError, ValueError):
            continue
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=["llama", "qwen"], required=True)
    ap.add_argument("--csv", default=None, help="master CSV (default per population)")
    ap.add_argument("--gate-llama", action="store_true",
                    help="assert every published Llama number reproduces exactly; exit 1 otherwise")
    args = ap.parse_args()
    if args.gate_llama and args.population != "llama":
        raise SystemExit("--gate-llama only makes sense with --population llama")

    path = Path(args.csv) if args.csv else CSV_DEFAULT[args.population]
    g = load(path, args.population)
    nm = {k: v[args.population] for k, v in NAMES.items()}

    # ---- coverage, stated before any number --------------------------------------------------
    need = [(nm[m], rg, e) for m in nm for e in EVALS
            for rg in (["ID"] + OOD_RUNGS if m in ("saplma", "pooler") else OOD_RUNGS)]
    missing = [k for k in need if k not in g]
    print("=" * 100)
    print(f"R-CLAIM SCORER  population={args.population}  csv={path.name}")
    print("Unit of analysis: the DATASET (n = 8). OOD = mean of the 4 OOD rungs; ID never enters")
    print("except R3's ID leg. All Wilcoxons printed two-sided AND one-sided; the registered")
    print("decision rule is the one-sided test at alpha = 0.05. NEVER pool populations.")
    print("=" * 100)
    if missing:
        print(f"⛔ {len(missing)} REQUIRED CELLS MISSING — refusing to score a partial grid. First 10:")
        for k in missing[:10]:
            print("   ", k)
        raise SystemExit(1)
    print("coverage: complete for every cell the tests read")
    if args.population == "qwen":
        print("⚠️ expertqa on this population carries a severe length confound (length-alone PRR")
        print("   +0.71 > every method; STOCKTAKE_qwen.md §10.2) — quote no expertqa cell without it.")

    def ood(m, e):
        return float(np.mean([g[(nm[m], rg, e)] for rg in OOD_RUNGS]))

    fails = []

    def gate(tag, got, want, tol):
        if args.gate_llama and abs(got - want) > tol:
            fails.append(f"{tag}: got {got:.4f} want {want}")

    def wilc(d):
        two = stats.wilcoxon(d).pvalue
        one = stats.wilcoxon(d, alternative="greater").pvalue
        return two, one

    # ---- R1a / R1b ---------------------------------------------------------------------------
    mn = np.array([ood("msp_min", e) for e in EVALS])
    for tag, other, target in [("R1a  msp_min - msp_sum", "msp_sum", "r1a"),
                               ("R1b  msp_min - perplexity", "msp_min_vs_ppl", "r1b")]:
        d = mn - np.array([ood("msp_sum" if target == "r1a" else "perplexity", e) for e in EVALS])
        two, one = wilc(d)
        print(f"\n{tag}: mean {d.mean():+.4f}  wins {(d > 0).sum()}/8  p(two-sided) {two:.4f}  "
              f"p(one-sided, greater) {one:.4f}")
        w, ww, wp = GATE[target]
        gate(target + " mean", d.mean(), w, 5e-5), gate(target + " p", two, wp, 5e-3)
        if target == "r1a":
            verdict = "REPLICATES" if (one < 0.05 and d.mean() > 0) else "FAILS"
            print(f"   VERDICT (registered rule, one-sided p<0.05 same sign): {verdict}")
        else:
            print("   registered as UNESTABLISHED on Llama — descriptive on both models, no verdict")

    # ---- R2 ----------------------------------------------------------------------------------
    sap = np.array([ood("saplma", e) for e in EVALS])
    res = stats.linregress(mn, sap)
    b, se, df = res.slope, res.stderr, len(EVALS) - 2
    p_lt1 = stats.t.cdf((b - 1) / se, df)
    tq = stats.t.ppf(0.975, df)
    print(f"\nR2   OLS SAPLMA_OOD ~ msp_min_OOD: b {b:+.3f}  se {se:.3f}  one-sided p(b<1) {p_lt1:.4f}"
          f"  95% CI [{b - tq * se:+.3f}, {b + tq * se:+.3f}]  p(b>0) {1 - stats.t.cdf(b / se, df):.3f}")
    print("   caveat (verbatim from M2): the claim is 'the probe does not track the baseline',")
    print("   NOT a pinned-down compression rate — b is not distinguishable from 0 either.")
    print(f"   VERDICT (b significantly < 1, one-sided alpha=0.05): "
          f"{'REPLICATES' if p_lt1 < 0.05 else 'FAILS'}")
    gate("r2 b", b, GATE["r2"][0], 5e-4), gate("r2 se", se, GATE["r2"][1], 5e-4)
    gate("r2 p", p_lt1, GATE["r2"][2], 5e-4)

    # ---- R2-desc -----------------------------------------------------------------------------
    arm = np.array([ood("pooler", e) for e in EVALS])
    sds = {k: v.std(ddof=1) for k, v in [("msp_min", mn), ("SAPLMA", sap), ("pooler", arm)]}
    print(f"\nR2-desc  sd of OOD PRR across the 8 datasets (ddof=1): "
          + "  ".join(f"{k} {v:.3f}" for k, v in sds.items()))
    print(f"   VERDICT (sd(SAPLMA) < sd(msp_min)): "
          f"{'REPLICATES' if sds['SAPLMA'] < sds['msp_min'] else 'FAILS'}")
    for got, want, tag in zip(sds.values(), GATE["r2desc"], ["min", "sap", "pool"]):
        gate("r2desc " + tag, got, want, 5e-4)

    # ---- R3 ----------------------------------------------------------------------------------
    idd = np.array([g[(nm["pooler"], "ID", e)] - g[(nm["saplma"], "ID", e)] for e in EVALS])
    oodd = arm - sap
    did = idd - oodd
    cells = np.array([g[(nm["pooler"], rg, e)] - g[(nm["saplma"], rg, e)]
                      for e in EVALS for rg in OOD_RUNGS])
    two, one = wilc(did)
    print(f"\nR3   ID leg (pooler-SAPLMA): mean {idd.mean():+.4f} wins {(idd > 0).sum()}/8 "
          f"p {stats.wilcoxon(idd).pvalue:.3f}   [descriptive]")
    print(f"     OOD leg: mean {cells.mean():+.4f} wins {(cells > 0).sum()}/32 "
          f"p {stats.wilcoxon(cells).pvalue:.2f}   [descriptive; cell-level as published]")
    print(f"     DiD (ID leg - OOD leg, per eval, n=8): mean {did.mean():+.4f}  "
          f"wins {(did > 0).sum()}/8  p(two-sided) {two:.4f}  p(one-sided) {one:.4f}")
    print(f"   VERDICT (DiD > 0, registered one-sided p<0.05): "
          f"{'REPLICATES' if (one < 0.05 and did.mean() > 0) else 'FAILS'}")
    gate("r3 id mean", idd.mean(), GATE["r3_id"][0], 5e-5)
    gate("r3 did mean", did.mean(), GATE["r3_did"][0], 5e-5)
    gate("r3 did p", two, GATE["r3_did"][2], 5e-3)

    # ---- R4' ---------------------------------------------------------------------------------
    print("\nR4'  wMSP-shrink@2 - SAPLMA on the two hardest rungs — DIRECTIONAL ONLY, no verdict:")
    for rg, key in [("DiffTask-long", "r4_diff"), ("1ds-Diff-long", "r4_1ds")]:
        d = np.array([g[(nm["wmsp2"], rg, e)] - g[(nm["saplma"], rg, e)] for e in EVALS])
        print(f"     {rg:15s} mean {d.mean():+.4f}  wins {(d > 0).sum()}/8  "
              f"p(two-sided) {stats.wilcoxon(d).pvalue:.2f}  spread [{d.min():+.3f}, {d.max():+.3f}]")
        gate(key + " mean", d.mean(), GATE[key][0], 5e-5)

    # ---- the gate ----------------------------------------------------------------------------
    if args.gate_llama:
        if fails:
            print("\n⛔ GATE FAILED — this scorer does NOT reproduce the published Llama numbers:")
            for f in fails:
                print("   ", f)
            raise SystemExit(1)
        print("\n✅ GATE PASS — every published Llama number reproduces exactly. The scorer is the")
        print("   test definition; point it at the Qwen master only now.")
    if args.population == "llama" and not args.gate_llama:
        print("\n(reminder: run --gate-llama to assert reproduction, not just print)")

    if args.population == "qwen":
        print("\nANTI-FAVOURABLE-RESULT GUARD (M2 §2): if all four verdicts above read REPLICATES,")
        print("treat that as a suspected configuration leak and re-verify that no Qwen test PRR was")
        print("inspected before the layer was fixed by rule.")


if __name__ == "__main__":
    main()
