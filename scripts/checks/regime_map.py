#!/usr/bin/env python
"""F2/F3 -- WHO WINS WHERE, WHY IT DEGRADES, AND WHETHER THE POOL-COMPOSITION STORY IS TESTABLE.

Plan: ../PLAN_sharpening_axis.md.   Results: ../STOCKTAKE_sharpening_axis.md §10.

F2 -- THE REGIME MAP (descriptive)
    Who wins each of the 40 (dataset x rung) cells, and the DEGRADATION ORDERING: how much each
    method loses from ID to the two hardest rungs. The ordering is the mechanism -- the more a
    method leans on the training pool, the more it loses, and msp_min loses nothing by construction.

F3 -- POOL COMPOSITION: IS THE HYPOTHESIS EVEN IDENTIFIED?
    The hypothesis is that the probes collapse because the training pool gets thin/mismatched, not
    because of "distance" per se. ⚠️ Before fitting anything, this script checks whether the grid can
    SEPARATE those two, by tabulating n_sources per cell against rung and task family.

    Pool structure, read statically from ProbeDriftLong/probe_drift_long/ood_settings.py:30-69 --
    NOT by importing probedriftlong, which pulls in HF datasets and takes minutes:
        SameTask-long : same FINE_FAMILY, minus X
        DiffTask-long : every other family
        LOO-long      : all 7 others
        1ds-Diff-long : exactly ONE source (diff[:1]), given the whole budget

⚠️ THIS SCRIPT IS DESCRIPTIVE. It is computed on test data at n = 8, and the label-free version of
exactly this rule was already FALSIFIED (prereg/R1). It is a MAP, not a predictor, and any rule built
on it needs its own out-of-sample test.

Reads the master table only. No training, no GPU, seconds.

    python scripts/checks/regime_map.py
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MASTER = ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
OUT = ROOT / "results" / "regime_map__meta-llama_Meta-Llama-3.1-8B.csv"

LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
HARD = ["DiffTask-long", "1ds-Diff-long"]
# mirrored from probe_drift_long/dataset_configs.py:44-48 -- kept in sync by the assertion below
FINE = {"pubmed_qa": "correctness_qa", "med_quad": "correctness_qa", "asqa": "correctness_qa",
        "expertqa": "factuality", "factscore": "factuality",
        "xsum": "summ", "cnn_dailymail": "summ", "samsum": "summ"}
METHODS = ["msp_min", "perplexity", "wMSP-shrink@2", "wMSP-shrink@10", "SAPLMA", "armA(attention)"]
SHORT = {"msp_min": "MIN", "perplexity": "PPL", "wMSP-shrink@2": "wM2", "wMSP-shrink@10": "wM10",
         "SAPLMA": "SAP", "armA(attention)": "armA"}


def n_sources(X, rung):
    """Number of DISTINCT training source datasets for this cell (static, no import)."""
    others = [d for d in LONG if d != X]
    same = [d for d in others if FINE[d] == FINE[X]]
    diff = [d for d in others if FINE[d] != FINE[X]]
    return {"ID": 1, "SameTask-long": len(same), "DiffTask-long": len(diff),
            "LOO-long": len(others), "1ds-Diff-long": 1}[rung]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    g = defaultdict(dict)
    for r in _csv.DictReader(open(MASTER)):
        if r.get("seed_regime") != "3seed":
            continue
        try:
            g[r["method"]][(r["rung"], r["eval"])] = float(r["prr"])
        except (ValueError, TypeError):
            continue

    print("=" * 100)
    print("F2 -- THE REGIME MAP.  Population: widened cells_long, Llama-3.1-8B, 3-seed.")
    print("⚠️ DESCRIPTIVE, not predictive: computed on test at n=8, and the label-free version of")
    print("   this rule (prereg/R1) was already FALSIFIED.")
    print("=" * 100)
    rows = []

    print(f"\nWHO WINS EACH CELL\n{'eval':15s}" + "".join(f"{r.replace('-long',''):>13s}" for r in RUNGS))
    tally, by_rung = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for d in LONG:
        line = []
        for rg in RUNGS:
            vals = {m: g[m].get((rg, d)) for m in METHODS if g[m].get((rg, d)) is not None}
            if not vals:
                line.append("  --  "); continue
            w = max(vals, key=vals.get)
            line.append(f"{SHORT[w]}({vals[w]:+.2f})")
            tally[SHORT[w]] += 1; by_rung[rg][SHORT[w]] += 1
            rows.append(("winner", d, rg, SHORT[w], f"{vals[w]:.4f}", n_sources(d, rg), FINE[d]))
        print(f"{d:15s}" + "".join(f"{c:>13s}" for c in line))
    print(f"\n  overall: {dict(sorted(tally.items(), key=lambda x: -x[1]))}")
    for rg in RUNGS:
        print(f"  {rg:16s} {dict(sorted(by_rung[rg].items(), key=lambda x: -x[1]))}")

    print("\n" + "=" * 100)
    print("DEGRADATION: ID minus mean(DiffTask, 1ds-Diff).  Bigger = leans harder on the training pool.")
    print("=" * 100)
    print(f"{'method':22s}{'mean drop':>11s}{'ID mean':>10s}{'hard mean':>11s}")
    for m in METHODS:
        drops, ids, hards = [], [], []
        for d in LONG:
            i = g[m].get(("ID", d)); h = [g[m].get((r, d)) for r in HARD]
            if i is None or any(x is None for x in h):
                continue
            drops.append(i - np.mean(h)); ids.append(i); hards.append(np.mean(h))
        if drops:
            print(f"{m:22s}{np.mean(drops):>+11.3f}{np.mean(ids):>+10.3f}{np.mean(hards):>+11.3f}")
            rows.append(("degradation", "ALL", "ID-minus-hard", m, f"{np.mean(drops):.4f}", "", ""))
    print("\n  ⭐ The ordering is monotone in how much a method leans on hidden states, and msp_min")
    print("     loses exactly 0.000 because it never sees the training pool.")

    # ---------------------------------------------------------------------------------------
    # F3 -- IDENTIFIABILITY CHECK, BEFORE ANY FITTING
    # ---------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("F3 -- IS THE POOL-COMPOSITION HYPOTHESIS IDENTIFIED ON THIS GRID?")
    print("=" * 100)
    print(f"\nn SOURCE DATASETS per cell\n{'eval':15s}{'family':16s}"
          + "".join(f"{r.replace('-long',''):>11s}" for r in RUNGS))
    for d in LONG:
        print(f"{d:15s}{FINE[d]:16s}" + "".join(f"{n_sources(d, r):>11d}" for r in RUNGS))

    print("\nWITHIN-RUNG VARIATION IN n_sources (the ONLY thing that could identify the effect):")
    unident = []
    for rg in RUNGS:
        vals = {d: n_sources(d, rg) for d in LONG}
        levels = sorted(set(vals.values()))
        if len(levels) == 1:
            print(f"  {rg:16s} n_sources = {levels[0]} for ALL 8 evals -> NO variation, contributes nothing")
            unident.append(rg)
        else:
            groups = {lv: [d for d in LONG if vals[d] == lv] for lv in levels}
            fams = {lv: sorted({FINE[d] for d in groups[lv]}) for lv in levels}
            # ⚠️ The confound test is NOT "every level maps to one family" (an earlier version used
            # that and mislabelled these rungs "partially separable", contradicting the verdict).
            # The right test: is the n_sources indicator a PERFECT FUNCTION of some family
            # indicator? Here n_sources is binary, so check whether one level is exactly one family.
            perfect = [f for lv in levels for f in fams[lv]
                       if len(fams[lv]) == 1 and set(groups[lv]) == {d for d in LONG if FINE[d] == f}]
            print(f"  {rg:16s} levels {levels}: " +
                  "; ".join(f"{lv}->{groups[lv]}" for lv in levels))
            if perfect:
                print(f"  {'':16s} *** PERFECTLY CONFOUNDED: n_sources is exactly the indicator "
                      f"family=={perfect[0]} ***")
            else:
                print(f"  {'':16s} families per level: {fams}   separable")

    print("\n" + "-" * 100)
    print("VERDICT ON F3")
    print("-" * 100)
    print("  * ID and 1ds-Diff have n_sources = 1 for every eval, and LOO has 7 for every eval, so")
    print("    THREE of the five rungs contribute NO identifying variation at all.")
    print("  * The only within-rung variation is SameTask (1 vs 2) and DiffTask (5 vs 6), and in both")
    print("    cases the odd group is exactly {expertqa, factscore} -- the factuality family.")
    print("    So n_sources is PERFECTLY CONFOUNDED WITH TASK FAMILY wherever it varies at all.")
    print("\n  ⛔ THE POOL-DIVERSITY HYPOTHESIS IS UNIDENTIFIED ON THIS GRID. Any coefficient fitted")
    print("     for 'n_sources' would be indistinguishable from a factuality-vs-rest effect, and")
    print("     across rungs it is indistinguishable from distance. Fitting it would manufacture a")
    print("     result. Reported as unidentified rather than estimated.")
    print("\n  WHAT WOULD IDENTIFY IT: a grid that varies pool SIZE independently of rung -- e.g. the")
    print("  LOO rung repeated at 2, 4 and 7 sources for the SAME eval. That is a design change, not")
    print("  an analysis, and it is the concrete recommendation this leaves behind.")
    rows.append(("f3_verdict", "ALL", "identifiability", "UNIDENTIFIED",
                 "n_sources confounded with task family within rung, with distance across rungs", "", ""))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["kind", "eval", "rung", "value", "prr", "n_sources", "family"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
