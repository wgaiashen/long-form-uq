#!/usr/bin/env python
"""F1 -- ARE HIDDEN STATES AND TOKEN PROBABILITIES REALLY ORTHOGONAL, OR IS THAT JUST NOISE?

Plan: ../PLAN_sharpening_axis.md.   Results: ../STOCKTAKE_sharpening_axis.md §9.

THE QUESTION
------------
STAT 2 measured mean per-example Spearman of ~0.18 between SAPLMA and msp_min, against ~0.71 between
the two probes, and that has been read as "the two signals are near-orthogonal, so a method that uses
both should win". Every attempt to build such a method has failed, and §8 has now shown the oracle
headroom that motivated them was a max-over-K artefact.

⭐ THE CONTROL THAT DECIDES BETWEEN THE TWO EXPLANATIONS (the author's, 2026-08-09):
compare each method's correlation with ITSELF ACROSS SEEDS to its correlation with other methods.

    self-agreement HIGH (~0.7) and cross LOW (~0.18)
        -> two genuinely distinct, STABLE signals. The missing headroom needs another explanation
           (e.g. the second signal is real but adds no INCREMENTAL information about correctness).

    self-agreement LOW (~0.25) and cross LOW (~0.18)
        -> the "orthogonality" is mostly NOISE. There was never anything to harvest, and §8's
           artefact finding is fully explained rather than merely consistent.

A correlation between two measurements cannot exceed the geometric mean of their reliabilities. So
the self-agreement is the CEILING on any cross-method correlation, and reporting a cross-correlation
without it is uninterpretable. We report the raw cross, the ceiling, and the DISATTENUATED value
    r_xy / sqrt(r_xx * r_yy)
which is the correlation the two signals would have if both were measured without noise.

⚠️ SANITY CHECK, ASSERTED: the three floors are deterministic functions of the cached logprobs, so
their seed-to-seed self-agreement MUST be exactly 1.0. If it is not, the sidecars are not what they
claim to be and nothing here is readable.

DATA: results/pdl_perex/<eval>__<rung>__<slug>.npz -- all 42 cells, 3 seeds, per-example uncertainty
for floor_min / floor_ppl / floor_sum / saplma plus the labels, on identical rows, self-gated to 1e-6
against the published PRR. Pure post-hoc read: no GPU, no training, seconds.

    python scripts/checks/orthogonality_map.py
"""
import argparse
import csv as _csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as _st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PEREX = ROOT / "results" / "pdl_perex"
OUT = ROOT / "results" / "orthogonality_map__meta-llama_Meta-Llama-3.1-8B.csv"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
OOD = RUNGS[1:]
DETERMINISTIC = ["floor_min", "floor_ppl", "floor_sum"]


def rho(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(_st.spearmanr(a, b).statistic)


def self_agreement(V):
    """Mean pairwise Spearman between seeds of the SAME method. V: (n_seeds, n_te)."""
    if V.shape[0] < 2:
        return np.nan
    return float(np.nanmean([rho(V[i], V[j]) for i, j in combinations(range(V.shape[0]), 2)]))


def cross_agreement(V, W):
    """Mean Spearman between method A and method B across DIFFERENT seeds.

    Different seeds only, so a shared random draw cannot inflate the cross-correlation. For a
    deterministic method every seed is identical, so this reduces to the ordinary correlation.
    """
    vals = [rho(V[i], W[j]) for i in range(V.shape[0]) for j in range(W.shape[0]) if i != j]
    return float(np.nanmean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print("=" * 100)
    print("F1 -- ORTHOGONALITY vs NOISE: cross-method agreement AGAINST each method's own")
    print("      seed-to-seed reliability, on the canonical grid.")
    print("Population: results/pdl_perex/, 8 long evals x 5 rungs, 3 seeds, per example.")
    print("=" * 100)

    rows, det_bad = [], []
    cells = {}
    for d in LONG:
        for rg in RUNGS:
            p = PEREX / f"{d}__{rg}__meta-llama_Meta-Llama-3.1-8B.npz"
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)
            meth = {k[len("unc__"):]: z[k] for k in z.files
                    if k.startswith("unc__") and k != "unc__fair_floor"}
            if "saplma" not in meth:
                continue
            cells[(d, rg)] = (meth, z["y"])

    if not cells:
        raise SystemExit(f"no per-example sidecars found under {PEREX}")
    print(f"\nCOVERAGE: {len(cells)}/40 canonical cells loaded "
          f"({len({d for d, _ in cells})} evals x {len({r for _, r in cells})} rungs)")
    missing = [(d, r) for d in LONG for r in RUNGS if (d, r) not in cells]
    if missing:
        print(f"⚠️ MISSING, named not silently dropped: {missing}")

    # ---- V1: the determinism assertion ----
    print("\nV1 SANITY: the three floors are deterministic, so seed self-agreement MUST be 1.0")
    for (d, rg), (meth, _) in cells.items():
        for m in DETERMINISTIC:
            if m in meth:
                s = self_agreement(meth[m])
                if not (np.isnan(s) or abs(s - 1.0) < 1e-9):
                    det_bad.append((d, rg, m, s))
    if det_bad:
        raise SystemExit(f"V1 FAIL: deterministic methods vary across seeds: {det_bad[:5]}")
    print("  PASS -- every floor has seed self-agreement exactly 1.0 on every cell.")

    # ---- the headline table ----
    print("\n" + "=" * 100)
    print("SAPLMA's RELIABILITY vs ITS AGREEMENT WITH THE FLOOR")
    print("  self  = SAPLMA vs SAPLMA across seeds (the CEILING on any correlation it can have)")
    print("  cross = SAPLMA vs msp_min, different seeds")
    print("  disatt= cross / sqrt(self * 1.0)   [the floor is deterministic, so its reliability = 1]")
    print("=" * 100)
    print(f"{'eval':15s}{'rung':16s}{'SAPLMA self':>13s}{'x msp_min':>11s}{'x ppl':>8s}{'disatt':>9s}")
    agg = {}
    for d in LONG:
        for rg in RUNGS:
            if (d, rg) not in cells:
                continue
            meth, _ = cells[(d, rg)]
            S = meth["saplma"]
            slf = self_agreement(S)
            xmin = cross_agreement(S, meth["floor_min"])
            xppl = cross_agreement(S, meth["floor_ppl"])
            dis = xmin / np.sqrt(slf) if slf and slf > 0 else np.nan
            print(f"{d:15s}{rg:16s}{slf:>13.3f}{xmin:>11.3f}{xppl:>8.3f}{dis:>9.3f}")
            agg.setdefault(rg, []).append((slf, xmin, xppl, dis))
            rows.append((d, rg, f"{slf:.4f}", f"{xmin:.4f}", f"{xppl:.4f}", f"{dis:.4f}"))

    print("\n" + "-" * 100)
    print(f"{'rung':16s}{'SAPLMA self':>13s}{'x msp_min':>11s}{'x ppl':>8s}{'disatt':>9s}   n")
    for rg in RUNGS:
        if rg not in agg:
            continue
        A = np.array(agg[rg], float)
        print(f"{rg:16s}{np.nanmean(A[:,0]):>13.3f}{np.nanmean(A[:,1]):>11.3f}"
              f"{np.nanmean(A[:,2]):>8.3f}{np.nanmean(A[:,3]):>9.3f}   {len(A)}")
    ALL = np.array([v for rg in agg for v in agg[rg]], float)
    OODA = np.array([v for rg in OOD if rg in agg for v in agg[rg]], float)
    print(f"{'ALL CELLS':16s}{np.nanmean(ALL[:,0]):>13.3f}{np.nanmean(ALL[:,1]):>11.3f}"
          f"{np.nanmean(ALL[:,2]):>8.3f}{np.nanmean(ALL[:,3]):>9.3f}   {len(ALL)}")
    print(f"{'OOD ONLY':16s}{np.nanmean(OODA[:,0]):>13.3f}{np.nanmean(OODA[:,1]):>11.3f}"
          f"{np.nanmean(OODA[:,2]):>8.3f}{np.nanmean(OODA[:,3]):>9.3f}   {len(OODA)}")

    # ---- the verdict ----
    slf_m, xmin_m = float(np.nanmean(OODA[:, 0])), float(np.nanmean(OODA[:, 1]))
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  SAPLMA seed-to-seed self-agreement (OOD): {slf_m:.3f}")
    print(f"  SAPLMA vs msp_min                  (OOD): {xmin_m:.3f}")
    if slf_m < 0.4:
        print("\n  ⛔ SELF-AGREEMENT IS LOW. SAPLMA barely agrees with ITSELF across seeds, so a low")
        print("     correlation with the floor is what noise looks like, not evidence of a second")
        print("     signal. The 'orthogonality' reading is NOT supported, and §8's missing headroom")
        print("     is fully explained: there was never a stable second signal to harvest.")
    elif slf_m > 2.5 * abs(xmin_m):
        print("\n  ✅ SELF-AGREEMENT IS HIGH relative to the cross-correlation. The two are genuinely")
        print("     distinct, STABLE signals -- so the missing headroom in §8 needs a DIFFERENT")
        print("     explanation (most likely: the second signal is real but adds no INCREMENTAL")
        print("     information about correctness once the first is known).")
    else:
        print("\n  ~ INTERMEDIATE. Report both numbers and draw no strong conclusion.")
    print("\n  ⚠️ A cross-correlation cannot exceed sqrt(reliability_x * reliability_y). Any future")
    print("     quote of 'the probe and the floor correlate only 0.18' MUST carry the ceiling.")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["eval", "rung", "saplma_self_seed", "x_msp_min", "x_perplexity", "disattenuated"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
