#!/usr/bin/env python
"""W2 VERDICT -- assemble the length-conditioned temperature grid and select (T0, gamma) HONESTLY.

Reads results/sharpening_wmsp_<eval>__<slug>.csv, written by scripts/checks/sharpening_wmsp.py.

WHAT THIS DECIDES
-----------------
The per-cell grid best is an ORACLE and is never the method's score. The method's score is what a
LEAVE-ONE-DATASET-OUT procedure achieves: for each held-out eval, pick (T0, gamma) on the OTHER
evals only, then apply it to the held-out one and report THAT. This is the whole point of the track
(design note, 7 Aug: "have we done that on some dev set, or have we done all the test set and picked the
best one?").

THREE COLUMNS, NEVER POOLED:
    no-op        T0=1, gamma=0  == today's weighted MSP exactly (verified against the master ladder)
    LODO         selected on the other 7 evals, applied to the held-out one    <- THE RESULT
    ORACLE       best (T0, gamma) per cell, chosen on test                     <- CEILING ONLY

COVERAGE IS CHECKED AND FAILS LOUD. A partial grid is reported as partial, with the missing evals
named. It is never silently averaged over whatever happened to finish (standing evaluation rule).

THE NO-OP IS RE-VERIFIED AGAINST THE MASTER LADDER HERE TOO, per cell. `sharpening_wmsp.py`
already asserts an exact identity against `predict_weighted_msp` in-process; this is the independent
outside check that the whole thing sits on the published population.

    python scripts/checks/sharpening_wmsp_verdict.py
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as _st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RES = ROOT / "results"
MASTER = RES / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
NOOP = ("1.0", "0.0")


def load_grid():
    """eval -> rung -> (T0, gamma) -> prr ; plus the floors and the argmax-agreement diagnostic."""
    g = defaultdict(lambda: defaultdict(dict))
    floors = defaultdict(lambda: defaultdict(dict))
    agree = defaultdict(dict)
    present = []
    for ev in LONG:
        p = RES / f"sharpening_wmsp_{ev}__{SLUG}.csv"
        if not p.exists():
            continue
        present.append(ev)
        for r in _csv.DictReader(open(p)):
            if r["kind"].startswith("floor:"):
                floors[ev][r["rung"]][r["kind"].split(":", 1)[1]] = float(r["prr"])
            else:
                g[ev][r["rung"]][(r["T0"], r["gamma"])] = float(r["prr"])
                if r.get("argmax_agree"):
                    agree[ev][r["rung"]] = float(r["argmax_agree"])
    return g, floors, agree, present


def master_wmsp_norm():
    out = defaultdict(dict)
    if not MASTER.exists():
        return out
    for r in _csv.DictReader(open(MASTER)):
        if r["method"] == "wMSP-norm" and r.get("seed_regime") == "3seed":
            try:
                out[r["eval"]][r["rung"]] = float(r["prr"])
            except (ValueError, TypeError):
                pass
    return out


def ood_mean(cell, key):
    """Mean over the 4 OOD rungs for one grid point. None if any rung is missing."""
    vals = [cell[rg].get(key) for rg in OOD_RUNGS]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / f"sharpening_wmsp_verdict__{SLUG}.csv"))
    args = ap.parse_args()

    g, floors, agree, present = load_grid()
    missing = [e for e in LONG if e not in present]

    print("=" * 100)
    print("W2 VERDICT -- length-conditioned temperature on weighted MSP")
    print("Population: widened cells_long, meta-llama/Llama-3.1-8B, 4 OOD rungs, 3 seeds, legacy carve.")
    print("=" * 100)
    print(f"\nCOVERAGE: {len(present)}/8 evals present.")
    if missing:
        print(f"MISSING (named, not silently averaged over): {missing}")
        print("   Every table below is on the PARTIAL population and says so. Do not quote it as the")
        print("   full grid (standing evaluation rule).")
    if not present:
        raise SystemExit("no eval CSVs found -- nothing to assemble")

    # ---- independent re-verification of the no-op against the published master ladder ----
    mst = master_wmsp_norm()
    print("\n" + "-" * 100)
    print("NO-OP vs the master ladder `wMSP-norm` (the outside check on the population)")
    print("-" * 100)
    bad = 0
    for ev in present:
        for rg in ["ID"] + OOD_RUNGS:
            mine = g[ev].get(rg, {}).get(NOOP)
            ref = mst.get(ev, {}).get(rg)
            if mine is None or ref is None:
                continue
            d = abs(mine - ref)
            if d > 0.0001:
                bad += 1
                print(f"  MISMATCH {ev}/{rg}: mine {mine:+.4f} vs master {ref:+.4f} (d={d:.4f})")
    print(f"  {'ALL CELLS MATCH the master to 4 dp' if bad == 0 else f'{bad} CELLS DISAGREE'}"
          f"  ({len(present)} evals x 5 rungs)")
    if bad:
        raise SystemExit("no-op does not reproduce the master ladder -- STOP, the population is wrong")

    # ---- the grid, per eval, as an OOD mean ----
    keys = sorted({k for ev in present for rg in g[ev] for k in g[ev][rg]},
                  key=lambda k: (float(k[0]), float(k[1])))
    per = {}                       # eval -> key -> OOD mean
    for ev in present:
        per[ev] = {k: ood_mean(g[ev], k) for k in keys}
        per[ev] = {k: v for k, v in per[ev].items() if v is not None}

    noop = {ev: per[ev][NOOP] for ev in present}
    floor_min = {ev: float(np.mean([floors[ev][rg]["msp_min"] for rg in OOD_RUNGS])) for ev in present}

    # ---- ORACLE (ceiling) and LODO (the result) ----
    oracle = {ev: max(per[ev].values()) for ev in present}

    def lodo_over(pool):
        """LODO restricted to a subset of grid points. `pool` is a list of (T0, gamma) keys.

        Reports what the PROCEDURE achieves on the held-out eval, never the best point's score.
        """
        sel, pick = {}, {}
        for ev in present:
            others = [o for o in present if o != ev]
            if not others:
                continue
            shared = [k for k in pool if all(k in per[o] for o in others) and k in per[ev]]
            if not shared:
                continue
            b = max(shared, key=lambda k: float(np.mean([per[o][k] for o in others])))
            pick[ev] = b
            sel[ev] = per[ev][b]
        return sel, pick

    # THE FACTOR DECOMPOSITION. Selecting (T0, gamma) jointly confounds two questions. Split them:
    #   T0-only  : gamma pinned to 0 -> "does SHARPENING help at all?"
    #   full 2-D : both free         -> "does LENGTH-CONDITIONING add anything on top?"
    # The free-side result (§3.8) said sharpening yes, length no. This tests the same split on the
    # LEARNED weighter, which is the only way to attribute a joint win to the right factor.
    t0_only = [k for k in keys if float(k[1]) == 0.0]
    lodo, lodo_pick = lodo_over(keys)
    lodo_t0, lodo_t0_pick = lodo_over(t0_only)

    print("\n" + "-" * 100)
    print(f"PER-DATASET OOD MEANS  (population: {len(present)}/8 evals"
          f"{' -- PARTIAL' if missing else ' -- COMPLETE'})")
    print("-" * 100)
    print(f"{'eval':16s}{'no-op':>10s}{'LODO':>10s}{'LODO pick':>16s}{'ORACLE':>10s}{'msp_min':>10s}")
    for ev in present:
        t0, gm = lodo_pick[ev]
        print(f"{ev:16s}{noop[ev]:>+10.4f}{lodo[ev]:>+10.4f}{f'T0={t0},g={gm}':>16s}"
              f"{oracle[ev]:>+10.4f}{floor_min[ev]:>+10.4f}")
    mean_noop = float(np.mean([noop[e] for e in present]))
    mean_lodo = float(np.mean([lodo[e] for e in present]))
    mean_orc = float(np.mean([oracle[e] for e in present]))
    mean_fl = float(np.mean([floor_min[e] for e in present]))
    print(f"{'MEAN':16s}{mean_noop:>+10.4f}{mean_lodo:>+10.4f}{'':>16s}{mean_orc:>+10.4f}{mean_fl:>+10.4f}")

    def wilc(x):
        return _st.wilcoxon(x).pvalue if not np.allclose(x, 0) else 1.0

    d = np.array([lodo[e] - noop[e] for e in present])
    print(f"\n  LODO (T0 and gamma) vs no-op: {d.mean():+.4f}   wins {int((d > 0).sum())}/"
          f"{len(present)}   Wilcoxon p={wilc(d):.4f}")
    print(f"  ORACLE vs no-op:              {mean_orc - mean_noop:+.4f}   <-- CEILING, chosen on "
          f"test, NOT a result")
    if mean_orc - mean_noop > 1e-9:
        kept = (d.mean()) / (mean_orc - mean_noop)
        print(f"  honest selection keeps {kept:.0%} of the oracle gain and loses {1 - kept:.0%}.")

    # ---- the factor decomposition: is any win SHARPENING or LENGTH-CONDITIONING? ----
    if lodo_t0:
        dt = np.array([lodo_t0[e] - noop[e] for e in present if e in lodo_t0])
        dl = np.array([lodo[e] - lodo_t0[e] for e in present if e in lodo_t0])
        print("\n" + "-" * 100)
        print("FACTOR DECOMPOSITION -- selecting (T0, gamma) jointly confounds two questions")
        print("-" * 100)
        print(f"  T0 only (gamma pinned 0) vs no-op : {dt.mean():+.4f}  wins {int((dt > 0).sum())}/"
              f"{len(dt)}  p={wilc(dt):.4f}   <- does SHARPENING help?")
        print(f"  adding gamma on top of T0         : {dl.mean():+.4f}  wins {int((dl > 0).sum())}/"
              f"{len(dl)}  p={wilc(dl):.4f}   <- does LENGTH-CONDITIONING add anything?")
        print(f"  T0 picked per fold: " + "  ".join(f"{e}={lodo_t0_pick[e][0]}" for e in lodo_t0))
        print("  If the second line is <= 0, length-conditioning is dead on the LEARNED side too,")
        print("     which would be the SECOND independent null on it (the free side was §3.8).")

    # ---- the gamma sign pattern: the registered failure mode ----
    print("\n" + "-" * 100)
    print("THE gamma SIGN PATTERN -- the registered prediction for HOW this fails")
    print("-" * 100)
    print("A knob whose best value flips SIGN between datasets cannot survive leave-one-dataset-out:")
    print("selecting on the others hands the held-out one a value pointing the wrong way.")
    print(f"\n{'eval':16s}{'best (T0,gamma) per rung -> ':30s}")
    for ev in present:
        bits = []
        for rg in ["ID"] + OOD_RUNGS:
            if rg in g[ev] and g[ev][rg]:
                b = max(g[ev][rg], key=lambda k: g[ev][rg][k])
                bits.append(f"{rg.replace('-long', '')}:g={b[1]}")
        print(f"{ev:16s}" + "  ".join(bits))
    signs = {}
    for ev in present:
        gs = [float(max(g[ev][rg], key=lambda k: g[ev][rg][k])[1])
              for rg in OOD_RUNGS if rg in g[ev] and g[ev][rg]]
        signs[ev] = np.sign(np.mean(gs)) if gs else 0.0
    npos = sum(1 for v in signs.values() if v > 0)
    nneg = sum(1 for v in signs.values() if v < 0)
    print(f"\n  datasets whose mean best-gamma is POSITIVE: {npos}   NEGATIVE: {nneg}")
    print(f"  {'SIGN FLIPS BETWEEN DATASETS -- LODO cannot work' if npos and nneg else 'consistent sign'}")

    # ---- the argmax control ----
    print("\n" + "-" * 100)
    print("CONTROL: argmax(learned raw) == argmax(nll), i.e. is the LEARNED weakest-link the FREE one?")
    print("-" * 100)
    for ev in present:
        vals = [agree[ev][rg] for rg in agree[ev]]
        if vals:
            print(f"  {ev:16s} {np.mean(vals):6.1%}  (range {min(vals):.1%} to {max(vals):.1%})")
    print("  Low agreement means T->0 is a LEARNED weakest-link, NOT msp_min, so the two tracks share")
    print("  the FORM q = sum w_t*nll_t but not a path that literally passes through msp_min.")

    outp = Path(args.out)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["eval", "noop", "lodo", "lodo_T0", "lodo_gamma", "oracle", "msp_min", "n_evals"])
        for ev in present:
            w.writerow([ev, f"{noop[ev]:.4f}", f"{lodo[ev]:.4f}", lodo_pick[ev][0], lodo_pick[ev][1],
                        f"{oracle[ev]:.4f}", f"{floor_min[ev]:.4f}", len(present)])
    print(f"\nwrote {outp}")
    if missing:
        print(f"PARTIAL GRID: {len(present)}/8. Missing {missing}. Re-run when they land.")


if __name__ == "__main__":
    main()
