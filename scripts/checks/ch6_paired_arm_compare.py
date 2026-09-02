#!/usr/bin/env python
"""Compare a fitted analysis across the two population arms, and check its control cells.

For a method fitted on a labelled source pool, a correction to one dataset can move results on a
target that did not change, whenever the corrected dataset sits in that cell's training sources. So
the invariance question is not "did the other seven targets stay put" but "did the cells the
correction touches nowhere stay put".

  affected cells  the corrected dataset is the evaluation population or a realised training source.
                  These are expected to move. Their movement is a result, not a failure.
  control cells   the corrected dataset appears nowhere. Run with identical code and seeds on both
                  populations, these must reproduce. A control cell that moves is a failure and
                  stops the analysis.

Optionally also checks the original arm against a historical artifact of the same analysis. That is
a code-drift check and is separate from the population question: if it fails, the difference between
the two arms cannot be attributed to the population alone.

    python scripts/checks/ch6_paired_arm_compare.py \\
        --corrected <csv> --original <csv> [--historical <csv>] --out <csv>
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ch6_cleanv2_affected_cells import CORRECTED, derive       # noqa: E402

CONTROL_TOL = 1e-9          # identical code, identical seeds, identical inputs: bitwise or nothing
HISTORICAL_TOL = 1e-6       # a stored artifact is rounded; this is still far tighter than any effect



def _rel(p):
    """Path for display. A path given on the command line need not sit under the repository."""
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p

def load(path, value_col="prr"):
    """(eval, rung, method) -> mean over seeds. Seeds are averaged inside a cell, never pooled."""
    acc = defaultdict(list)
    for r in csv.DictReader(open(path)):
        v = r.get(value_col, "")
        if v == "":
            continue
        acc[(r["eval"], r["rung"], r["method"])].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def compare(a, b, keys):
    d = [(k, a[k] - b[k]) for k in keys if k in a and k in b]
    if not d:
        return None, 0, []
    worst = max(d, key=lambda t: abs(t[1]))
    return worst, len(d), d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--historical", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--value-col", default="prr")
    args = ap.parse_args()

    cells = derive()
    role = {(r["eval"], r["rung"]): r["role"] for r in cells}
    cor = load(args.corrected, args.value_col)
    org = load(args.original, args.value_col)

    shared = sorted(set(cor) & set(org))
    if not shared:
        raise SystemExit("the two arms share no cells; they are not the same analysis")
    only_cor = sorted(set(cor) - set(org))
    only_org = sorted(set(org) - set(cor))
    if only_cor or only_org:
        raise SystemExit(f"the arms cover different cells (corrected only {len(only_cor)}, "
                         f"original only {len(only_org)}); refusing a partial comparison. "
                         f"First few: {(only_cor + only_org)[:4]}")

    ctrl = [k for k in shared if role.get((k[0], k[1])) == "control"]
    aff = [k for k in shared if role.get((k[0], k[1])) == "affected"]
    print("=" * 100)
    print("PAIRED-ARM COMPARISON")
    print(f"  corrected  {Path(args.corrected).name}")
    print(f"  original   {Path(args.original).name}")
    print(f"  {len(shared)} shared method-cells: {len(ctrl)} control, {len(aff)} affected")
    print("=" * 100)

    failed = False

    if args.historical:
        hist = load(args.historical, args.value_col)
        worst, n, _ = compare(org, hist, shared)
        ok = worst is not None and abs(worst[1]) <= HISTORICAL_TOL
        print(f"\nCODE DRIFT  original arm against {Path(args.historical).name}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {n} cells compared, max abs difference "
              f"{abs(worst[1]):.2e} at {worst[0]}" if worst else "  no shared cells")
        if not ok:
            failed = True
            print("  The original arm does not reproduce the stored artifact, so a difference "
                  "between the arms cannot be attributed to the population alone.")

    worst, n, deltas = compare(cor, org, ctrl)
    print(f"\nCONTROL CELLS  must reproduce between arms")

    # NO CONTROL CELLS IS NOT A FAILED CHECK, AND MUST NOT BE REPORTED AS ONE. The corrected dataset
    # is its own evaluation population in every one of its cells, so it has none by construction.
    # Treating "nothing to check" as "the check failed" would raise a false alarm on exactly the one
    # dataset the correction is about -- and a warning that cries wolf there is worse than no warning,
    # because it trains the reader to dismiss it. Any OTHER dataset having no control cells is a real
    # problem, because it would mean the affectedness derivation is wrong.
    evals_here = {k[0] for k in shared}
    if not ctrl:
        if evals_here == {CORRECTED}:
            print(f"  not applicable: this is the corrected dataset, so all {len(aff)} of its cells "
                  "are affected by construction. Its invariance is established by the other "
                  "datasets, not by this one.")
            ok = True
        else:
            print(f"  [FAIL] no control cells for {sorted(evals_here)}, which is not the corrected "
                  "dataset. The affectedness derivation is wrong; do not read anything below.")
            ok = False
    else:
        ok = abs(worst[1]) <= CONTROL_TOL
        print(f"  [{'PASS' if ok else 'FAIL'}] {n} control method-cells, max abs difference "
              f"{abs(worst[1]):.2e} at {worst[0]}")
    if not ok:
        failed = True
        big = sorted((d for d in deltas if abs(d[1]) > CONTROL_TOL),
                     key=lambda t: -abs(t[1]))[:10]
        if big:
            print("  Cells that moved without touching the corrected dataset:")
            for k, v in big:
                print(f"    {k}  {v:+.6f}")
        print("  STOP. Investigate before publishing any number from this analysis.")

    print(f"\nAFFECTED CELLS  expected to move; this is the result, not a failure")
    by_method = defaultdict(list)
    for k in aff:
        by_method[k[2]].append(cor[k] - org[k])
    rows = []
    for m in sorted(by_method):
        v = np.array(by_method[m])
        print(f"  {m:32s} n={len(v):3d}  mean {v.mean():+.4f}  "
              f"max abs {np.abs(v).max():.4f}")
        rows.append({"scope": "affected cells", "method": m, "n_cells": len(v),
                     "mean_delta": round(float(v.mean()), 6),
                     "max_abs_delta": round(float(np.abs(v).max()), 6)})
    for m in sorted({k[2] for k in ctrl}):
        v = np.array([cor[k] - org[k] for k in ctrl if k[2] == m])
        rows.append({"scope": "control cells", "method": m, "n_cells": len(v),
                     "mean_delta": round(float(v.mean()), 9),
                     "max_abs_delta": round(float(np.abs(v).max()), 9)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "method", "n_cells", "mean_delta",
                                           "max_abs_delta"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {_rel(Path(args.out))}")

    if failed:
        raise SystemExit("paired-arm comparison FAILED")
    print("Paired-arm comparison passed.")


if __name__ == "__main__":
    main()
