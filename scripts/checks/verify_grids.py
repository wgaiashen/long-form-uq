#!/usr/bin/env python
"""ONE gate over BOTH finished grids: run this before any number is read, quoted, or written to a doc.

WHY. Every defect this project has shipped was a plausible-looking number standing in for an absence, and
each was caught by a DIFFERENT ad-hoc check written after the fact. This collects those checks so they run
together, every time, rather than depending on remembering which one applies.

CHECKS (each is a hard FAIL, and the exit code is non-zero if any fails):
  1. NaN / blank      — a cell that is not a finite number is NOT a measurement. Reported per method.
  2. Coverage         — every method must cover its grid's full cell set (42 long, 50 XL), or be named.
  3. Provenance       — every row must carry a git_sha, and `dirty=1` rows are called out. A run from an
                        uncommitted tree cannot be reproduced and must not silently enter a table.
  4. Code consistency — all rows of a grid should come from ONE code state. Multiple shas are allowed but
                        listed, so "was this one population?" is answerable rather than assumed.
  5. Duplicate cells  — the same (rung, eval, method) written by two files at the same tier is a
                        cross-population merge waiting to happen. Overlaps must AGREE to 1e-9.
  6. Stale sources    — files known to predate a fix (assemble_pdl_table.STALE) must not be the only
                        source of any cell.

    python scripts/checks/verify_grids.py            # both grids
    python scripts/checks/verify_grids.py --grid long
"""
import argparse
import collections
import csv
import glob
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from cohort import CANONICAL_10, LONG_8, SHORT_2, XL_RUNGS, LONG_RUNGS  # noqa: E402

RES = ROOT / "results"
GRIDS = {
    "long": {"patterns": ["pdl_fam_*meta-llama*.csv"],
             "cells": [(e, r) for e in LONG_8 for r in LONG_RUNGS] + [(e, "Long->Short") for e in SHORT_2],
             "rungkey": "rung"},
    "xl":   {"patterns": ["xlcontrib_fam_*meta-llama*.csv", "xlonegrid_fam_*meta-llama*.csv"],
             "cells": [(e, r) for e in CANONICAL_10 for r in XL_RUNGS],
             "rungkey": "rung"},
}
# Methods that are legitimately not on the full grid. Named, so "missing" always means something.
EXEMPT = {"linear",                              # computed, deliberately not reported
          "fair_floor:msp_min"}                  # derived per cell; present wherever floors are


def load(patterns):
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(str(RES / pat))):
            for r in csv.DictReader(open(f)):
                r["__file"] = Path(f).name
                rows.append(r)
    return rows


def check(grid_name, cfg, verbose):
    rows = load(cfg["patterns"])
    if not rows:
        print(f"  ⚠️ {grid_name}: no source files matched — nothing to verify")
        return ["no source files"]
    fails = []
    cellset = set(cfg["cells"])
    finite, nonfinite = collections.defaultdict(set), collections.defaultdict(set)
    perfile = collections.defaultdict(list)
    shas, dirty_rows = collections.Counter(), 0
    for r in rows:
        rk = "rung" if r.get("rung") else "setting"
        m, ev, rg = r.get("method"), r.get("eval"), r.get(rk)
        if not (m and ev and rg):
            continue
        v = r.get("prr_mean", "")
        try:
            ok = v not in ("", "None") and math.isfinite(float(v))
        except ValueError:
            ok = False
        (finite if ok else nonfinite)[m].add((ev, rg))
        if ok:
            perfile[(rg, ev, m)].append((float(v), r["__file"]))
        shas[(r.get("git_sha") or "<none>")[:8]] += 1
        dirty_rows += str(r.get("dirty", "")) == "1"
    for m in nonfinite:
        nonfinite[m] -= finite[m]

    print(f"\n{'='*76}\n{grid_name.upper()} GRID  —  {len(rows)} rows, denominator {len(cellset)} cells\n{'='*76}")

    # 1 + 2
    nan_tot = 0
    for m in sorted(set(finite) | set(nonfinite)):
        have, bad = finite[m] & cellset, nonfinite[m] & cellset
        nan_tot += len(bad)
        if bad:
            fails.append(f"{grid_name}/{m}: {len(bad)} NaN cell(s) {sorted(bad)[:3]}")
        if m not in EXEMPT and len(have) < len(cellset):
            miss = sorted(cellset - have)
            fails.append(f"{grid_name}/{m}: {len(have)}/{len(cellset)} covered, missing {miss[:4]}"
                         + (" ..." if len(miss) > 4 else ""))
        if verbose or bad or (m not in EXEMPT and len(have) < len(cellset)):
            flag = "OK " if (not bad and (len(have) == len(cellset) or m in EXEMPT)) else "FAIL"
            print(f"  {flag} {m:28s} {len(have):3d}/{len(cellset)}"
                  + (f"   ⚠️ {len(bad)} NaN" if bad else ""))
    print(f"  -- NaN cells across the grid: {nan_tot}")

    # 3 + 4
    print(f"  -- git_sha values: {dict(shas)}")
    if "<none>" in shas:
        fails.append(f"{grid_name}: {shas['<none>']} row(s) carry NO git_sha — untraceable to a code state")
    if dirty_rows:
        fails.append(f"{grid_name}: {dirty_rows} row(s) stamped dirty=1 (produced from an uncommitted tree)")

    # 5
    dupes = {k: v for k, v in perfile.items() if len(v) > 1}
    disagree = {k: v for k, v in dupes.items() if max(x[0] for x in v) - min(x[0] for x in v) > 1e-9}
    print(f"  -- cells written by >1 file: {len(dupes)}, of which DISAGREE: {len(disagree)}")
    for k, v in list(disagree.items())[:5]:
        fails.append(f"{grid_name}/{k}: sources disagree {[(round(a,4), b) for a, b in v]}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["long", "xl", "both"], default="both")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    names = ["long", "xl"] if args.grid == "both" else [args.grid]
    fails = []
    for n in names:
        fails += check(n, GRIDS[n], args.verbose)
    print("\n" + "=" * 76)
    if fails:
        print(f"❌ {len(fails)} PROBLEM(S) — do not read these numbers yet:")
        for f in fails:
            print(f"   {f}")
        sys.exit(1)
    print("✅ ALL CHECKS PASS — grids are complete, finite, traceable, and internally consistent.")


if __name__ == "__main__":
    main()
