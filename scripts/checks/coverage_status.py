#!/usr/bin/env python
"""Completion status: for every METHOD, how many of its grid's cells actually exist on disk.

WHY. "50/50 cells" was reported once for the XL grid when a cell counted as present if ANY method had
landed in it. True per-method coverage was far lower, and the headline read as complete. So this counts
per method, states the denominator, and never collapses methods into one number.

Two grids, two denominators:
  * XL  = 10 evals x 5 rungs                      = 50 cells/method
  * LONG = 8 long evals x 5 rungs + 2 short evals x 1 (Long->Short) = 42 cells/method

A method absent from a grid entirely is shown as 0, not omitted -- an omitted row reads as "not part of
the study", a zero reads as "not run", and those must not be confusable.

EXCLUDED METHODS (author's decision 2026-08-04): `linear` is not a baseline the report uses, so it is
listed under EXCLUDED rather than dropped in silence -- a method that simply vanishes from a coverage
table is indistinguishable from one that was forgotten. It is still COMPUTED, because it is a logistic
regression on pooled vectors already in memory (seconds per cell, against poolers that dominate the job)
and removing it would mean editing a driver whose jobs are in flight. Its rows stay in the CSVs; it just
does not appear in the completion accounting or any reported table.

    python scripts/checks/coverage_status.py
    python scripts/checks/coverage_status.py --missing     # also list which cells are absent
"""
import argparse
import collections
import math
import csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from cohort import CANONICAL_10, LONG_8, SHORT_2, XL_RUNGS, LONG_RUNGS  # noqa: E402

RES = ROOT / "results"

EXCLUDED = ["linear"]      # computed, never reported — see the module docstring

XL_CELLS = [(e, r) for e in CANONICAL_10 for r in XL_RUNGS]
LONG_CELLS = ([(e, r) for e in LONG_8 for r in LONG_RUNGS]
              + [(e, "Long->Short") for e in SHORT_2])


def scan(patterns):
    """method -> (measured cells, NaN cells).

    A CELL COUNTS ONLY IF IT HOLDS A FINITE NUMBER (fixed 2026-08-05). This used to count any row whose
    key existed, so a cell written as `nan` scored as covered. That is the SAME error as the "50/50 cells"
    claim it was written to replace — presence is not measurement. It mattered immediately:
    `wmsp_seg_softmax` reported 42/42 while 17 of those 42 were NaN, and that would have gone into the
    project record as a complete method.
    """
    got = collections.defaultdict(set)
    bad = collections.defaultdict(set)
    for pat in patterns:
        for f in sorted(glob.glob(str(RES / pat))):
            for r in csv.DictReader(open(f)):
                rk = "rung" if "rung" in r else "setting"
                if not r.get("method") or not r.get("eval") or not r.get(rk):
                    continue
                cell = (r["eval"], r[rk])
                v = r.get("prr_mean", "")
                try:
                    ok = v not in ("", "None") and math.isfinite(float(v))
                except ValueError:
                    ok = False
                (got if ok else bad)[r["method"]].add(cell)
    for m in bad:                       # a cell that is NaN somewhere and finite elsewhere is measured
        bad[m] -= got[m]
    return got, bad


def report(title, got, bad, cells, groups, show_missing):
    n = len(cells)
    cellset = set(cells)
    evals = list(dict.fromkeys(e for e, _ in cells))
    print(f"\n{'='*78}\n{title}  —  denominator {n} cells "
          f"({len(evals)} evals x rungs)\n{'='*78}")
    for gname, members in groups:
        print(f"\n  {gname}")
        for m in members:
            have = got.get(m, set()) & cellset
            stray = got.get(m, set()) - cellset
            pct = 100.0 * len(have) / n
            bar = "#" * int(round(pct / 5)) + "." * (20 - int(round(pct / 5)))
            mark = "OK  " if len(have) == n else ("--  " if not have else "..  ")
            nbad = len(bad.get(m, set()) & cellset)
            print(f"    {mark}{m:26s} {len(have):3d}/{n}  [{bar}] {pct:5.1f}%"
                  + (f"   {nbad} cell(s) NaN — computed but NOT measured" if nbad else "")
                  + (f"   !! {len(stray)} row(s) outside the grid" if stray else ""))
            if show_missing and have and len(have) < n:
                miss = sorted(cellset - have)
                per = collections.defaultdict(list)
                for e, r in miss:
                    per[e].append(r)
                print("          missing: " + "; ".join(f"{e}[{','.join(v)}]" for e, v in per.items()))
    # Computed but deliberately not reported. Named, so its absence above is a decision on the record
    # rather than something that quietly fell off the table.
    for m in EXCLUDED:
        if m in got:
            have = got[m] & cellset
            print(f"\n  EXCLUDED (computed, not reported — author's decision 2026-08-04)"
                  f"\n        {m:26s} {len(have):3d}/{n} cells exist on disk")
    # per-eval completeness across ALL REPORTED methods in this grid
    allm = [m for _, ms in groups for m in ms]
    print(f"\n  per-eval (across the {len(allm)} methods above):")
    for e in evals:
        rungs = [r for ev, r in cells if ev == e]
        tot = len(rungs) * len(allm)
        have = sum(1 for m in allm for r in rungs if (e, r) in got.get(m, set()))
        print(f"    {e:16s} {have:4d}/{tot:4d}  {100.0*have/tot:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", action="store_true", help="list the absent cells per method")
    args = ap.parse_args()

    xl, xl_bad = scan(["xlcontrib_fam_*meta-llama*.csv", "xlonegrid_fam_*meta-llama*.csv"])
    report("XL GRID (ProbeDrift-XL)", xl, xl_bad, XL_CELLS, [
        ("poolers / aggregation", ["uniform", "attention", "mean-pool+MLP", "last-token",
                                   "per-sentence", "per-token"]),
        ("contribution (weighted MSP)", ["weighted_msp_norm", "weighted_msp_unc"]),
        ("supervised baselines", ["ptrue_accurate", "lookback"]),
        ("unsupervised floors", ["msp_min", "msp_sum", "perplexity", "fair_floor:msp_min"]),
        ("verdicts (paired bootstrap)", ["VERDICT:attention_vs_floor", "VERDICT:wmsp_norm_vs_floor",
                                         "VERDICT:wmsp_norm_vs_attention"]),
    ], args.missing)

    pdl, pdl_bad = scan(["pdl_fam_*meta-llama*.csv"])
    report("LONG GRID (ProbeDriftLong)", pdl, pdl_bad, LONG_CELLS, [
        ("poolers / aggregation", ["uniform", "attention"]),
        ("contribution (weighted MSP)", ["wmsp_norm", "wmsp_shrink2", "wmsp_shrink10", "wmsp_blondel",
                                         "wmsp_shrink2_blondel", "wmsp_shrink10_blondel",
                                         "wmsp_seg_flat", "wmsp_seg_softmax"]),
        ("supervised baselines", ["saplma", "ptrue", "lookback"]),
        ("unsupervised floors", ["floor_min", "floor_ppl", "floor_sum", "fair_floor", "ptrue_unsup"]),
        ("verdicts (paired bootstrap)", ["VERDICT:attention_vs_fairfloor", "VERDICT:bestw_vs_fairfloor",
                                         "VERDICT:bestw_vs_bestpooler"]),
    ], args.missing)


if __name__ == "__main__":
    main()
