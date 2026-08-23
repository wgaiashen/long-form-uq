#!/usr/bin/env python
"""Concatenate the eight per-eval ladder files for one model into a single master table.

WHY THIS IS A PLAIN CONCATENATION and needs no gates. Each eval was run as its own job writing its
own file, but every one of them ran the same driver over the same eight-dataset source pool at the
same layer and seeds. There is nothing to reconcile: no cell is inherited from another population,
no method is filled from a second source, and no row is recomputed. That is the opposite of the
clean-answer-span master, where 21 of 40 cells came from a different population and had to be proved
identical before they could be carried across.

WHAT IS STILL CHECKED, because a concatenation can hide exactly two things:
  - a SHORT GRID. The driver skips a cell in silence when a per-token cache is missing, so eight
    files that each look finished can still add up to fewer than 40 cells. The expected count is
    asserted, not printed for someone to notice.
  - a SPLIT POPULATION. Every row carries git_sha and carve. If the eight jobs did not all run the
    same code against the same carve rule they are not one table, and pooling them would average
    over a difference rather than measure one.

    python scripts/checks/assemble_sens8_master.py --model google/gemma-2-9b --write
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CELLS = 40                      # 8 evals x 5 rungs, nothing legitimately omitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b")
    ap.add_argument("--write", action="store_true", help="persist; otherwise report and stop")
    args = ap.parse_args()
    slug = args.model.replace("/", "_")

    src = sorted((ROOT / "results").glob(f"wmodels_sens8__{slug}__*.csv"))
    src = [f for f in src if "master" not in f.name]
    if not src:
        sys.exit(f"no per-eval files found for {args.model}")
    print(f"  {len(src)} per-eval files")

    frames, cols = [], None
    for f in src:
        d = pd.read_csv(f)
        if cols is None:
            cols = list(d.columns)
        elif list(d.columns) != cols:
            sys.exit(f"schema mismatch in {f.name}: these are not one table")
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)

    cells = m.groupby(["eval", "rung"]).ngroups
    print(f"  rows {len(m)}   cells {cells}/{EXPECTED_CELLS}   "
          f"evals {m['eval'].nunique()}   methods {m['method'].nunique()}")

    per = m.groupby("eval")["rung"].nunique()
    short = per[per < 5]
    if len(short):
        print(f"  **SHORT GRID** {dict(short)} -- a per-token cache was probably missing")

    nulls = int(m["prr_mean"].isna().sum())
    shas = sorted(m["git_sha"].dropna().unique()) if "git_sha" in m else []
    carves = sorted(m["carve"].dropna().unique()) if "carve" in m else []
    print(f"  null PRR {nulls}   git_sha {shas}   carve {carves}")

    ok = (cells == EXPECTED_CELLS) and not len(short) and nulls == 0 and len(carves) <= 1
    if len(shas) > 1:
        print("  NOTE more than one git_sha: the jobs did not all run the same code")
        ok = False
    if len(carves) > 1:
        print("  **more than one carve rule -- these are not one population**")

    cov = m.groupby("method").apply(lambda g: g.groupby(["eval", "rung"]).ngroups)
    partial = cov[cov < EXPECTED_CELLS]
    print(f"  methods at {EXPECTED_CELLS}/{EXPECTED_CELLS}: {int((cov == EXPECTED_CELLS).sum())}/{len(cov)}"
          + (f"   PARTIAL {dict(partial)}" if len(partial) else ""))

    print("\n  " + ("VALID -- complete grid, one population"
                    if ok else "**NOT VALID -- do not quote this table**"))
    if not args.write:
        print("  (dry run: pass --write to persist)")
        return
    if not ok:
        sys.exit("refusing to write an invalid master")

    out = ROOT / "results" / f"wmodels_sens8_master__{slug}.csv"
    m = m.sort_values(["eval", "rung", "method"], kind="stable")
    m.to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
