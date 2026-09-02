#!/usr/bin/env python
"""Assemble the per-target position-prior files into one grid, refusing anything partial.

The prior ladder trains a pooler for every arm, prior, transfer rung and seed, so it is run one
evaluation target per job. This joins those files back into a single table.

A TARGET IS EITHER COMPLETE OR ABSENT. A job killed at its walltime leaves the rungs it finished, and
a grid silently missing rungs is the worst possible output: it averages to a plausible number and
reads as measured. So every target is checked for all five rungs, and one that falls short is
reported and excluded rather than folded in.

    python scripts/checks/ch6_assemble_prior_ladder.py --tag ch6_cleanv2 --out <csv>
"""
import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
A = ROOT / "results" / "analysis"
EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = {"ID", "SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"}


def _rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="ch6_cleanv2 or ch6_origspan")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else A / f"{args.tag}_fixed_prior_ladder__{SLUG}.csv"

    print("=" * 92)
    print(f"ASSEMBLING THE POSITION-PRIOR GRID  tag={args.tag}")
    print("=" * 92)
    rows, complete, incomplete, absent, fields = [], [], [], [], None
    for ev in EVALS:
        f = A / f"{args.tag}_fixed_prior_ladder_{ev}__{SLUG}.csv"
        if not f.exists():
            absent.append(ev)
            print(f"  {ev:14s} ABSENT   no file; the target was never produced")
            continue
        rs = list(csv.DictReader(open(f)))
        fields = fields or list(rs[0].keys())
        got = {r["rung"] for r in rs}
        if got != RUNGS:
            incomplete.append((ev, sorted(RUNGS - got)))
            print(f"  {ev:14s} PARTIAL  {len(got)}/5 rungs, missing {sorted(RUNGS - got)} -> EXCLUDED")
            continue
        complete.append(ev)
        rows.extend(rs)
        print(f"  {ev:14s} complete {len(rs)} rows")

    print(f"\n{len(complete)} of {len(EVALS)} targets complete")
    if incomplete or absent:
        print("WARNING: THIS IS NOT THE FULL GRID. Do not report a macro over it as an "
              "eight-target result.")
        for ev, miss in incomplete:
            print(f"   {ev}: missing {miss}")
        if absent:
            print(f"   never produced: {absent}")
    if not rows:
        raise SystemExit("nothing complete to assemble")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {_rel(out)} ({len(rows)} rows, targets: {', '.join(complete)})")


if __name__ == "__main__":
    main()
