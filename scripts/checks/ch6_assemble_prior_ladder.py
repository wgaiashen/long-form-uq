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
    disagreements = []
    for ev in EVALS:
        # A target whose per-setting cost does not fit one walltime is split across several jobs, each
        # writing its own file. Those runs each repeat the matched setting, because the driver always
        # includes it. The repeats are not discarded blindly: they must AGREE, and a disagreement is a
        # reproducibility failure worth surfacing rather than a duplicate worth dropping.
        parts = sorted(glob.glob(str(A / f"{args.tag}_fixed_prior_ladder_{ev}__{SLUG}.csv"))
                       + sorted(glob.glob(str(A / f"{args.tag}_fixed_prior_ladder_{ev}_*__{SLUG}.csv"))))
        if not parts:
            absent.append(ev)
            print(f"  {ev:14s} ABSENT   no file; the target was never produced")
            continue
        seen, rs = {}, []
        for part in parts:
            for r in csv.DictReader(open(part)):
                key = (r["rung"], r["eval"], r["method"], r.get("train", ""))
                if key in seen:
                    a, b = seen[key].get("prr_mean", ""), r.get("prr_mean", "")
                    if a != b:
                        disagreements.append((ev, key, a, b))
                    continue
                seen[key] = r
                rs.append(r)
        fields = fields or list(rs[0].keys())
        got = {r["rung"] for r in rs}
        if got != RUNGS:
            incomplete.append((ev, sorted(RUNGS - got)))
            print(f"  {ev:14s} PARTIAL  {len(got)}/5 rungs, missing {sorted(RUNGS - got)} -> EXCLUDED")
            continue
        complete.append(ev)
        rows.extend(rs)
        note = f" (joined from {len(parts)} files)" if len(parts) > 1 else ""
        print(f"  {ev:14s} complete {len(rs)} rows{note}")

    if disagreements:
        print(f"\n  [FAIL] {len(disagreements)} repeated cells disagree between split files:")
        for ev, key, a, b in disagreements[:8]:
            print(f"    {ev} {key}: {a} vs {b}")
        raise SystemExit("a repeated computation did not reproduce; stop and investigate before "
                         "assembling anything from these files")
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
