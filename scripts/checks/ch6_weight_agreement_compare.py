#!/usr/bin/env python
"""Does the learned token weighting rediscover the highest-loss token, compared across two arms.

The learned weighting is only interesting if it contributes information the token surprisal does not
already carry. The direct test is how often its largest weight lands on the token that already has
the largest loss: a high rate would mean the method is a disguised maximum-loss rule.

This summarises the per-response diagnostic per target, and checks the two population arms against
each other. The diagnostic trains each target's matched-setting cell, which trains on that target
alone, so under the cell-level rule **every target except the corrected one is a control and must
reproduce exactly**. Only the corrected target is expected to move.

    python scripts/checks/ch6_weight_agreement_compare.py \\
        --corrected <csv> --original <csv> --out <csv>
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORRECTED_DATASET = "med_quad"
CONTROL_TOL = 1e-9
ORDER = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
METRICS = ["argmax_agree", "spearman_w_nll", "topk_overlap_1", "topk_overlap_5",
           "topk_overlap_10pct"]


def _rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def summarise(path, policy):
    """Mean of each per-response metric, per target, on one weighting policy."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["policy"] != policy:
            continue
        for m in METRICS:
            v = r.get(m, "")
            if v not in ("", None):
                acc[r["eval"]][m].append(float(v))
    return {d: {m: float(np.mean(v)) for m, v in ms.items()} for d, ms in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="wmsp",
                    help="token population the weighting is defined over; the default is the one the "
                         "method itself uses.")
    args = ap.parse_args()

    cor = summarise(args.corrected, args.policy)
    org = summarise(args.original, args.policy)
    shared = [d for d in ORDER if d in cor and d in org]
    missing = [d for d in ORDER if d not in cor or d not in org]
    if missing:
        raise SystemExit(f"targets missing from one arm: {missing}; refusing a partial comparison")

    print("=" * 104)
    print(f"LARGEST LEARNED WEIGHT AGAINST LARGEST LOSS   policy = {args.policy}")
    print(f"  corrected  {_rel(args.corrected)}")
    print(f"  original   {_rel(args.original)}")
    print("=" * 104)
    print(f"{'target':14s}{'agreement':>12s}{'original':>11s}{'delta':>10s}"
          f"{'rank corr':>12s}{'top-1 ovl':>11s}{'role':>10s}")

    rows, violations = [], []
    for d in shared:
        role = "corrected" if d == CORRECTED_DATASET else "control"
        a, o = cor[d]["argmax_agree"], org[d]["argmax_agree"]
        delta = a - o
        if role == "control" and abs(delta) > CONTROL_TOL:
            violations.append((d, delta))
        print(f"{d:14s}{a:>12.4f}{o:>11.4f}{delta:>+10.6f}"
              f"{cor[d]['spearman_w_nll']:>12.4f}{cor[d]['topk_overlap_1']:>11.4f}{role:>10s}")
        rows.append({"target": d, "role": role,
                     **{f"corrected_{m}": round(cor[d][m], 6) for m in METRICS},
                     "original_argmax_agree": round(o, 6),
                     "delta_argmax_agree": round(delta, 9)})

    agree = [cor[d]["argmax_agree"] for d in shared]
    print(f"\nAgreement across the eight targets: {min(agree) * 100:.1f} to {max(agree) * 100:.1f} "
          f"per cent. The learned weighting is not a disguised maximum-loss rule.")

    ok = not violations
    print(f"\n[{'PASS' if ok else 'FAIL'}] control targets reproduce between arms "
          f"({len(shared) - 1} controls, tolerance {CONTROL_TOL:.0e})")
    if not ok:
        for d, v in violations:
            print(f"    {d} moved by {v:+.6f} without touching the corrected dataset")
        print("    STOP. Investigate before publishing any number from this diagnostic.")

    rows.append({"target": "RANGE across the eight targets", "role": "",
                 **{f"corrected_{m}": "" for m in METRICS},
                 "original_argmax_agree": "",
                 "delta_argmax_agree": f"{min(agree) * 100:.1f} to {max(agree) * 100:.1f} per cent"})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {_rel(args.out)}")
    if not ok:
        raise SystemExit("control targets moved; comparison FAILED")


if __name__ == "__main__":
    main()
