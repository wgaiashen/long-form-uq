#!/usr/bin/env python
"""Recompute the `different_label_projection` flag in the XL contribution CSVs, PER RUNG.

WHY. The flag used to be computed as `different_label_projection(eval_target)` — i.e. "is the TARGET
scored on a different projection of 'good' than correctness?" — and then applied to every non-ID rung of
that eval. That was right while the only factuality set was ExpertQA and it appeared as an eval target
only. It stopped being right when factscore joined the cohort on 2026-08-04, because factuality sets are
now TRAINING SOURCES in almost every pool. Two opposite errors resulted, 42 rows in all:

  * expertqa/SameTask (train=factscore) and factscore/SameTask (train=expertqa) were flagged TRUE. Both
    sides carry `factuality`, so those rungs are SAME-label. The caveat would have wrongly discounted the
    one rung the factuality family has.
  * asqa/LOO, med_quad/LOO, samsum/LOO and samsum/DiffTask were flagged FALSE while their pools DO
    include expertqa+factscore. That is the dangerous direction: an unflagged cross-label pool reads as
    a clean same-label result.

The PRR values are unaffected — each dataset is always scored on its own label via `label_of(d)`, which
was already per-dataset. This is a caveat/metadata correction only, so the numbers are not recomputed;
the flag is re-derived from the `eval` and `train` columns the run itself recorded.

⚠️ RUN ONLY AFTER EVERY xlcontrib JOB HAS FINISHED. A job still running holds its rows in memory and
rewrites the whole CSV at each cell, so patching underneath it would be overwritten.

    python scripts/checks/fix_crosslabel_flag.py --dry-run     # report, change nothing
    python scripts/checks/fix_crosslabel_flag.py
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from cohort import LABEL_FIELD  # noqa: E402

PATTERN = str(ROOT / "results" / "xlcontrib_fam_*meta-llama*.csv")


def correct_flag(eval_ds, train_field):
    """True iff any ACTUAL training source is scored on a different projection than the eval target."""
    srcs = [s.split(":")[0] for s in (train_field or "").split("+") if s]
    tgt = LABEL_FIELD.get(eval_ds, "correctness")
    return any(LABEL_FIELD.get(d, "correctness") != tgt for d in srcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    total = 0
    for f in sorted(glob.glob(PATTERN)):
        rows = list(csv.DictReader(open(f)))
        if not rows or "different_label_projection" not in rows[0]:
            continue
        changed = 0
        for r in rows:
            # ⚠️ BLANKS STAY BLANK. The `fair_floor:` and `VERDICT:` rows were written by two append
            # sites that never carried this field, so their flag is EMPTY = "not stated". Filling it
            # here would be recording a value the run never emitted, which the standing rule forbids
            # (a blank reads as "not measured", a value reads as "measured"). The driver now stamps
            # those sites, so future runs carry it honestly; old rows are left as they were.
            if str(r["different_label_projection"]).strip() == "":
                continue
            want = correct_flag(r["eval"], r.get("train", ""))
            if str(r["different_label_projection"]).lower() != str(want).lower():
                r["different_label_projection"] = want
                changed += 1
        if not changed:
            continue
        total += changed
        print(f"  {os.path.basename(f)}: {changed} row(s) re-flagged")
        if args.dry_run:
            continue
        tmp = f + ".partial"                       # atomic, like the drivers' own flush
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        os.replace(tmp, f)
    print(f"\n{total} row(s) {'would be' if args.dry_run else ''} corrected")


if __name__ == "__main__":
    main()
