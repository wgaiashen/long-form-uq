#!/usr/bin/env python
"""Remove VERDICT rows from RESTRICTED probedriftlong runs, where the verdict does not mean what it says.

THE PROBLEM. `VERDICT:bestw_vs_fairfloor` compares the BEST wMSP variant against the fair floor. In a full
run "best wMSP" is the best of eight variants. In a run launched with `--wmsp-only wmsp_seg_softmax` there
is only ONE variant, so "best wMSP" silently means "seg_softmax" — a different quantity carrying the same
name. The two disagree by up to 0.20 on asqa, and nothing in the row says which one it is.

That is the project's recurring failure shape exactly: not a wrong formula, but a name that quietly changes
meaning depending on how the run was invoked, producing a well-formed number for a comparison nobody made.

WHY REMOVE RATHER THAN RENAME. The restricted verdict answers a question no one asked ("does seg_softmax
alone beat the floor?"), and the full-run verdict for the same cell already exists in the main file. Keeping
both under any name invites the wrong one being read. The floors, SAPLMA and seg_softmax rows in these files
ARE valid and are left untouched — they are computed identically regardless of the variant restriction, and
they agree with the main run's values to 1e-9, which is what proves the restricted run is on-population.

Idempotent: re-run after any further refill.

    python scripts/checks/strip_restricted_verdicts.py --dry-run
    python scripts/checks/strip_restricted_verdicts.py
"""
import argparse
import csv
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Only files produced by a RESTRICTED invocation. A full run's verdicts are correct and must not be touched.
RESTRICTED_GLOBS = ["results/pdl_fam_*_segsm__*.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    total = 0
    for pat in RESTRICTED_GLOBS:
        for f in sorted(glob.glob(str(ROOT / pat))):
            rows = list(csv.DictReader(open(f)))
            keep = [r for r in rows if not (r.get("method", "").startswith("VERDICT:"))]
            n = len(rows) - len(keep)
            if not n:
                continue
            total += n
            print(f"  {Path(f).name}: dropping {n} restricted VERDICT row(s)")
            if args.dry_run:
                continue
            tmp = f + ".partial"
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(keep)
            os.replace(tmp, f)
    print(f"\n{total} row(s) {'would be' if args.dry_run else ''} removed")


if __name__ == "__main__":
    main()
