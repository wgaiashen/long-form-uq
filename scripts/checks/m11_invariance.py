"""Did a correction move a number it had no business moving?

The single-layer driver was corrected in one respect for the full-layer reproduction: the
ten-component reduction now runs before the sequence-probability and entropy columns are appended,
which is the order the reference implementation uses. At one cached layer that reordering cannot have
any effect, because the reduction requires ten input columns and there are at most three, so it never
executes. The expected result of re-running the driver is therefore that EVERY recorded value is
unchanged, exactly.

That expectation is worth checking rather than asserting. A correction that moves a single-layer
number is a correction in the wrong place, and this is the cheapest place to find that out.

Blank cells are compared as blanks. A number opposite a blank is a disagreement about whether a cell
was measurable at all, which matters more than any tolerance and is reported separately.

    python scripts/checks/m11_invariance.py --new <rerun.csv> --recorded <on_record.csv>
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    with open(path) as fh:
        return {(r["eval"], r["rung"], r["method"]): r for r in _csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="the CSV the re-run just produced")
    ap.add_argument("--recorded", required=True, help="the CSV already on record")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="0.0 by default, and that is deliberate: the reordering cannot fire at one "
                         "layer, so anything other than exact equality is a finding")
    args = ap.parse_args()

    a, b = load(ROOT / args.new), load(ROOT / args.recorded)
    common = sorted(set(a) & set(b))
    only_new, only_rec = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    print(f"re-run {len(a)} rows | on record {len(b)} rows | comparable {len(common)}")
    if only_new:
        print(f"  only in the re-run: {len(only_new)}")
        for k in only_new[:5]:
            print(f"    {k}")
    if only_rec:
        print(f"  only on record: {len(only_rec)}")
        for k in only_rec[:5]:
            print(f"    {k}")

    worst, where, blanks, moved = 0.0, "", [], []
    for k in common:
        x, y = a[k]["prr_mean"], b[k]["prr_mean"]
        if (x == "") != (y == ""):
            blanks.append(f"{k}: re-run {x or 'blank'}, on record {y or 'blank'}")
            continue
        if x == "":
            continue
        d = abs(float(x) - float(y))
        if d > worst:
            worst, where = d, str(k)
        if d > args.tol:
            moved.append(f"{k}: {y} -> {x} (|d| = {d:.3e})")

    print(f"\nworst absolute difference {worst:.6e} at {where or 'nothing compared'}")
    print(f"bar: {args.tol:.1e}, and disagreement about which cells are blank is never allowed")
    for line in blanks[:10]:
        print(f"  BLANK MISMATCH {line}")
    for line in moved[:10]:
        print(f"  MOVED {line}")
    ok = not blanks and not moved and not only_rec and bool(common)
    print("INVARIANCE CONTROL: " + ("PASS, nothing moved" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
