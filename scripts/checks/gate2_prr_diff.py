"""GATE 2 — diff the post-library PRR against the frozen master table.

Pairs with `pbs/g2_library_prr.pbs`. That job re-runs the ladder under LUQ_CARVE=legacy on a
couple of evals, writing to `results/g2_relib_<eval>__<slug>.csv`; this reads those and compares
every (rung, eval, method) cell against `results/pdl_master__<slug>.csv`.

WHAT COUNTS AS PASSING
----------------------
EXACT to the CSV's own precision (4 dp), not "close". Gate 1 already showed the row lists are
byte-identical, so any movement here is a real defect in the method half, not rounding — and the
right response is to find it, not to widen a tolerance.

⚠️ TWO FAILURE MODES THIS DELIBERATELY SEPARATES:
  * a cell that MOVED            -> the library changed a number. Hard fail.
  * a cell that is MISSING       -> the re-run did not produce it (trimmed method set, or a cell
                                    that legitimately does not exist). Reported, never counted as
                                    a pass, and never silently skipped.
A missing cell reads as "not measured"; a matching cell reads as "measured and equal". Those must
never be confusable, which is why they are counted in separate columns.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
KEY = ["rung", "eval", "method"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="samsum,factscore")
    ap.add_argument("--tol", type=float, default=5e-5,
                    help="CSVs carry 4 dp; anything above half a unit in the last place is real movement")
    args = ap.parse_args()

    master_path = ROOT / f"results/pdl_master__{SLUG}.csv"
    if not master_path.exists():
        raise SystemExit(f"missing frozen master: {master_path}")
    master = pd.read_csv(master_path)[KEY + ["prr"]].rename(columns={"prr": "prr_master"})

    frames = []
    for ev in args.evals.split(","):
        p = ROOT / f"results/g2_relib_{ev}__{SLUG}.csv"
        if not p.exists():
            print(f"⚠️  {ev}: {p.name} not found — job not finished? SKIPPING (not a pass)")
            continue
        df = pd.read_csv(p)
        df = df[df["method"] != "VERDICT"] if "method" in df else df
        col = "prr_mean" if "prr_mean" in df.columns else "prr"
        frames.append(df[KEY + [col]].rename(columns={col: "prr_new"}))
    if not frames:
        raise SystemExit("no gate-2 outputs found — nothing to diff")
    new = pd.concat(frames, ignore_index=True).dropna(subset=["prr_new"])

    m = new.merge(master, on=KEY, how="left")
    matched = m.dropna(subset=["prr_master"]).copy()
    missing = m[m["prr_master"].isna()]
    matched["delta"] = (matched["prr_new"] - matched["prr_master"]).abs()
    moved = matched[matched["delta"] > args.tol]

    print(f"GATE 2 — PRR diff vs {master_path.name}\n")
    print(f"  cells re-run and present in the master : {len(matched)}")
    print(f"  cells IDENTICAL (<= {args.tol:g})        : {len(matched) - len(moved)}")
    print(f"  cells MOVED                            : {len(moved)}")
    print(f"  cells not in the master (not a pass)   : {len(missing)}")

    if len(missing):
        print("\n  --- present in the re-run, absent from the master (reported, not passed) ---")
        for _, r in missing.head(20).iterrows():
            print(f"    {r['rung']:16} {r['eval']:14} {r['method']}")

    if len(moved):
        print("\n  --- MOVED — investigate, do NOT widen the tolerance ---")
        for _, r in moved.sort_values("delta", ascending=False).head(30).iterrows():
            print(f"    {r['rung']:16} {r['eval']:14} {r['method']:22} "
                  f"master {r['prr_master']:+.4f} -> new {r['prr_new']:+.4f}  Δ {r['delta']:.5f}")

    ok = len(moved) == 0 and len(matched) > 0
    print("\n" + "=" * 78)
    print(f"GATE 2: {'PASS — the library move changed no PRR' if ok else 'FAIL — a number moved'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
