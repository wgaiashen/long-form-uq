#!/usr/bin/env python
"""One-off invariance sweep for the 2026-08-11/12 Qwen clean-span correction.

Not a general pipeline script -- ties together the specific old (results/prev/2026-08-11/) and new
(results/analysis/pdl_qwenclean_*.csv) files for this one correction, so it lives beside the other
one-off checks rather than in a reusable driver.

For every (eval, rung) cell: if NEITHER the eval nor any dataset in the training pool is one of the
four cleaned datasets (samsum, med_quad, expertqa, factscore), the cell must reproduce the canonical
master EXACTLY (floors are exact by construction; supervised methods can carry tiny cross-seed float
noise, so the tolerance is 1e-6, not "eyeballed"). If the cell DOES touch a cleaned dataset, no
assertion is made -- movement there is expected, and its magnitude is reported, not gated.

Mirrors the same check RCS ran for the Llama MedQuAD correction (the project results log): "0 cells
moved without [a cleaned dataset] in train or eval" is the correctness control this script proves.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_DIR = ROOT / "results" / "prev" / "2026-08-11"
NEW_DIR = ROOT / "results" / "analysis"
SLUG = "Qwen_Qwen2.5-14B"
EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
CLEANED = {"samsum", "med_quad", "expertqa", "factscore"}
TOL = 1e-6


def load(path):
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["method"].startswith("VERDICT"):
                continue
            rows[(r["rung"], r["method"])] = (float(r["prr_mean"]), r.get("train", ""))
    return rows


def train_sources(train_field):
    # "xsum:360+cnn_dailymail:360+..." -> {"xsum", "cnn_dailymail", ...}; "pubmed_qa:1800" -> {"pubmed_qa"}
    return {tok.split(":")[0] for tok in train_field.split("+") if tok}


def main():
    n_checked = n_exact_ok = n_exact_fail = n_expected_move = n_unexpected_move = n_missing = 0
    fails = []
    moves = []

    for ev in EVALS:
        old = load(OLD_DIR / f"probedriftlong__{SLUG}__{ev}.csv")
        new = load(NEW_DIR / f"pdl_qwenclean_{ev}__{SLUG}.csv")
        if not old or not new:
            print(f"[{ev}] MISSING old or new file -- old={len(old)} rows, new={len(new)} rows")
            n_missing += 1
            continue
        for (rung, method), (new_val, train_field) in new.items():
            key = (rung, method)
            if key not in old:
                continue  # method not in old registry (shouldn't happen for the frozen 9, but don't crash)
            old_val, _ = old[key]
            sources = train_sources(train_field) | {ev}
            touches_cleaned = bool(sources & CLEANED)
            n_checked += 1
            delta = new_val - old_val
            if touches_cleaned:
                if abs(delta) > TOL:
                    n_expected_move += 1
                    moves.append((ev, rung, method, old_val, new_val, delta))
                # else: touches a cleaned dataset but happens not to move (e.g. a floor cell that
                # coincidentally lands the same) -- fine, not a violation either way
            else:
                if abs(delta) <= TOL:
                    n_exact_ok += 1
                else:
                    n_exact_fail += 1
                    fails.append((ev, rung, method, old_val, new_val, delta))

    print("=" * 100)
    print("QWEN CLEAN-SPAN INVARIANCE SWEEP")
    print("=" * 100)
    print(f"cells checked: {n_checked}")
    print(f"  untouched-pool cells matching exactly (<= {TOL}):  {n_exact_ok}")
    print(f"  untouched-pool cells that MOVED (violation):       {n_exact_fail}")
    print(f"  cleaned-pool cells that moved (expected):          {n_expected_move}")
    print(f"  datasets with missing old/new file:                {n_missing}")

    if fails:
        print("\nINVARIANCE VIOLATIONS -- a cell with NO cleaned dataset in train/eval moved:")
        for ev, rung, method, o, n, d in fails:
            print(f"  [{ev:14s}] {rung:14s} {method:14s} old={o:+.4f} new={n:+.4f} delta={d:+.4f}")
    else:
        print("\nNO INVARIANCE VIOLATIONS — every cell with zero cleaned-dataset exposure in "
              "train or eval reproduces canonical exactly.")

    print(f"\n{'='*100}\nMOVED CELLS (expected, cleaned dataset in train or eval), sorted by |delta|:\n{'='*100}")
    moves.sort(key=lambda x: -abs(x[5]))
    print(f"{'eval':14s}{'rung':16s}{'method':14s}{'old':>9s}{'new':>9s}{'delta':>9s}")
    for ev, rung, method, o, n, d in moves:
        print(f"{ev:14s}{rung:16s}{method:14s}{o:>+9.4f}{n:>+9.4f}{d:>+9.4f}")

    out = NEW_DIR / "qwen_cleanspan_invariance_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval", "rung", "method", "old_prr", "new_prr", "delta", "category"])
        for ev, rung, method, o, n, d in fails:
            w.writerow([ev, rung, method, o, n, d, "VIOLATION"])
        for ev, rung, method, o, n, d in moves:
            w.writerow([ev, rung, method, o, n, d, "expected_move"])
    print(f"\nwrote {out}")

    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
