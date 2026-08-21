"""GATE 1 — prove the ProbeDriftLong library changed NOTHING, before anything is built on it.

WHY THIS IS THE FIRST THING WRITTEN
-----------------------------------
Extracting the grid into a library is a refactor, and this project's recurring failure is exactly
"refactor -> a number moves -> a tidy story arrives to explain it -> the check that would have
caught it never runs". So the check runs first, and it is cheap and decisive.

WHAT IT PROVES
--------------
For every cell of the canonical grid (8 long evals x 5 rungs, + the Long->Short transfer cells)
and every seed, the library must emit **byte-identical `(train_rows, test_rows)` index lists** to
the current `probedriftlong` / `xl_rungs` code, under `carve="legacy"`.

That is a stronger and far cheaper test than diffing PRR: PRR is a deterministic function of
(which rows, which labels, which scores). If the row lists are identical then no PRR can move,
and the whole 1,664-cell master table is safe by construction. Running the ladder to find that
out would cost ~24h of queue; this costs seconds.

It deliberately runs with `carve="legacy"`. The carve-then-filter change is a SEPARATE,
INTENTIONAL change of the numbers on expertqa + factscore (gate 3). Conflating the two is how a
packaging bug would hide inside a legitimate revision, so they are never tested together.
`--report-new-carve` additionally PREVIEWS the intended change, but never gates on it.

USAGE
    python scripts/checks/library_equivalence.py                 # the gate
    python scripts/checks/library_equivalence.py --report-new-carve
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

# --- the OLD implementations (the incumbents being checked against) --------------------------
import probedriftlong as OLD_PDL          # noqa: E402
import xl_rungs as OLD_XL                 # noqa: E402

# --- the NEW library --------------------------------------------------------------------------
import probe_drift_long as PDL            # noqa: E402
from luq import cache                     # noqa: E402
from luq.config import Config             # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def load_split_and_labels(dataset):
    """(split array, labelled mask, records) straight from the frozen cache — no ML deps."""
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=REGIME.get(dataset, ""))
    key = cache.run_key(MODEL, dataset, "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    lab = PDL.label_of(dataset)

    def val(r):
        try:
            return float(r.get(lab))
        except (TypeError, ValueError):
            return float("nan")

    y = np.array([val(r) for r in recs], dtype=float)
    return np.array([r["split"] for r in recs]), np.isfinite(y), recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--report-new-carve", action="store_true",
                    help="also PREVIEW the carve-then-filter change (never gates)")
    ap.add_argument("--out", default=str(ROOT / "results/g1_library_equivalence.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    evals = list(PDL.LONG_DATASETS) + list(PDL.SHORT_DATASETS)
    print(f"GATE 1 — library equivalence | probe_drift_long {PDL.__version__} | seeds {seeds}\n")

    # ---- 0. taxonomy must agree with BOTH incumbents -----------------------------------------
    print("--- taxonomy (the four duplicated copies must collapse to one) ---")
    ok = True
    for name, old, new in [("LONG", OLD_PDL.LONG, PDL.LONG_DATASETS),
                           ("LONG_SRC", OLD_PDL.LONG_SRC, PDL.LONG_SRC),
                           ("SHORT", OLD_PDL.SHORT, PDL.SHORT_DATASETS),
                           ("FINE", OLD_PDL.FINE, PDL.FINE_FAMILIES),
                           ("XL_TOTAL", OLD_PDL.XL_TOTAL, PDL.XL_TOTAL)]:
        same = (list(old) == list(new)) if isinstance(old, list) else (old == new)
        ok &= same
        print(f"  probedriftlong.{name:9} {'== library' if same else '!! DIFFERS'}")
    for d in PDL.LONG_DATASETS:
        same = OLD_XL.label_of(d) == PDL.label_of(d)
        ok &= same
        if not same:
            print(f"  !! label_of({d}): xl_rungs {OLD_XL.label_of(d)!r} vs library {PDL.label_of(d)!r}")
    print(f"  label_of agrees on all {len(PDL.LONG_DATASETS)} long datasets: "
          f"{all(OLD_XL.label_of(d) == PDL.label_of(d) for d in PDL.LONG_DATASETS)}")

    # ---- 1. load the frozen caches -----------------------------------------------------------
    print("\n--- caches ---")
    splits, labels, recs = {}, {}, {}
    for d in evals:
        try:
            s, m, r = load_split_and_labels(d)
        except FileNotFoundError:
            print(f"  {d:14} no cache -> skipped (cell coverage will say so)")
            continue
        splits[d], labels[d], recs[d] = s, m, r
        n_lab, n, frac = PDL.coverage(m)
        PDL.assert_canonical_order(r, d)                    # the order guard, exercised for real
        print(f"  {d:14} {n:5d} rows  coverage {n_lab:5d}/{n:<5d} = {100*frac:5.1f}%  "
              f"order OK")
    sources = set(splits)

    # ---- 2. cell grid must be identical ------------------------------------------------------
    old_cells = OLD_PDL.cells_long(sources, evals)
    new_cells = PDL.cells_long(sources, evals)
    cells_same = old_cells == new_cells
    ok &= cells_same
    print(f"\n--- cell grid ---\n  {len(old_cells)} cells | "
          f"{'IDENTICAL' if cells_same else '!! DIFFERS'}")
    if not cells_same:
        for a, b in zip(old_cells, new_cells):
            if a != b:
                print(f"    old {a}\n    new {b}")

    # ---- 3. THE GATE: byte-identical row lists, every cell x every seed -----------------------
    print("\n--- row lists (carve='legacy') — THE GATE ---")
    # The old build_rows wants PT[d] = (states, split, y, records); only index [1] is ever read.
    PT = {d: (None, splits[d], None, recs[d]) for d in splits}
    # ...and the old driver filtered unlabelled rows BEFORE splitting, so replicate that here:
    # its arrays were the labelled-only subset. Build the same filtered view.
    f_splits = {d: splits[d][labels[d]] for d in splits}
    f_PT = {d: (None, f_splits[d], None, None) for d in splits}

    n_checked, n_bad = 0, 0
    per_cell = []
    for rung, X, spec in new_cells:
        for sd in seeds:
            old_tr, old_te = OLD_XL.build_rows(X, spec, f_PT, sd, OLD_PDL.sampled_train_idx)
            new_tr, new_te = PDL.build_rows(X, spec, f_splits, sd, carve="legacy")
            same = (old_tr == new_tr) and (old_te == new_te)
            n_checked += 1
            n_bad += (not same)
            if not same:
                print(f"  !! MISMATCH {rung}/{X}/seed{sd}: "
                      f"train {len(old_tr)}->{len(new_tr)}, test {len(old_te)}->{len(new_te)}")
            per_cell.append({"rung": rung, "eval": X, "seed": sd, "identical": bool(same),
                             "n_train": len(new_tr), "n_test": len(new_te)})
    ok &= (n_bad == 0)
    print(f"  {n_checked - n_bad}/{n_checked} cell x seed row lists BYTE-IDENTICAL")

    # ---- 4. preview of the deliberate change (never gates) -----------------------------------
    preview = {}
    if args.report_new_carve:
        print("\n--- PREVIEW: carve-then-filter (gate 3's deliberate change — NOT gated here) ---")
        print(f"  {'dataset':14} {'legacy test':>11} {'new scored':>11} {'new carved':>11}   verdict")
        for d in PDL.LONG_DATASETS:
            if d not in splits:
                continue
            _, old_te = PDL.eval_split(f_splits[d], carve="legacy")
            _, new_te = PDL.eval_split(splits[d], labels[d], carve="all-rows")
            # map both into ORIGINAL row space to compare like with like
            orig = np.where(labels[d])[0]
            old_orig, new_orig = set(orig[old_te]), set(int(i) for i in new_te)
            n_carved = len(PDL.eval_split(splits[d], np.ones(len(splits[d]), bool),
                                          carve="all-rows")[1])
            moved = old_orig != new_orig
            print(f"  {d:14} {len(old_orig):11d} {len(new_orig):11d} {n_carved:11d}   "
                  + ("MOVES" if moved else "identical — does not move"))
            preview[d] = {"legacy_test": len(old_orig), "new_scored": len(new_orig),
                          "new_carved": n_carved, "moves": bool(moved)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"library_version": PDL.__version__, "verdict": "PASS" if ok else "FAIL",
                   "n_cells": len(new_cells), "n_checked": n_checked, "n_mismatch": n_bad,
                   "coverage": {d: PDL.coverage(labels[d]) for d in splits},
                   "cells": per_cell, "new_carve_preview": preview}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("=" * 78)
    print(f"GATE 1: {'PASS — packaging is inert, safe to build on' if ok else 'FAIL — DO NOT PROCEED'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
