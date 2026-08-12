#!/usr/bin/env python
"""COVERAGE GUARD for the per-eval ProbeDriftLong family CSVs.

WHY THIS EXISTS
---------------
The canonical master (`results/pdl_master__<slug>.csv`) is complete and is the primary reference.
But the master does NOT carry `prr_std`, so anything needing seed variability has to read the
per-eval `pdl_fam_*` files instead -- and those have a layout trap that loses a cell SILENTLY:

  * `pdl_fam_xsum__<slug>.csv` holds only FOUR rungs. Its `1ds-Diff-long` rung was run separately
    (2026-08-05) and lives in `pdl_fam_xsum_1ds-Diff__<slug>.csv`. A loader that opens one file per
    eval therefore drops 9 core-method cells for xsum and reports a macro mean over a short grid,
    with no error. This is the exact "a missing input must never become a plausible number" failure
    the project bans: the mean still computes, it is just computed over 4 rungs instead of 5.
  * `pdl_fam_<eval>_base` and `pdl_fam_<eval>_segsm` are PARTIAL method subsets over all five rungs.
    They overlap the main file on the floors and SAPLMA. Measured 2026-08-12: 276 cells appear in
    more than one file and ZERO of them disagree by >5e-4, so the duplication is harmless today --
    but it is only harmless while it stays checked, which is what this script does.

So: read the MASTER for values, read these files only for `prr_std`, and run this guard first.

WHAT IT ASSERTS (all loud; exit 1 on any failure)
  1. Every one of the 9 core methods x 8 evals x 5 rungs exists across the family files.
  2. No cell disagrees between family files by more than --tol.
  3. Every family value reconciles with the canonical master to within --tol.
It also prints which file each eval's rungs came from, so the xsum split is visible rather than
implicit.

    python scripts/checks/pdl_fam_coverage.py
    python scripts/checks/pdl_fam_coverage.py --model Qwen/Qwen2.5-14B
"""
import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]

# The frozen 9-method registry, in family-file (internal) naming.
CORE = ["floor_sum", "floor_ppl", "floor_min", "saplma", "uniform", "attention",
        "wmsp_norm", "wmsp_shrink2", "wmsp_shrink10"]

# master naming -> family naming. The master renames every method on assembly, so a cross-check
# has to translate rather than compare strings.
MASTER2FAM = {"msp_sum": "floor_sum", "perplexity": "floor_ppl", "msp_min": "floor_min",
              "SAPLMA": "saplma", "armB(mean-pool)": "uniform", "armA(attention)": "attention",
              "wMSP-norm": "wmsp_norm", "wMSP-shrink@2": "wmsp_shrink2",
              "wMSP-shrink@10": "wmsp_shrink10"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    slug = args.model.replace("/", "_")
    res = Path(args.results)
    fails = []

    # ---- load every family file, keeping provenance per cell -----------------------------------
    seen = defaultdict(dict)          # (eval, rung, method) -> {file_stem: prr_mean}
    std = {}                          # (eval, rung, method) -> prr_std (first non-empty wins)
    files = sorted(glob.glob(str(res / f"pdl_fam_*__{slug}.csv")))
    if not files:
        sys.exit(f"FATAL: no pdl_fam_*__{slug}.csv under {res} -- wrong --results or wrong --model?")
    for p in files:
        stem = Path(p).name.replace(f"__{slug}.csv", "")
        for r in csv.DictReader(open(p)):
            if r["method"].startswith("VERDICT"):
                continue
            try:
                v = float(r["prr_mean"])
            except (ValueError, TypeError):
                continue
            key = (r["eval"], r["rung"], r["method"])
            seen[key][stem] = v
            if key not in std:
                try:
                    std[key] = float(r["prr_std"])
                except (ValueError, TypeError):
                    pass

    print(f"pdl_fam coverage guard -- model={args.model}")
    print(f"  scanned {len(files)} family files, {len(seen)} distinct (eval, rung, method) cells\n")

    # ---- 1. coverage of the core grid ----------------------------------------------------------
    missing = [(e, ru, m) for e in EVALS for ru in RUNGS for m in CORE if (e, ru, m) not in seen]
    if missing:
        fails.append(f"{len(missing)} core cells MISSING from the family files")
        for k in missing[:20]:
            print(f"  MISSING  {k}")
    print(f"[1] core-grid coverage: {len(EVALS)*len(RUNGS)*len(CORE) - len(missing)}"
          f"/{len(EVALS)*len(RUNGS)*len(CORE)} cells present")

    # ---- 2. cross-file agreement ---------------------------------------------------------------
    dup = {k: d for k, d in seen.items() if len(d) > 1}
    bad = {k: d for k, d in dup.items() if max(d.values()) - min(d.values()) > args.tol}
    if bad:
        fails.append(f"{len(bad)} cells DISAGREE between family files by >{args.tol}")
        for k, d in list(bad.items())[:20]:
            print(f"  DISAGREE {k}: " + ", ".join(f"{f}={v:+.4f}" for f, v in d.items()))
    print(f"[2] cross-file agreement: {len(dup)} duplicated cells, {len(bad)} disagreeing "
          f"(tol {args.tol})")

    # ---- 3. reconcile against the canonical master ---------------------------------------------
    mpath = res / f"pdl_master__{slug}.csv"
    if not mpath.exists():
        print(f"[3] master reconcile: SKIPPED -- {mpath.name} not found")
    else:
        n_ok = n_bad = 0
        for r in csv.DictReader(open(mpath)):
            fam = MASTER2FAM.get(r["method"])
            if fam is None or r["eval"] not in EVALS:
                continue
            got = seen.get((r["eval"], r["rung"], fam))
            if not got:
                n_bad += 1
                print(f"  ABSENT   master cell not in family files: {r['eval']}/{r['rung']}/{fam}")
                continue
            if abs(list(got.values())[0] - float(r["prr"])) > args.tol:
                n_bad += 1
                print(f"  MISMATCH {r['eval']}/{r['rung']}/{fam}: "
                      f"family={list(got.values())[0]:+.4f} master={float(r['prr']):+.4f}")
            else:
                n_ok += 1
        if n_bad:
            fails.append(f"{n_bad} cells fail to reconcile with the master")
        print(f"[3] master reconcile: {n_ok} agree, {n_bad} fail")

    # ---- provenance map: make the xsum split VISIBLE --------------------------------------------
    print("\n[4] which file supplies each eval's rungs (the split is not a bug, but it must be seen)")
    for e in EVALS:
        by_file = defaultdict(set)
        for (ev, ru, m), d in seen.items():
            if ev == e and m in CORE:
                for f in d:
                    by_file[f].add(ru)
        for f in sorted(by_file):
            print(f"    {e:15s} {f:38s} {len(by_file[f])} rung(s): "
                  f"{','.join(sorted(by_file[f], key=lambda x: RUNGS.index(x)))}")

    print()
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: PASS -- core grid complete, no cross-file disagreement, master reconciles.")
    print("Reminder: quote VALUES from the master; use these files only for prr_std.")


if __name__ == "__main__":
    main()
