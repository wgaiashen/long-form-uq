#!/usr/bin/env python
"""STEP 4 verifier -- assert the rendered ProbeDriftLong markdown == the master CSV (no transcription drift).

The master `.md` and `.csv` are both emitted by `assemble_pdl_table.py`, but this re-parses the RENDERED per-rung
tables independently and checks every numeric cell against the CSV, keyed on (rung, eval, method) -- the guard
that would catch a hand-edit or a rendering bug (the same pattern as `verify_partA_consistency.py`). Read-only;
exits 1 on any mismatch.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
CSV = ROOT / "results" / f"pdl_master__{SLUG}.csv"
MD = ROOT / "results" / f"pdl_master__{SLUG}.md"
TOL = 2e-3
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]


def num(cell):
    """parse a rendered cell -> float or None. Strips **bold**, the seed-1 dagger, unicode minus, spaces; a
    blank / middle-dot means 'not measured' -> None."""
    s = cell.strip().replace("*", "").replace("†", "").replace("−", "-").strip()
    if s in ("", "·", "."):
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def main():
    if not CSV.exists() or not MD.exists():
        print(f"MISSING: {CSV if not CSV.exists() else MD}"); sys.exit(1)
    # CSV ground truth: (rung,eval,method) -> prr
    truth = {}
    for r in csv.DictReader(open(CSV)):
        truth[(r["rung"], r["eval"], r["method"])] = float(r["prr"])

    lines = MD.read_text().splitlines()
    fails = []
    checked = 0
    rung = None
    evals = None
    for ln in lines:
        if ln.startswith("## "):                                 # ANY section header resets context; only a
            mh = re.match(r"^## rung = (\S+)", ln)                # "## rung = X" header (re)enters a rung table,
            rung = mh.group(1) if (mh and mh.group(1) in RUNGS) else None  # so the aggregate/seed-1 tables below
            evals = None                                         # the last rung are NOT mis-parsed as rung cells.
            continue
        if rung is None or not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and cells[0] == "method":                       # header row -> eval column order
            evals = cells[1:]
            continue
        if evals is None or cells[0] in ("---", "") or set(cells[0]) <= set("-"):
            continue
        if cells[0].startswith("*") and "supervised" in cells[0]:
            continue
        method = cells[0]
        for j, ev in enumerate(evals):
            if j + 1 >= len(cells):
                continue
            v = num(cells[j + 1])
            key = (rung, ev, method)
            if v is None:
                if key in truth:
                    fails.append(f"{key}: md blank but CSV has {truth[key]:+.3f}")
                continue
            if key not in truth:
                fails.append(f"{key}: md has {v:+.3f} but CSV has no such cell")
                continue
            checked += 1
            if abs(v - truth[key]) > TOL:
                fails.append(f"{key}: md {v:+.3f} != CSV {truth[key]:+.3f} (Δ{v-truth[key]:+.4f})")

    print(f"[verify_pdl_table] checked {checked} rendered cells against the CSV (tol {TOL})")
    if fails:
        print(f"FAIL: {len(fails)} mismatches:")
        for f in fails[:40]:
            print("  ", f)
        sys.exit(1)
    print("PASS: rendered markdown == master CSV on every numeric cell.")


if __name__ == "__main__":
    main()
