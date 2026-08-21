#!/usr/bin/env python
"""Verifier for the ProbeDrift-XL master table — asserts the rendered MD matches the CSV, AND that the
grid is actually complete.

Mirrors `verify_pdl_table.py` (re-parse the rendered tables independently, check every numeric cell
against the CSV keyed on (rung, eval, method)) and adds the check that one exists at all:

**THE CELL-COUNT ASSERTION.** Tonight the HBO driver silently reported **4 cells instead of 10**
because a source-pool filter dropped three rungs with no warning, and the run looked successful. A
verifier that only checks MD-vs-CSV consistency would have passed that happily — both would have been
consistently wrong. So this exits 1 on a short grid and NAMES the missing cells.

Read-only. Exits 1 on any mismatch or shortfall.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
CSV_PATH = ROOT / "results" / f"xl_master__{SLUG}.csv"
MD_PATH = ROOT / "results" / f"xl_master__{SLUG}.md"
TOL = 2e-3

XL_EVALS = ["sciq", "trivia_qa", "pubmed_qa", "med_quad", "asqa",
            "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask", "LOO", "DiffTask", "OneDatasetDiffTask"]
TARGET_CELLS = len(XL_EVALS) * len(RUNGS)          # 50


def num(cell):
    """Rendered cell -> float or None. A blank means NOT MEASURED and must stay None: turning it into
    0.0 would report an absence as a measurement, which is the silent-default bug class."""
    s = cell.strip().replace("*", "").replace("†", "").replace("−", "-").strip()
    if s in ("", "·", "."):
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


def main():
    for p in (CSV_PATH, MD_PATH):
        if not p.exists():
            print(f"MISSING: {p}")
            sys.exit(1)

    truth = {}
    for r in csv.DictReader(open(CSV_PATH)):
        truth[(r["rung"], r["eval"], r["method"])] = float(r["prr"])

    # ---- CHECK 1: grid completeness (the one that would have caught the HBO 4-vs-10 bug)
    present = {(rg, ev) for (rg, ev, _m) in truth}
    missing = [(rg, ev) for ev in XL_EVALS for rg in RUNGS if (rg, ev) not in present]
    print(f"CHECK 1 — coverage: {len(present)}/{TARGET_CELLS} cells")
    if missing:
        print(f"  {len(missing)} MISSING, named:")
        for rg, ev in missing:
            print(f"     {rg:20s} {ev}")
    else:
        print("  complete grid")

    # ---- CHECK 2: every rendered cell matches the CSV
    lines = MD_PATH.read_text().splitlines()
    fails, checked = [], 0
    rung, evals = None, None
    for ln in lines:
        m = re.match(r"^### rung = (\S+)", ln)
        if m:
            rung, evals = m.group(1), None
            continue
        if ln.startswith("| method |"):
            evals = [c.strip() for c in ln.strip("|").split("|")[1:]]
            continue
        if rung and evals and ln.startswith("|") and not ln.startswith("|---"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            method, vals = cells[0], cells[1:]
            for ev, cell in zip(evals, vals):
                got = num(cell)
                exp = truth.get((rung, ev, method))
                if got is None and exp is None:
                    continue
                if got is None or exp is None or abs(got - exp) > TOL:
                    fails.append((rung, ev, method, got, exp))
                else:
                    checked += 1
    print(f"CHECK 2 — rendered vs CSV: {checked} cells agree, {len(fails)} mismatched")
    for f in fails[:20]:
        print(f"  {f[0]}/{f[1]}/{f[2]}: md={f[3]} csv={f[4]}")

    # ---- CHECK 3: no method is reported on a partial cell set without that being visible
    by_method = {}
    for (rg, ev, m) in truth:
        by_method.setdefault(m, set()).add((rg, ev))
    partial = {m: len(c) for m, c in by_method.items() if len(c) < len(present)}
    print(f"CHECK 3 — methods on a PARTIAL cell set: {len(partial)}")
    for m, n in sorted(partial.items(), key=lambda t: t[1]):
        print(f"     {m:28s} {n}/{len(present)} cells  never average this against a full-coverage method")

    if fails or missing:
        print("\nVERIFY FAILED — do not report this table.")
        sys.exit(1)
    print("\nXL master table verified: complete grid, rendering matches the CSV.")


if __name__ == "__main__":
    main()
