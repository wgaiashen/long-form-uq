#!/usr/bin/env python
"""Sweep the codebase for SILENT DROPS — places where something real is discarded with no message.

WHY THIS EXISTS
---------------
On 2026-08-03 four separate instances of one mechanism surfaced within a few hours, each found by
accident rather than by looking:

  1. `assemble_pdl_table.ALIAS.get(m, "__skip__")` dropped any unrecognised method — and `ptrue`,
     `lookback`, `linear` were never in ALIAS, so the table backing "our method beats existing probes"
     contained NO existing probes.
  2. Making that loud immediately exposed the `zavg_*` ensembles: computed for weeks, never rendered.
  3. `assemble_xl_table` read only the `rung` column, but `ood_onegrid` names it `setting` — every
     supervised-baseline row would have been dropped.
  4. `ood_onegrid.EVALS` and `contribution_ladder.EVALS` were hard-coded to 3 datasets, so a run without
     `--evals` silently produced a 15-cell grid while reporting as the full 50-cell XL run.

The standing rule is to sweep the MECHANISM, not the name: "searching for `_vs_floor` found 4 drivers;
searching for the actual computation found 15." This script is that sweep, kept so the class stays
closed rather than being rediscovered a fifth time.

WHAT IT FLAGS (each is a place an absence can masquerade as a measurement)
-------------------------------------------------------------------------
  A. dict-lookup drop      `.get(x, SENTINEL)` / `x not in D` followed by `continue`, with no print
  B. silent except         `except ...: pass` / `except ...: continue`
  C. hard-coded cohorts    module-level EVALS/DATASETS lists that may under-cover the 10-dataset grid
  D. absence -> number     `.get(k, 0)` / `or 0` / `fillna(0)` on a value column (not-measured becoming 0)
  E. glob-and-take-first   `glob(...)[0]` — picks one match with no check that it is the right one

⚠️ This is a LINT, not a proof. Every hit needs judgement: some drops are correct and documented. The
output is a worklist, and a hit that is deliberate should get a comment saying so, which is also what
makes it disappear from a future reading of this report.

    python scripts/checks/sweep_silent_drops.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = [ROOT / "scripts", ROOT / "src" / "luq"]
SKIP_PARTS = {"__pycache__", ".egg-info", ".ipynb_checkpoints"}

# The ten-dataset universe the grids are supposed to cover.
FULL_UNIVERSE = {"sciq", "trivia_qa", "pubmed_qa", "med_quad", "asqa",
                 "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"}

PATTERNS = [
    ("A dict-drop",
     re.compile(r"\.get\([^)]*,\s*(\"__skip__\"|'__skip__'|None)\s*\)")),
    ("B silent-except",
     re.compile(r"except[^:]*:\s*(pass|continue)\s*$")),
    ("D absence->0",
     re.compile(r"\.get\([^)]*,\s*0(\.0)?\s*\)|\bor\s+0\.0\b|fillna\(\s*0")),
    ("E glob[0]",
     re.compile(r"glob\([^)]*\)\s*\[\s*0\s*\]|glob\.glob\([^)]*\)\[0\]")),
]

# A comment within this many lines of a hit that acknowledges the drop makes it "reviewed".
ACK = re.compile(r"⚠️|LOUD|loud|deliberate|intentional|by design|never zero|reported|skip(ped)? loudly",
                 re.I)


def iter_py():
    for d in SCAN_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.py")):
            if any(part in SKIP_PARTS for part in p.parts):
                continue
            yield p


def acknowledged(lines, i, window=4):
    lo, hi = max(0, i - window), min(len(lines), i + window + 1)
    return any(ACK.search(lines[j]) for j in range(lo, hi))


def main():
    findings = {tag: [] for tag, _ in PATTERNS}
    findings["C cohort"] = []

    for p in iter_py():
        try:
            lines = p.read_text().splitlines()
        except Exception:
            continue
        rel = p.relative_to(ROOT)

        for i, ln in enumerate(lines):
            for tag, rx in PATTERNS:
                if rx.search(ln):
                    findings[tag].append((rel, i + 1, ln.strip()[:100], acknowledged(lines, i)))

        # C: module-level cohort lists that under-cover the universe
        for i, ln in enumerate(lines):
            m = re.match(r"^(EVALS|DATASETS|LONG|SHORT|ALL)\s*=\s*\[(.*)\]", ln)
            if not m:
                continue
            got = set(re.findall(r"[\"']([a-z_0-9]+)[\"']", m.group(2)))
            if got and got < FULL_UNIVERSE and len(got) < len(FULL_UNIVERSE):
                missing = sorted(FULL_UNIVERSE - got)
                findings["C cohort"].append(
                    (rel, i + 1, f"{m.group(1)} covers {len(got)}/10, missing: {', '.join(missing)}",
                     acknowledged(lines, i)))

    total = sum(len(v) for v in findings.values())
    unack = sum(1 for v in findings.values() for f in v if not f[3])
    print(f"SILENT-DROP SWEEP | {total} hits across {len(list(iter_py()))} files "
          f"| {unack} NOT acknowledged in a nearby comment\n")

    for tag in sorted(findings):
        hits = findings[tag]
        if not hits:
            continue
        new = [h for h in hits if not h[3]]
        print(f"=== {tag}: {len(hits)} hits ({len(new)} unreviewed) ===")
        for rel, ln, txt, ack in hits:
            if ack:
                continue          # reviewed: a nearby comment already owns this one
            print(f"  {rel}:{ln}\n      {txt}")
        print()

    print("⚠️ A hit is a QUESTION, not a defect. Two things make one real:")
    print("   (a) the dropped thing was actually computed somewhere, and")
    print("   (b) nothing downstream says it went missing.")
    print("Where a drop is deliberate, add a comment saying so — that also clears it from this report.")
    return 1 if unack else 0


if __name__ == "__main__":
    sys.exit(main())
