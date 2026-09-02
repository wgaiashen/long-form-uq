#!/usr/bin/env python
"""Derive which grid cells the span correction can move, from the driver's own source construction.

A correction that changes one dataset's responses does not affect only that dataset's results. Any
cell that trains on the corrected dataset can move on a target that did not change, and that is the
expected outcome rather than a validation failure. The affected set is therefore a property of the
realised training sources, not of the evaluation target.

  affected  the corrected dataset is the evaluation population, or appears in the realised sources
  control   the corrected dataset appears nowhere in the cell

Controls are what an invariance check tests: run the same code and seeds on both populations, and a
control cell must reproduce. This is derived from `cells_long` every time rather than carried over
from another experiment, because an auxiliary experiment can build its sources differently and the
count would then be wrong in a way nothing would flag.

    python scripts/checks/ch6_cleanv2_affected_cells.py
    python scripts/checks/ch6_cleanv2_affected_cells.py --against results/cleanv2/pdl_cleanv2_master__meta-llama_Meta-Llama-3.1-8B.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_drift_long import cells_long                       # noqa: E402

CORRECTED = "med_quad"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OUT = ROOT / "results" / "analysis" / "ch6_cleanv2_affected_cells.csv"


def derive(corrected=CORRECTED, sources=LONG, evals=LONG):
    rows = []
    for rung, ev, srcs in cells_long(sources, evals):
        names = [s for s, _cap in srcs]
        touched = (ev == corrected) or (corrected in names)
        rows.append({"rung": rung, "eval": ev,
                     "sources": "+".join(names),
                     "corrected_is_eval": ev == corrected,
                     "corrected_in_sources": corrected in names,
                     "role": "affected" if touched else "control"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--against", default=None,
                    help="an assembled master to reconcile against; its provenance column must mark "
                         "exactly the derived affected cells as recomputed.")
    args = ap.parse_args()

    rows = derive()
    aff = {(r["eval"], r["rung"]) for r in rows if r["role"] == "affected"}
    print("=" * 92)
    print(f"CELL AFFECTEDNESS  corrected dataset={CORRECTED}  cells={len(rows)}")
    print("=" * 92)
    print(f"{'rung':16s}{'eval':14s}{'role':10s}sources")
    for r in sorted(rows, key=lambda r: (r["eval"], r["rung"])):
        print(f"{r['rung']:16s}{r['eval']:14s}{r['role']:10s}{r['sources']}")
    print(f"\naffected {len(aff)}   control {len(rows) - len(aff)}")

    if args.against:
        path = Path(args.against)
        stamped = set()
        for r in csv.DictReader(open(path)):
            if r.get("provenance") == "recomputed_cleanv2":
                stamped.add((r["eval"], r["rung"]))
        agree = aff == stamped
        print(f"\nreconciliation against {path.name}: "
              f"{'AGREE' if agree else 'DISAGREE'} "
              f"(derived {len(aff)}, stamped {len(stamped)})")
        if not agree:
            print(f"  differing cells: {sorted(aff ^ stamped)}")
            raise SystemExit("the derived affected set does not match the assembled master; "
                             "the source construction assumed here is not the one that was run")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
