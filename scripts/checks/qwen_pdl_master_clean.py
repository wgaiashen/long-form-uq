#!/usr/bin/env python
"""Assemble the CLEAN-SPAN Qwen2.5-14B per-eval ladder CSVs into one shadow master table.

One-off sibling of `qwen_pdl_master.py` for the 2026-08-11/12 clean-span correction. That script's
read glob (`results/probedriftlong__<slug>__<eval>.csv`) and write path
(`results/pdl_master__<slug>.csv`) are both hardcoded to the canonical population on purpose -- so
rather than bolt an override onto it, this reads the clean-span shadow CSVs
(`results/analysis/pdl_qwenclean_<eval>__<slug>.csv`, written by `qwen_pdl_ladder_clean.sbatch`) and
writes to a separate shadow path. Same long-format columns `replication_claims.py --csv` expects
(method, rung, eval, prr_mean), so the M2 scorer can run against either population unmodified.

    python scripts/checks/qwen_pdl_master_clean.py --strict
    python scripts/checks/replication_claims.py --population qwen \
        --csv results/analysis/pdl_master_qwenclean__Qwen_Qwen2.5-14B.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"
LONG_EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum",
              "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
METHODS = ["floor_sum", "floor_ppl", "floor_min", "fair_floor", "saplma", "uniform", "attention",
           "wmsp_norm", "wmsp_shrink2", "wmsp_shrink10"]


def slug(model):
    return model.replace("/", "_")


def load(model):
    cells, seen_files, missing = {}, [], []
    for ev in LONG_EVALS:
        p = ANALYSIS / f"pdl_qwenclean_{ev}__{slug(model)}.csv"
        if not p.exists():
            missing.append(ev)
            continue
        seen_files.append(p.name)
        for r in csv.DictReader(open(p)):
            if r["method"].startswith("VERDICT"):
                continue
            key = (r["eval"], r["rung"], r["method"])
            if key in cells:
                raise SystemExit(f"duplicate cell {key} in {p.name} -- refusing to average silently")
            cells[key] = r
    return cells, seen_files, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    cells, files, missing_evals = load(args.model)
    out_csv = ANALYSIS / f"pdl_master_qwenclean__{slug(args.model)}.csv"

    print(f"CLEAN-SPAN QWEN MASTER (shadow)   model={args.model}")
    print(f"sources ({len(files)}/8): {', '.join(files) if files else '(none)'}")
    if missing_evals:
        print(f"MISSING: {', '.join(missing_evals)}")

    present = sum(1 for ev in LONG_EVALS for rg in RUNGS
                  if any((ev, rg, m) in cells for m in METHODS))
    total = len(LONG_EVALS) * len(RUNGS)
    print(f"COVERAGE: {present}/{total} eval x rung cells")

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "eval", "rung", "method", "prr_mean", "prr_std", "n_seeds", "carve"])
        for ev in LONG_EVALS:
            for rg in RUNGS:
                for m in METHODS:
                    r = cells.get((ev, rg, m))
                    if r is None:
                        continue
                    w.writerow([args.model, ev, rg, m, r.get("prr_mean", ""), r.get("prr_std", ""),
                                r.get("n_seeds", ""), r.get("carve", "")])
                    n_written += 1
    print(f"wrote {out_csv} ({n_written} rows)")

    if args.strict and (present < total or missing_evals):
        print("STRICT: coverage incomplete -> exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
