#!/usr/bin/env python
"""Assemble the Qwen2.5-14B ProbeDriftLong per-eval ladder CSVs into one master table.

WHY THIS IS NOT `assemble_pdl_table.py --model`. That assembler joins about a dozen source globs
accumulated over the Llama campaign (`pdl_fam_*`, `probedriftlong_*_widened_wmsp`,
`fixed_prior_ladder*`, `multihead_*`, …). None of those have a Qwen counterpart, so pointing it at
this population would produce a table that is blank by construction — which reads as "not measured"
rather than "does not exist". Qwen has exactly ONE source: the eight per-eval CSVs written by
`probedriftlong.py --model Qwen/Qwen2.5-14B`. Concatenating those is the whole job.

WHAT IT DOES **NOT** DO — deliberately. It computes no M2 replication verdict. That is
`scripts/checks/replication_claims.py` (committed 2026-08-10), which is the test definition for
R1a/R1b/R2/R2-desc/R3/R4′ and scores BOTH populations from one code path. Keeping assembly separate
from scoring is the point: this file's output is the input that scorer refuses to run on unless the
grid is complete. Until it existed the tests lived only as prose in the prereg, and re-deriving them
per model would have risked running a subtly different test on each and calling the difference a
replication result.

Coverage is reported UP FRONT and every genuinely-absent cell stays BLANK, never zero.

    python scripts/checks/qwen_pdl_master.py --strict     # exit 1 unless all 40 cells are present
    python scripts/checks/replication_claims.py --population qwen
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"
LONG_EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum",
              "expertqa", "factscore"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
OOD_RUNGS = RUNGS[1:]

# The Tier-1 method set M2 §4 names, in reporting order. Anything the ladder emits that is not here
# is still carried into the CSV -- this list only fixes the ORDER and the coverage denominator.
METHODS = ["floor_sum", "floor_ppl", "floor_min", "fair_floor", "saplma", "uniform", "attention",
           "wmsp_norm", "wmsp_shrink2", "wmsp_shrink10", "wmsp_seg_flat", "wmsp_seg_softmax",
           "wmsp_blondel", "wmsp_shrink2_blondel", "wmsp_shrink10_blondel"]
# M2 §4 also names these two, and they are NOT computable on this population: no ptrue and no
# lookback pooled feature cache exists for Qwen (only __saplma.npz). Named here so the gap is
# reported as a gap rather than being invisible.
DECLARED_BUT_ABSENT = ["ptrue", "lookback"]


def slug(model):
    return model.replace("/", "_")


def load(model):
    """rows keyed (eval, rung, method) -> dict. Fails loud on a duplicate key."""
    cells, seen_files, missing = {}, [], []
    for ev in LONG_EVALS:
        p = RESULTS / f"probedriftlong__{slug(model)}__{ev}.csv"
        if not p.exists():
            missing.append(ev)
            continue
        seen_files.append(p.name)
        for r in csv.DictReader(open(p)):
            if r["method"].startswith("VERDICT"):
                continue                                  # margins/CIs, not PRRs -- kept out
            key = (r["eval"], r["rung"], r["method"])
            if key in cells:
                raise SystemExit(f"duplicate cell {key} in {p.name} -- refusing to average silently")
            cells[key] = r
    return cells, seen_files, missing


def prr(cells, ev, rung, method):
    r = cells.get((ev, rung, method))
    if r is None:
        return None
    v = r.get("prr_mean", "")
    if v in ("", "nan", "None"):
        return None
    return float(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless every eval x rung cell is present")
    args = ap.parse_args()

    cells, files, missing_evals = load(args.model)
    out_csv = RESULTS / f"pdl_master__{slug(args.model)}.csv"
    out_md = RESULTS / f"pdl_master__{slug(args.model)}.md"

    print("=" * 104)
    print(f"QWEN PROBEDRIFTLONG MASTER   model={args.model}")
    print("Population: ProbeDriftLong, 8 long evals x cells_long, 3 seeds, fp32 + eager, layer 23")
    print("            (fixed rule ceil(N/2)-1), judge gpt-5-mini. expertqa/factscore are scored on")
    print("            their judge-COVERED subsets (1603/2016 and 466/500) -- a null label is the")
    print("            judge declining, not incomplete labelling.")
    print("=" * 104)
    print(f"\nsources ({len(files)}/8): {', '.join(files) if files else '(none)'}")
    if missing_evals:
        print(f"⚠️ MISSING EVALS (not yet run or failed): {', '.join(missing_evals)}")

    # ---------------- coverage, stated before any number ----------------
    present = sum(1 for ev in LONG_EVALS for rg in RUNGS
                  if any((ev, rg, m) in cells for m in METHODS))
    total = len(LONG_EVALS) * len(RUNGS)
    print(f"\nCOVERAGE: {present}/{total} eval x rung cells")
    gaps = [(ev, rg) for ev in LONG_EVALS for rg in RUNGS
            if not any((ev, rg, m) in cells for m in METHODS)]
    if gaps:
        print(f"  absent cells ({len(gaps)}), named not hidden:")
        for ev, rg in gaps:
            print(f"    {ev} x {rg}")
    print(f"\n⚠️ DECLARED IN M2 §4 BUT NOT COMPUTABLE ON THIS POPULATION: {', '.join(DECLARED_BUT_ABSENT)}")
    print("   No ptrue and no lookback pooled feature cache exists for Qwen (features/ holds only")
    print("   __saplma.npz for all 8). Their columns are ABSENT, not zero. Producing them needs a")
    print("   separate extraction per method; none of M2's registered claims (R1a/R1b/R2/R2-desc/")
    print("   R3/R4') reads either of them, so the replication itself is unaffected.")

    # ---------------- long-format master ----------------
    RESULTS.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "eval", "rung", "method", "prr_mean", "prr_std", "n_seeds",
                    "carve", "eval_med_len", "source"])
        for ev in LONG_EVALS:
            for rg in RUNGS:
                for m in METHODS:
                    r = cells.get((ev, rg, m))
                    if r is None:
                        continue                          # absent stays absent; no placeholder row
                    w.writerow([args.model, ev, rg, m, r.get("prr_mean", ""), r.get("prr_std", ""),
                                r.get("n_seeds", ""), r.get("carve", ""), r.get("eval_med_len", ""),
                                f"probedriftlong__{slug(args.model)}__{ev}.csv"])
                    n_written += 1
    print(f"\nwrote {out_csv}  ({n_written} rows)")

    # ---------------- rendered tables ----------------
    lines = [f"# ProbeDriftLong master — `{args.model}`", "",
             "> *Population: ProbeDriftLong, 8 long evals × `cells_long`, 3 seeds, "
             "`Qwen/Qwen2.5-14B` (base), fp32 + eager, judge label gpt-5-mini, layer 23 by the "
             "fixed rule `ceil(N/2) − 1`.*", "",
             f"> Coverage: **{present}/{total}** eval × rung cells. "
             f"`ptrue` and `lookback` are **absent** (no feature cache on this population), not zero.",
             ""]
    for rg in RUNGS:
        lines += [f"## {rg}", "", "| method | " + " | ".join(LONG_EVALS) + " | mean |",
                  "|---" * (len(LONG_EVALS) + 2) + "|"]
        for m in METHODS:
            vals = [prr(cells, ev, rg, m) for ev in LONG_EVALS]
            got = [v for v in vals if v is not None]
            cellstr = " | ".join(f"{v:+.3f}" if v is not None else "" for v in vals)
            mean = f"**{np.mean(got):+.3f}**" if len(got) == len(LONG_EVALS) else (
                f"{np.mean(got):+.3f} ({len(got)}/8)" if got else "")
            lines.append(f"| `{m}` | {cellstr} | {mean} |")
        lines.append("")

    # cross-dataset OOD aggregate: average the 4 OOD rungs within an eval FIRST, so each dataset
    # contributes once. The old "32 OOD cells" framing was 4x pseudo-replication for the free
    # floors, whose PRR does not depend on the training pool (M2 §2, carried verbatim).
    lines += ["## Cross-dataset OOD aggregate (n = 8 datasets, NOT 32 cells)", "",
              "> Each eval's four OOD rungs are averaged **first**, so a dataset contributes one "
              "number. The unsupervised floors do not depend on the training pool, so their four "
              "OOD rungs are identical — treating them as 32 cells would be 4× pseudo-replication.",
              "", "| method | mean OOD PRR | sd across datasets | n datasets |", "|---|---|---|---|"]
    for m in METHODS:
        per_ds = []
        for ev in LONG_EVALS:
            got = [prr(cells, ev, rg, m) for rg in OOD_RUNGS]
            got = [v for v in got if v is not None]
            if got:
                per_ds.append(float(np.mean(got)))
        if per_ds:
            lines.append(f"| `{m}` | {np.mean(per_ds):+.4f} | {np.std(per_ds, ddof=1) if len(per_ds) > 1 else float('nan'):.4f} "
                         f"| {len(per_ds)}/8 |")
        else:
            lines.append(f"| `{m}` | | | 0/8 |")
    lines += ["", "## Not computable on this population", "",
              "| method | why |", "|---|---|",
              "| `ptrue` | no `ptrue_accurate` pooled feature cache for Qwen — needs its own extraction |",
              "| `lookback` | no `lookback` feature cache for Qwen — needs the attention-ratio extraction |",
              "", "Neither is read by any M2 registered claim.", ""]
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")

    print("\nNo replication verdict is computed here -- that is scripts/checks/replication_claims.py")
    print("   (committed 2026-08-10), the test definition for R1a/R1b/R2/R2-desc/R3/R4', which runs")
    print("   over BOTH populations from one code path and refuses a partial grid. Next:")
    print("     python scripts/checks/replication_claims.py --population qwen")

    if args.strict and (gaps or missing_evals):
        print("\nSTRICT: coverage incomplete -> exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
