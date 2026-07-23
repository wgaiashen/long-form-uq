"""THE unified results table: canonical methods x datasets x evaluation family, from every results CSV.

Answers, in one place, "what do we actually have?" — which was previously spread over dozens of CSVs using
66 different names for ~25 methods.

TWO EVALUATION FAMILIES, determined EMPIRICALLY from the rung names present (not guessed from filenames):
  STANDARD  ProbeDrift(XL) ladder: ID / SameTask / LOO / OneDatasetDiffTask / DiffTask.
            Training pool is MIXED short+long. Within it, the core-5 eval rows (sciq, trivia_qa, pubmed_qa,
            xsum, cnn_dailymail) are the FAITHFUL ProbeDrift reproduction via get_training_spec; the XL eval
            rows (med_quad, samsum, expertqa, asqa) are our extension via the task-family taxonomy.
  LONG      ProbeDriftLong: rungs suffixed `-long` (+ Long->Short). Training pool is long-form ONLY.

CELL SELECTION. A (family, eval, method, rung) cell can appear in several CSVs from different waves. We take
the MOST RECENTLY MODIFIED file containing it and record which file that was, so every number in the table is
traceable to one artifact. Reported per cell: PRR at ID and the mean over OOD rungs.

Read-only: touches no CSV, no driver, no running job.

    python scripts/checks/unified_results_table.py                 # markdown to stdout
    python scripts/checks/unified_results_table.py --coverage      # coverage matrix only
"""
import argparse
import csv as _csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq.method_names import canonical, family  # noqa: E402

EVALS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail",
         "med_quad", "samsum", "expertqa", "asqa"]
CORE = {"sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail"}


def load():
    """cell[(fam, eval, method, rung)] = (prr, mtime, filename), keeping the newest per cell."""
    cell = {}
    for f in glob.glob(str(ROOT / "results" / "*.csv")):
        if "_baseline_" in f:
            continue
        try:
            rows = list(_csv.DictReader(open(f)))
        except Exception:
            continue
        if not rows:
            continue
        k = rows[0].keys()
        if "eval" not in k:
            continue
        # The method column is named `method`, `mode`, OR `variant` depending on the driver wave.
        # Missing `variant` was why weighted_msp_all_variants (the shrink@2/@10/Blondel numbers on the
        # STANDARD ladder -- our PRIMARY method on our PRIMARY ladder) was invisible in the table (2026-07-23).
        mk = next((c for c in ("method", "mode", "variant") if c in k), None)
        if not mk:
            continue
        mt = os.path.getmtime(f)
        base = os.path.basename(f)
        rungs = {r.get("rung", "") for r in rows}
        fam = "LONG" if any(str(x).endswith("-long") or x == "Long->Short" for x in rungs) else "STANDARD"
        for r in rows:
            m = r.get(mk, "")
            if not m or m.startswith("VERDICT"):
                continue
            e, rg = r.get("eval", ""), r.get("rung", "")
            if not e:
                continue
            try:
                v = float(r.get("prr_mean", r.get("prr", "")))
            except (TypeError, ValueError):
                continue
            key = (fam, e, canonical(m), rg)
            if key not in cell or mt > cell[key][1]:
                cell[key] = (v, mt, base)
    return cell


def summarise(cell):
    """(fam, eval, method) -> (id_prr or None, mean OOD prr or None, n_rungs, source file)"""
    byme = defaultdict(dict)
    src = {}
    for (fam, e, m, rg), (v, mt, base) in cell.items():
        byme[(fam, e, m)][rg] = v
        src[(fam, e, m)] = base
    out = {}
    for key, rr in byme.items():
        idv = rr.get("ID")
        ood = [v for rg, v in rr.items() if rg != "ID"]
        out[key] = (idv, (sum(ood) / len(ood)) if ood else None, len(rr), src[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true", help="print only the coverage matrix")
    args = ap.parse_args()
    S = summarise(load())

    methods = sorted({m for (_, _, m) in S}, key=lambda m: (family(m), m))
    for fam, title in [("STANDARD", "STANDARD ProbeDrift(XL) ladder  (mixed short+long training pool)"),
                       ("LONG", "ProbeDriftLong  (long-form-only training pool)")]:
        print(f"\n\n## {title}\n")
        print("`ID | OOD` = PRR at ID and the mean over that eval's OOD rungs. `·` = not run.\n")
        hdr = f"| {'method':34s} | " + " | ".join(f"{e[:9]:>13s}" for e in EVALS) + " |"
        print(hdr)
        print("|" + "-" * 36 + "|" + "|".join(["-" * 15] * len(EVALS)) + "|")
        last_fam = None
        for m in methods:
            fm = family(m)
            if not any((fam, e, m) in S for e in EVALS):
                continue
            if fm != last_fam:
                print(f"| **{fm}** " + "|" * (len(EVALS) + 1))
                last_fam = fm
            cells = []
            for e in EVALS:
                if (fam, e, m) in S:
                    idv, oodv, n, _ = S[(fam, e, m)]
                    a = f"{idv:+.2f}" if idv is not None else "  ·  "
                    b = f"{oodv:+.2f}" if oodv is not None else "  ·  "
                    cells.append(f"{a}|{b}".rjust(13))
                else:
                    cells.append(f"{'·':>13s}")
            print(f"| {m:34s} | " + " | ".join(cells) + " |")

    print("\n\n## Coverage summary (methods present per eval target)\n")
    print(f"| eval | type | STANDARD | ProbeDriftLong |")
    print("|---|---|---|---|")
    for e in EVALS:
        ns = len({m for (f_, e_, m) in S if f_ == "STANDARD" and e_ == e})
        nl = len({m for (f_, e_, m) in S if f_ == "LONG" and e_ == e})
        print(f"| {e} | {'core' if e in CORE else 'XL'} | {ns} | {nl} |")


if __name__ == "__main__":
    main()
