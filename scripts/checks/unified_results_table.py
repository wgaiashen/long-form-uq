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
        # MODEL PIN (2026-07-27, blast-radius guard): Llama-3.1-8B is the ONLY model. Our result CSVs are
        # named `..__meta-llama_Meta-Llama-3.1-8B.csv`. Skip anything else so a stray Qwen/Gemma CSV can never
        # enter the cross-dataset table by construction (not only by having archived them out of results/).
        if "Meta-Llama-3.1-8B" not in Path(f).name:
            print(f"  SKIP non-Llama CSV: {Path(f).name}", flush=True)
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
        # exclude ID and the STRAY BLANK rung (an ID-valued row from ood_onegrid that wrongly inflated the
        # OOD mean -- e.g. sciq attention 0.55 -> 0.626). Keep every real rung (incl. the LONG -long rungs).
        ood = [v for rg, v in rr.items() if rg not in ("ID", "")]
        out[key] = (idv, (sum(ood) / len(ood)) if ood else None, len(rr), src[key])
    return out


# the FOUR canonical OOD rungs. summarise() averages *any* non-ID rung, which wrongly includes a stray
# blank-rung row (an ID-valued row from ood_onegrid) -> inflates the pooler OOD-mean. §B.3 must average ONLY
# these four (this reproduces the verified 0.300/0.274 pooler means; summarise's laxer set gave 0.326/0.300).
OOD_RUNGS = ("SameTask", "LOO", "OneDatasetDiffTask", "DiffTask")


def b3_report(cell):
    """V6 / §B.3 RIGOUR: replace the flattering '+0.016 mean beats all 3 floors OOD' with the honest reading
    -- per-DATASET win/loss vs the pre-registered bar AND the per-dataset dual-report bar, plus a paired
    bootstrap CI (over the 9 datasets) on the cross-dataset mean difference. A cross-dataset MEAN can be
    carried by two datasets; the per-dataset count and the CI say whether it is real."""
    import numpy as np
    METHODS = ["Attention pooler", "Uniform pooler (mean-pool)", "wMSP-normalised"]
    FLOORS = ["MSP-sum", "Perplexity", "MSP-min"]
    BAR = "MSP-min"                                   # the pre-registered bar
    N_BOOT = 5000
    print("\n## §B.3 — pooler vs the unsupervised floor, OOD (STANDARD ladder, per-dataset + paired bootstrap)\n")
    print("Per-dataset OOD-mean PRR = mean over the 4 OOD rungs. `dual bar` = the STRONGEST free floor per "
          "dataset (max of sum/ppl/min). CI = 95% paired bootstrap over the 9 datasets "
          f"({N_BOOT} resamples, seed 1) on the cross-dataset mean difference.\n")

    def ood(fam, e, m):
        """OOD-mean over ONLY the 4 canonical rungs (excludes the stray blank rung)."""
        vals = [cell[(fam, e, m, rg)][0] for rg in OOD_RUNGS if (fam, e, m, rg) in cell]
        return sum(vals) / len(vals) if vals else None
    evs = [e for e in EVALS if ood("STANDARD", e, BAR) is not None]
    floor_by_e = {e: {fl: ood("STANDARD", e, fl) for fl in FLOORS} for e in evs}
    dual = {e: max(v for v in floor_by_e[e].values() if v is not None) for e in evs}
    minbar = {e: floor_by_e[e][BAR] for e in evs}
    for meth in METHODS:
        me = {e: ood("STANDARD", e, meth) for e in evs if ood("STANDARD", e, meth) is not None}
        ev = [e for e in evs if e in me]
        if not ev:
            continue
        d_min = np.array([me[e] - minbar[e] for e in ev])
        d_dual = np.array([me[e] - dual[e] for e in ev])
        rs = np.random.RandomState(1)
        def boot(d):
            bs = [d[rs.randint(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        lo_m, hi_m = boot(d_min); lo_d, hi_d = boot(d_dual)
        winmin = [e for e in ev if me[e] > minbar[e]]
        windual = [e for e in ev if me[e] > dual[e]]
        print(f"### {meth}  (n={len(ev)} evals)")
        print(f"  cross-dataset mean OOD PRR: {np.mean([me[e] for e in ev]):+.3f}  "
              f"(vs {BAR} {np.mean([minbar[e] for e in ev]):+.3f}, vs dual {np.mean([dual[e] for e in ev]):+.3f})")
        print(f"  vs {BAR} (pre-registered): **{len(winmin)}/{len(ev)}** wins; "
              f"mean Δ {d_min.mean():+.3f}, 95% CI [{lo_m:+.3f},{hi_m:+.3f}] "
              f"{'(excludes 0)' if lo_m>0 else '(includes 0 -> NOT distinguishable from the floor)'}")
        print(f"      wins on: {', '.join(winmin) or '(none)'}")
        print(f"  vs DUAL bar (strongest free floor): **{len(windual)}/{len(ev)}** wins; "
              f"mean Δ {d_dual.mean():+.3f}, 95% CI [{lo_d:+.3f},{hi_d:+.3f}] "
              f"{'(excludes 0)' if lo_d>0 else '(includes 0)'}")
        print(f"      wins on: {', '.join(windual) or '(none)'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true", help="print only the coverage matrix")
    ap.add_argument("--b3", action="store_true", help="V6/§B.3 pooler-vs-floor per-dataset + paired bootstrap")
    args = ap.parse_args()
    C = load()
    if args.b3:
        b3_report(C)
        return
    S = summarise(C)

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
