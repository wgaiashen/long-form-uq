#!/usr/bin/env python
"""W3 -- THE HONEST-SELECTION AUDIT: re-select the weighted-MSP lambda WITHOUT looking at test.

Plan: ../PLAN_sharpening_axis.md §6.   Results doc: ../STOCKTAKE_sharpening_axis.md §5.

WHY THIS EXISTS
---------------
Joe, 7 August (reference/MEETING_NOTES_7Aug.md:102-120):
    "Have we done that on some dev set, or have we done all the test set and picked the best one?
     ... that's probably not very science."
and at :111 specifically: "The shrink factor (2 vs 10 vs normalised) is currently chosen on test."

All three weighted-MSP arms were run on every test cell and `shrink@2` was promoted to the headline
because it scored highest. That is a maximum over three noisy draws and is optimistically biased.
This script re-selects it by LEAVE-ONE-DATASET-OUT and reports whatever that procedure yields --
INCLUDING if it is worse than the +0.2287 currently quoted.

⚠️ ONE CORRECTION TO THE RECEIVED STORY, verified in the sources rather than assumed. The VALUES
{2, 10} did not come from the long-form test grid at all. They came from a coarse July grid on three
short/QA datasets (reports/OVERNIGHT_10jul.md:44, "First reasonable grid"), and a finer sweep
(reports/OVERNIGHT_12jul.md:23-25) concluded "Sweet spot ~ shrink@1-2" and was NEVER PROPAGATED to
the long grid. So lambda = 1 and 1.5 have never been tried on long-form. That needs a run and is NOT
what this script does; this script re-selects honestly among the three arms that were actually
measured on the full grid.

⚠️ UNIT OF ANALYSIS. Weighted MSP is TRAINED, so unlike the free floors it is NOT rung-invariant and
its cells are not pseudo-replicated. Selection is still done per DATASET (n = 8), because that is
the level a deployment choice is made at, and the held-out unit has to be a whole dataset for the
procedure to mean anything.

Reads the assembled master table only. No training, no GPU, no new cells.

    python scripts/checks/selection_audit.py
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MASTER = ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
OUT = ROOT / "results" / "sharpening_selection_audit__meta-llama_Meta-Llama-3.1-8B.csv"

LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
ARMS = ["wMSP-norm", "wMSP-shrink@2", "wMSP-shrink@10"]     # the three actually on the full grid
REFS = ["msp_min", "SAPLMA"]                                 # the two bars we report against


def load_master():
    """method -> rung -> eval -> prr, restricted to the 3-seed block so nothing crosses populations."""
    if not MASTER.exists():
        raise SystemExit(f"master table not found: {MASTER}")
    g = defaultdict(dict)
    with open(MASTER) as fh:
        for r in _csv.DictReader(fh):
            if r.get("seed_regime") != "3seed":
                continue                                     # seed-1 post-hoc rows are a DIFFERENT regime
            try:
                g[r["method"]][(r["rung"], r["eval"])] = float(r["prr"])
            except (ValueError, TypeError):
                continue                                     # blank = NOT MEASURED, never zero-filled
    return g


def ood_mean_by_dataset(grid, method):
    """One number per dataset: the mean over the 4 OOD rungs. Returns None if any cell is missing --
    a partial mean would silently compare different cell sets across methods."""
    out = {}
    for d in LONG:
        vals = [grid[method].get((rg, d)) for rg in OOD_RUNGS]
        if any(v is None for v in vals):
            return None, [f"{d}/{rg}" for rg, v in zip(OOD_RUNGS, vals) if v is None]
        out[d] = float(np.mean(vals))
    return out, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    grid = load_master()
    per = {}
    for m in ARMS + REFS:
        vals, missing = ood_mean_by_dataset(grid, m)
        if vals is None:
            raise SystemExit(f"COVERAGE FAIL [{m}]: missing cells {missing}. A partial grid is not "
                             f"reported as a full one -- fill the cells or drop the method loudly.")
        per[m] = vals

    print("=" * 100)
    print("W3 -- HONEST RE-SELECTION OF THE WEIGHTED-MSP LAMBDA")
    print("Population: widened cells_long, meta-llama/Llama-3.1-8B, 4 OOD rungs, 3-seed, n = 8 DATASETS.")
    print(f"Source: {MASTER.name}")
    print("=" * 100)

    print(f"\n{'dataset':16s}" + "".join(f"{a:>16s}" for a in ARMS)
          + f"{'msp_min':>10s}{'SAPLMA':>10s}")
    for d in LONG:
        print(f"{d:16s}" + "".join(f"{per[a][d]:>+16.4f}" for a in ARMS)
              + f"{per['msp_min'][d]:>+10.4f}{per['SAPLMA'][d]:>+10.4f}")
    print(f"{'MEAN (n=8)':16s}" + "".join(f"{np.mean(list(per[a].values())):>+16.4f}" for a in ARMS)
          + f"{np.mean(list(per['msp_min'].values())):>+10.4f}"
          + f"{np.mean(list(per['SAPLMA'].values())):>+10.4f}")

    # ---- what was actually done: pick the arm with the best mean, on test ----
    test_pick = max(ARMS, key=lambda a: np.mean(list(per[a].values())))
    test_score = float(np.mean(list(per[test_pick].values())))

    # ---- the honest version: leave one DATASET out, select on the other 7, apply to the held-out ----
    lodo_vals, lodo_pick = [], {}
    for d in LONG:
        others = [o for o in LONG if o != d]
        pick = max(ARMS, key=lambda a: np.mean([per[a][o] for o in others]))
        lodo_pick[d] = pick
        lodo_vals.append(per[pick][d])
    lodo_score = float(np.mean(lodo_vals))

    print("\n" + "=" * 100)
    print("OLD (test-selected) BESIDE NEW (leave-one-dataset-out). Both reported, whatever the sign.")
    print("=" * 100)
    print(f"  test-selected      arm = {test_pick:16s} OOD mean = {test_score:+.4f}   <-- how the "
          f"headline was chosen")
    print(f"  LODO-selected      per-fold          OOD mean = {lodo_score:+.4f}")
    print(f"  optimism of the test-selected number:          {test_score - lodo_score:+.4f}")
    print(f"\n  per fold: " + "  ".join(f"{d}->{lodo_pick[d]}" for d in LONG))
    n_agree = sum(1 for d in LONG if lodo_pick[d] == test_pick)
    print(f"  folds agreeing with the test pick: {n_agree}/8")
    if n_agree == 8:
        print("  ⚠️ All 8 folds chose the same arm. The selection is STABLE, so the test-selected")
        print("     number is optimistic by little here -- but the PROCEDURE was still not honest,")
        print("     and the stability is the reason to say so rather than a defence of the method.")

    print(f"\n  bars: msp_min {np.mean(list(per['msp_min'].values())):+.4f}   "
          f"SAPLMA {np.mean(list(per['SAPLMA'].values())):+.4f}")

    rows = [("wmsp_lambda", "test-selected", test_pick, test_score),
            ("wmsp_lambda", "lodo-selected", "per-fold", lodo_score)]
    for d in LONG:
        rows.append(("wmsp_lambda_fold", d, lodo_pick[d], per[lodo_pick[d]][d]))

    # ---- leverage check on the LODO result: does the win survive dropping any one dataset? ----
    print("\n" + "=" * 100)
    print("LEVERAGE CHECK on the re-selection (the check to run when the answer is convenient)")
    print("=" * 100)
    flips = []
    for drop in LONG:
        o = [d for d in LONG if d != drop]
        best = max(ARMS, key=lambda a: np.mean([per[a][d] for d in o]))
        if best != test_pick:
            flips.append(drop)
    print(f"  arm chosen on the other 7 flips away from {test_pick} when dropping: "
          f"{flips if flips else 'NONE -- the selection is stable to any single dataset'}")
    wins = {a: sum(1 for d in LONG if max(ARMS, key=lambda z: per[z][d]) == a) for a in ARMS}
    print(f"  per-dataset winners (the MEAN hides this): " + "  ".join(f"{a}={wins[a]}/8" for a in ARMS))
    print("  ⚠️ So the headline arm wins the MEAN but is not the per-dataset winner everywhere. The")
    print("     honest statement is 'the procedure picks the same arm', not 'the arm is best everywhere'.")

    # ---- is the shrink level itself a SHARPENING parameter? Tested, and reported as a NULL. ----
    # Heavier shrinkage pulls the learned weights toward uniform, i.e. toward `perplexity`. If shrink
    # were the same axis the W1 family sweeps, then the datasets that prefer more shrinkage should be
    # the ones where `perplexity` beats `msp_min`. This is a real prediction, so it is tested here.
    x = np.array([per["msp_min"][d] for d in LONG])
    ppl_vals, _ = ood_mean_by_dataset(grid, "perplexity")
    xg = np.array([ppl_vals[d] - per["msp_min"][d] for d in LONG])      # >0 = the SPREAD regime
    yg = np.array([per["wMSP-shrink@10"][d] - per["wMSP-shrink@2"][d] for d in LONG])  # >0 = wants more shrink
    from scipy import stats as _st
    pear = float(np.corrcoef(xg, yg)[0, 1])
    sp = _st.spearmanr(xg, yg)
    keep = [i for i, d in enumerate(LONG) if d not in ("pubmed_qa", "cnn_dailymail")]
    pear6 = float(np.corrcoef(xg[keep], yg[keep])[0, 1])
    sp6 = _st.spearmanr(xg[keep], yg[keep])
    signs = int(np.sum(np.sign(xg) == np.sign(yg)))
    print("\n" + "=" * 100)
    print("IS THE SHRINK LEVEL THE SAME SHARPENING AXIS? -- tested, and it is a NULL")
    print("=" * 100)
    print("  Prediction: heavier shrinkage pulls the learned weights toward uniform, i.e. toward")
    print("  perplexity, so datasets preferring shrink@10 over shrink@2 should be the SPREAD ones.")
    print(f"  all 8:            Pearson {pear:+.3f}   Spearman {sp.statistic:+.3f}  p={sp.pvalue:.3f}")
    print(f"  drop pubmed+cnn:  Pearson {pear6:+.3f}   Spearman {sp6.statistic:+.3f}  p={sp6.pvalue:.3f}")
    print(f"  sign agreement:   {signs}/8")
    print("  ⚠️ VERDICT: NOT ESTABLISHED. The Pearson of +0.86 on all 8 is carried entirely by the two")
    print("     datasets with the largest endpoint margins (pubmed_qa −0.545, cnn_dailymail +0.290);")
    print("     removing both collapses it to +0.32 / −0.03. Spearman never reaches significance at")
    print("     n = 8. samsum is a clear counterexample. Reported as suggestive-at-best, and the")
    print("     W1 family sweep does NOT get to cite this as supporting evidence.")
    rows += [("shrink_is_sharpening", "pearson_n8", f"{pear:+.3f}", ""),
             ("shrink_is_sharpening", "spearman_n8", f"{sp.statistic:+.3f}", f"p={sp.pvalue:.3f}"),
             ("shrink_is_sharpening", "pearson_drop2", f"{pear6:+.3f}", "pubmed+cnn removed"),
             ("shrink_is_sharpening", "verdict", "NOT ESTABLISHED", "leverage-driven")]

    # ---- the audit list itself: every choice made by reading test results ----
    print("\n" + "=" * 100)
    print("THE AUDIT LIST -- configuration choices made by looking at TEST results")
    print("(file:line evidence in ../STOCKTAKE_sharpening_axis.md §5; this is the summary)")
    print("=" * 100)
    AUDIT = [
        ("weighted-MSP lambda {norm,2,10}", "all run on every test cell, best promoted",
         "RE-SELECTED here by LODO"),
        ("best-of-eight wMSP verdict rows", "argmax over 8 arms on test PRR per cell "
         "(probedriftlong.py:513-514)", "asymmetry: max-of-THREE was rejected for the BASELINE"),
        ("prior-tilt beta = 0.5", "max over 6 values on test cells",
         "already self-labelled an oracle (STOCKTAKE_post31July.md:1153)"),
        ("armD:top-25% prior arm", "best win-count against SAPLMA",
         "already flagged post-hoc; the registered single-k primary FAILED"),
        ("Orgad threshold tau = 0.3", "soft tier beat the hard mask on OOD test cells",
         "line ended negative anyway, so it props up no positive claim"),
        ("Orgad locator prompt (narrow -> broad)", "switched after the narrow prompt scored badly",
         "prompt-variant selection on test"),
        ("P(True) wording", "a-priori task-agnostic rationale, CONFIRMED on ID test PRR",
         "mixed; and on the dropped Gemma population"),
        ("fair_floor = max of three", "argmax on test, per cell",
         "inflates the BASELINE, so it biases AGAINST our own claims; superseded by msp_min"),
    ]
    print(f"{'choice':40s}{'how it was picked':52s}{'status'}")
    for a, b, c in AUDIT:
        print(f"{a:40s}{b:52s}{c}")
        rows.append(("audit", a, b, c))

    print("\n" + "=" * 100)
    print("CLEAN BY CONSTRUCTION (recorded so the audit is not read as 'everything was tuned')")
    print("=" * 100)
    for a, b in [
        ("probe layer 15", "a priori middle-layer heuristic, explicitly NOT the best-scoring layer"),
        ("attention-pooler temperature", "selected on a 20% VALIDATION slice carved from TRAIN"),
        ("auxiliary-loss lambda", "selected by holding out a whole SOURCE dataset"),
        ("entropy-penalty tau", "selected on a validation slice carved from train"),
        ("SAPLMA / wMSP optimiser settings", "fixed a priori from Joe's recipe, never swept"),
        ("decoding repetition_penalty", "chosen on degeneracy/relevance BEFORE labelling, not on PRR"),
    ]:
        print(f"  {a:38s}{b}")

    print("\n⚠️ NOT COVERED BY THIS RUN, and it needs one: lambda = 1 and 1.5 have NEVER been tried on")
    print("   the long grid, despite the 12 July sweep concluding 'sweet spot ~ shrink@1-2'. The")
    print("   re-selection above can only choose among the three arms that were actually measured.")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["kind", "key", "value", "detail"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
