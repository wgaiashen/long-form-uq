#!/usr/bin/env python
"""W5 VERDICT -- does the pre-committed lambda = 1.5 beat the incumbent shrink@2?

Pre-registration: prereg/W5_lambda_and_nll_prior.md.
Reads results/sharpening_lambda_<eval>__<slug>.csv plus the master table.

WHAT IS BEING TESTED
--------------------
lambda = 1.5 was PRE-COMMITTED (prereg §2.1) on the strength of a prediction made on 2026-07-12,
a month before the widened long grid existed: "Sweet spot ~ shrink@1-2", naming the same two rungs
(DiffTask, 1ds-Diff) where wMSP@2 now leads. lambda = 1 and 1.5 had never been run on the long grid.

THE BAR IS shrink@2 (+0.2287), NOT `norm`. Beating the UNREGULARISED wMSP is not the question --
that only re-establishes that some shrinkage helps, which is already known (+0.090, the largest
single lever measured in the project). The question is whether the July-predicted value beats the
incumbent that was chosen by looking at test.

REGISTERED FAILURE READING (prereg §2.4): if lambda = 1.5 misses, that is a failure OF THE
PRE-COMMITTED VALUE, even if some other lambda in the sweep clears the bar. Promoting a different one
afterwards is banned.

THE CROSS-SOURCE JOIN, AND THE CHECK THAT VALIDATES IT
------------------------------------------------------
lambda in {norm, 2, 10} live in `pdl_master`; lambda in {1, 1.5} come from the new runs. Joining two
sources is exactly the bug class this project has been bitten by, so the `norm` arm -- present in
BOTH -- is compared cell by cell first. If it does not match to 4 dp the join is unsound and nothing
below is readable.

    python scripts/checks/sharpening_lambda_verdict.py
"""
import argparse
import csv as _csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as _st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RES = ROOT / "results"
MASTER = RES / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
PRIMARY = "lam1.5"
BAR_MARGIN, BAR_SIGNS, BAR_P = 0.010, 6, 0.05


def load_master():
    g = defaultdict(dict)
    for r in _csv.DictReader(open(MASTER)):
        if r.get("seed_regime") != "3seed":
            continue
        try:
            g[r["method"]][(r["rung"], r["eval"])] = float(r["prr"])
        except (ValueError, TypeError):
            continue
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / f"sharpening_lambda_verdict__{SLUG}.csv"))
    args = ap.parse_args()

    g = load_master()
    mine = defaultdict(dict)          # param -> (rung, eval) -> prr
    tilt = defaultdict(lambda: defaultdict(dict))
    present = set()
    for f in sorted(glob.glob(str(RES / f"sharpening_lambda_*__{SLUG}.csv"))):
        if "SMOKE" in f or "verdict" in f:
            continue
        for r in _csv.DictReader(open(f)):
            key = (r["rung"], r["eval"])
            present.add(r["eval"])
            try:
                v = float(r["prr"])
            except (ValueError, TypeError):
                continue
            if r["kind"] == "lambda":
                mine[r["param"]][key] = v
            elif r["kind"].startswith("tilt_"):
                tilt[r["kind"]][r["param"]][key] = v

    present = [e for e in LONG if e in present]
    missing = [e for e in LONG if e not in present]
    print("=" * 100)
    print("W5 VERDICT -- pre-committed lambda = 1.5 against the incumbent shrink@2")
    print("Population: widened cells_long, Llama-3.1-8B, 4 OOD rungs, 3 seeds, legacy carve.")
    print("=" * 100)
    print(f"\nCOVERAGE: {len(present)}/8 evals.")
    if missing:
        print(f"MISSING, named not silently averaged over: {missing}")
        print("   Everything below is on the PARTIAL population and says so.")
    if not present:
        raise SystemExit("no eval CSVs yet")

    # ---- the join check ----
    bad = [(k, mine["norm"][k], g["wMSP-norm"][k]) for k in mine.get("norm", {})
           if k in g["wMSP-norm"] and abs(mine["norm"][k] - g["wMSP-norm"][k]) > 1e-4]
    print(f"\nJOIN CHECK (`norm` appears in both sources): "
          f"{'ALL MATCH to 4dp' if not bad else f'{len(bad)} MISMATCH -- STOP'}")
    if bad:
        for k, a, b in bad[:5]:
            print(f"   {k}: mine {a:+.4f} vs master {b:+.4f}")
        raise SystemExit("cross-source join is unsound; nothing below is readable")

    def dmean(src, key):
        """per-dataset OOD mean; None if any rung missing"""
        out = {}
        for d in present:
            vals = [src.get((rg, d)) for rg in OOD]
            if any(v is None for v in vals):
                return None
            out[d] = float(np.mean(vals))
        return out

    arms = {"norm": mine.get("norm", {}), "lam1": mine.get("lam1", {}),
            "lam1.5": mine.get("lam1.5", {}),
            "shrink@2": g["wMSP-shrink@2"], "shrink@10": g["wMSP-shrink@10"]}
    per = {k: dmean(v, k) for k, v in arms.items()}
    per = {k: v for k, v in per.items() if v is not None}
    fl = dmean(g["msp_min"], "msp_min")
    sap = dmean(g["SAPLMA"], "SAPLMA")

    print(f"\nPER-DATASET OOD MEANS ({len(present)}/8 evals)")
    print(f"{'eval':15s}" + "".join(f"{k:>11s}" for k in ["norm", "lam1", "lam1.5", "shrink@2",
                                                          "shrink@10"] if k in per)
          + f"{'msp_min':>10s}{'SAPLMA':>9s}")
    for d in present:
        print(f"{d:15s}" + "".join(f"{per[k][d]:>+11.4f}" for k in
                                   ["norm", "lam1", "lam1.5", "shrink@2", "shrink@10"] if k in per)
              + f"{fl[d]:>+10.4f}{sap[d]:>+9.4f}")
    print(f"{'MEAN':15s}" + "".join(f"{np.mean(list(per[k].values())):>+11.4f}" for k in
                                    ["norm", "lam1", "lam1.5", "shrink@2", "shrink@10"] if k in per)
          + f"{np.mean(list(fl.values())):>+10.4f}{np.mean(list(sap.values())):>+9.4f}")

    # ---- the registered comparison ----
    rows = []
    if PRIMARY in per and "shrink@2" in per:
        d = np.array([per[PRIMARY][e] - per["shrink@2"][e] for e in present])
        p = _st.wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
        signs = int((d > 0).sum())
        ok = (d.mean() > BAR_MARGIN) and (signs >= BAR_SIGNS) and (p < BAR_P)
        print("\n" + "=" * 100)
        print(f"REGISTERED COMPARISON: lambda = 1.5 vs shrink@2  (n = {len(present)} datasets)")
        print("=" * 100)
        print(f"  margin  {d.mean():+.4f}   bar > +{BAR_MARGIN:.3f}   "
              f"{'PASS' if d.mean() > BAR_MARGIN else 'FAIL'}")
        print(f"  signs   {signs}/{len(present)}       bar >= {BAR_SIGNS}/8        "
              f"{'PASS' if signs >= BAR_SIGNS else 'FAIL'}")
        print(f"  Wilcoxon p {p:.4f}   bar < {BAR_P}       {'PASS' if p < BAR_P else 'FAIL'}")
        print(f"\n  VERDICT: {'YES' if ok else 'NO'}"
              + ("" if len(present) == 8 else "   (PARTIAL -- not final until 8/8)"))
        print("\n  Registered reading: if lambda=1.5 misses, that is a failure OF THE PRE-COMMITTED")
        print("     VALUE even if another lambda clears the bar. Promoting a different one is banned.")
        rows.append(("registered", PRIMARY, "shrink@2", f"{d.mean():.4f}", signs, f"{p:.4f}",
                     len(present)))

        # secondary: also against msp_min, and the per-rung breakdown July predicted
        d2 = np.array([per[PRIMARY][e] - fl[e] for e in present])
        print(f"\n  secondary, vs msp_min: {d2.mean():+.4f}, {int((d2>0).sum())}/{len(present)}")

    # ---- LODO over the five lambdas ----
    cands = [k for k in ["norm", "lam1", "lam1.5", "shrink@2", "shrink@10"] if k in per]
    if len(cands) > 1 and len(present) > 1:
        sel, picks = [], {}
        for e in present:
            others = [o for o in present if o != e]
            b = max(cands, key=lambda k: float(np.mean([per[k][o] for o in others])))
            picks[e] = b
            sel.append(per[b][e])
        print(f"\n  LODO over {cands}: {np.mean(sel):+.4f}")
        print(f"    picks: " + "  ".join(f"{e}->{picks[e]}" for e in present))
        rows.append(("lodo", "|".join(cands), "", f"{np.mean(sel):.4f}", "", "", len(present)))

    # ---- the tilt arms (exploratory, no pre-committed parameter) ----
    for kind in ("tilt_exp", "tilt_pow"):
        if kind not in tilt:
            continue
        print(f"\n  {kind} (EXPLORATORY, per-cell best is an ORACLE):")
        params = sorted(tilt[kind], key=lambda x: (x == "inf", float(x) if x != "inf" else 1e9))
        for pm in params:
            pv = dmean(tilt[kind][pm], pm)
            if pv:
                print(f"     {pm:>6s}: {np.mean(list(pv.values())):+.4f}")

    outp = Path(args.out)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["kind", "arm", "vs", "margin", "signs", "p", "n_evals"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}")
    if missing:
        print(f"PARTIAL: {len(present)}/8. Missing {missing}. Re-run when they land.")


if __name__ == "__main__":
    main()
