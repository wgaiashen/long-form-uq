#!/usr/bin/env python
"""F5b VERDICT -- does anchoring the wMSP penalty at msp_min work, and is it the ANCHOR that works?

Prereg: prereg/F5_anchor_at_msp_min.md.  Results: the project results log
Reads results/anchor_msp_min_<eval>__logpen__<slug>.csv (the log-penalty runs; the linear runs are
the §16.1 stalled-optimisation record and are not read here).

WHAT IS REPORTED, IN ORDER
--------------------------
1. JOIN CHECK: each eval's lambda=0 arm vs pdl_master's wMSP-norm, cell by cell.
2. CONTROL D per eval: max mean p[k] reached. An eval where the penalty never bites (p[k] stays
   near 1/n) CANNOT test the anchoring idea -- it is reported as UNTESTED there, never as a null.
3. The OOD picture per eval: best-lambda anchor arm vs lambda=0, vs the RANDOM-anchor control at the
   same lambda, vs msp_min / shrink@2 / SAPLMA.
4. THE MECHANISM TEST: per-dataset anchor gain vs that dataset's msp_min quality (the §15.2 logic,
   now for the new anchor). With leave-one-out on the correlation, as always.
5. HONEST-SELECTION ARM: lambda chosen per dataset by LODO over the other 7; reported as what the
   procedure achieves. Per-cell best is an ORACLE and is never a result.

    python scripts/checks/anchor_verdict.py
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
SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
import os as _os
SUFFIX = _os.environ.get("F5_SUFFIX", "logpen")     # logpen = F5b; logws = F5c (warm-start+combo)
BITE_MIN = 0.20        # if max p[k] over the grid never reaches this, the eval is UNTESTED, not null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / f"anchor_verdict__{SLUG}.csv"))
    args = ap.parse_args()

    g = defaultdict(dict)
    for r in _csv.DictReader(open(RES / f"pdl_master__{SLUG}.csv")):
        if r.get("seed_regime") != "3seed":
            continue
        try:
            g[r["method"]][(r["rung"], r["eval"])] = float(r["prr"])
        except (ValueError, TypeError):
            continue

    anc = defaultdict(dict)   # (rung, eval, lambda) -> prr ; and pk / random alongside
    rnd = defaultdict(dict)
    cmb = defaultdict(dict)
    wso = defaultdict(dict)
    pk = defaultdict(dict)
    present = set()
    import argparse as _a
    for f in sorted(glob.glob(str(RES / f"anchor_msp_min_*__{SUFFIX}__{SLUG}.csv"))):
        if "SMOKE" in f:
            continue
        for r in _csv.DictReader(open(f)):
            if r["mode"] == "floor":
                continue
            present.add(r["eval"])
            key = (r["rung"], r["eval"], float(r["lambda"]))
            try:
                v = float(r["prr"])
            except (ValueError, TypeError):
                continue
            if r["mode"] == "anchor":
                anc[key] = v
                try:
                    pk[key] = float(r["mean_p_anchor"])
                except (ValueError, TypeError):
                    pass
            elif r["mode"] == "random":
                rnd[key] = v
            elif r["mode"] == "combo":
                cmb[key] = v
            elif r["mode"] == "wsonly":
                wso[key] = v
    present = [e for e in LONG if e in present]
    missing = [e for e in LONG if e not in present]

    print("=" * 100)
    print(f"F5 VERDICT [{SUFFIX}] -- anchoring weighted MSP at msp_min")
    print("Population: widened cells_long, Llama-3.1-8B, 3 seeds, legacy carve.")
    print("=" * 100)
    print(f"\nCOVERAGE: {len(present)}/8 evals." + (f"  MISSING, named: {missing}" if missing else ""))
    if not present:
        raise SystemExit("no logpen CSVs yet")

    # 1. join check
    bad = 0
    for e in present:
        for rg in ["ID"] + OOD:
            mine = anc.get((rg, e, 0.0))
            ref = g["wMSP-norm"].get((rg, e))
            if mine is not None and ref is not None and abs(mine - ref) > 1e-4:
                bad += 1
                print(f"  MISMATCH {e}/{rg}: {mine:+.4f} vs master {ref:+.4f}")
    print(f"JOIN CHECK (lambda=0 vs pdl_master wMSP-norm): "
          f"{'ALL MATCH to 4dp' if bad == 0 else f'{bad} MISMATCH -- STOP'}")
    if bad:
        raise SystemExit("join unsound")

    # 2. does the penalty bite, per eval (OOD cells only)
    print(f"\n{'eval':15s}{'max p[k] (OOD)':>15s}{'tested?':>9s}")
    tested = {}
    for e in present:
        mx = max([pk.get((rg, e, l), 0.0) for rg in OOD for l in LAMBDAS] or [0.0])
        tested[e] = mx >= BITE_MIN
        print(f"{e:15s}{mx:>15.3f}{('yes' if tested[e] else 'NO -> UNTESTED'):>9s}")

    # 3. per-eval OOD summary at the best lambda (ORACLE, labelled) + the same lambda's random arm
    def ood_mean(store, e, l):
        vals = [store.get((rg, e, l)) for rg in OOD]
        return None if any(v is None for v in vals) else float(np.mean(vals))

    print("\n" + "-" * 100)
    print("PER-EVAL OOD MEANS.  best-lambda is an ORACLE (labelled); 'random@same' is Control C.")
    print("-" * 100)
    print(f"{'eval':15s}{'l=0':>8s}{'best-l':>8s}{'(l)':>6s}{'rand@same':>10s}{'anc-rand':>9s}"
          f"{'msp_min':>9s}{'shr@2':>8s}{'SAPLMA':>8s}")
    rows = []
    gains, anchors_q = [], []
    for e in present:
        base = ood_mean(anc, e, 0.0)
        cand = [(l, ood_mean(anc, e, l)) for l in LAMBDAS[1:]]
        cand = [(l, v) for l, v in cand if v is not None]
        if base is None or not cand:
            continue
        bl, bv = max(cand, key=lambda t: t[1])
        rv = ood_mean(rnd, e, bl)
        mn = float(np.mean([g["msp_min"][(rg, e)] for rg in OOD]))
        s2 = float(np.mean([g["wMSP-shrink@2"][(rg, e)] for rg in OOD]))
        sap = float(np.mean([g["SAPLMA"][(rg, e)] for rg in OOD]))
        print(f"{e:15s}{base:>+8.3f}{bv:>+8.3f}{bl:>6g}{(rv if rv is not None else float('nan')):>+10.3f}"
              f"{(bv - rv if rv is not None else float('nan')):>+9.3f}{mn:>+9.3f}{s2:>+8.3f}{sap:>+8.3f}")
        rows.append((e, base, bv, bl, rv, mn, s2, sap, tested[e]))
        if tested[e]:
            gains.append(bv - base)
            anchors_q.append(mn)

    # 4. the mechanism test on the TESTED evals only
    if len(gains) >= 4:
        sp = _st.spearmanr(anchors_q, gains)
        print(f"\nMECHANISM (tested evals only, n={len(gains)}): "
              f"Spearman(anchor quality, oracle gain) = {sp.statistic:+.3f}  p={sp.pvalue:.3f}")
        for i in range(len(gains)):
            k = [j for j in range(len(gains)) if j != i]
            print(f"   LOO drop #{i}: {_st.spearmanr([anchors_q[j] for j in k], [gains[j] for j in k]).statistic:+.3f}")

    # 5. honest-selection arm: one lambda per held-out eval, chosen on the others (tested evals only)
    T = [e for e in present if tested[e]]
    if len(T) > 2:
        sel = {}
        for e in T:
            others = [o for o in T if o != e]
            l = max(LAMBDAS[1:], key=lambda L: np.mean([ood_mean(anc, o, L) or -9 for o in others]))
            sel[e] = (l, ood_mean(anc, e, l))
        base_m = np.mean([ood_mean(anc, e, 0.0) for e in T])
        sel_m = np.mean([v for _, v in sel.values()])
        s2_m = np.mean([np.mean([g["wMSP-shrink@2"][(rg, e)] for rg in OOD]) for e in T])
        print(f"\nHONEST LODO ARM (tested evals, n={len(T)}): {sel_m:+.4f}   "
              f"vs lambda=0 {base_m:+.4f}   vs shrink@2 {s2_m:+.4f}")
        print("   picks: " + "  ".join(f"{e}->{l}" for e, (l, _) in sel.items()))

    # ---- F5c arms: combo (shrink@2-const + anchor) and wsonly (warm-start only) ----
    if cmb:
        print("\n" + "-" * 100)
        print("COMBO ARM (fixed 2.0 uniform + lambda_a anchor) vs shrink@1.5, OOD means; and wsonly")
        print("-" * 100)
        print(f"{'eval':15s}{'l=0':>8s}{'combo-best':>11s}{'(l)':>6s}{'shr@1.5':>9s}{'wsonly@1':>10s}{'wsonly@10':>10s}")
        l15 = defaultdict(dict)
        import glob as _g
        for f2 in _g.glob(str(RES / f"sharpening_lambda_*__{SLUG}.csv")):
            if 'SMOKE' in f2 or 'verdict' in f2:
                continue
            for r2 in _csv.DictReader(open(f2)):
                if r2['kind'] == 'lambda' and r2['param'] == 'lam1.5':
                    l15[r2['eval']][r2['rung']] = float(r2['prr'])
        combo_m, s15_m = [], []
        for e in present:
            base = ood_mean(anc, e, 0.0)
            cc = [(l, ood_mean(cmb, e, l)) for l in LAMBDAS[1:]]
            cc = [(l, v) for l, v in cc if v is not None]
            if not cc:
                continue
            bl, bv = max(cc, key=lambda t: t[1])
            s15 = float(np.mean([l15[e][rg] for rg in OOD])) if e in l15 else float('nan')
            w1 = ood_mean(wso, e, 1.0); w10 = ood_mean(wso, e, 10.0)
            combo_m.append(bv); s15_m.append(s15)
            print(f"{e:15s}{(base if base is not None else float('nan')):>+8.3f}{bv:>+11.3f}{bl:>6g}"
                  f"{s15:>+9.3f}{(w1 if w1 is not None else float('nan')):>+10.3f}"
                  f"{(w10 if w10 is not None else float('nan')):>+10.3f}")
        if combo_m:
            print(f"{'MEAN':15s}{'':8s}{np.mean(combo_m):>+11.4f}{'':6s}{np.nanmean(s15_m):>+9.4f}")
            print("  combo-best is a per-eval ORACLE over lambda_a; the honest read is LODO, and")
            print("     wsonly ~= combo would mean the gain is INITIALISATION, not the anchor.")
        # honest LODO for the combo, all evals
        cands = LAMBDAS[1:]
        sel = []
        pick = {}
        for e in present:
            others = [o for o in present if o != e]
            vals = {l: np.mean([ood_mean(cmb, o, l) or -9 for o in others]) for l in cands}
            b = max(vals, key=vals.get)
            v = ood_mean(cmb, e, b)
            if v is not None:
                sel.append(v); pick[e] = b
        if sel:
            print(f"  COMBO LODO ({len(sel)} evals): {np.mean(sel):+.4f}   picks: " +
                  " ".join(f"{e}->{pick[e]:g}" for e in pick))
    outp = Path(args.out)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["eval", "l0", "best", "best_lambda", "random_at_best", "msp_min", "shrink2",
                    "saplma", "penalty_bites"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}")
    if missing:
        print(f"PARTIAL: {len(present)}/8. Missing {missing}.")


if __name__ == "__main__":
    main()
