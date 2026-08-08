#!/usr/bin/env python
"""MASK ABLATION VERDICT -- does excluding end-of-text from the weight logits help wMSP?

Results: ../STOCKTAKE_sharpening_axis.md §14. Reads results/mask_ablation_<eval>__<slug>.csv.

THE QUESTION. The mask was introduced because the learned weighter CONCENTRATED on EOS, never
because concentrating there was MEASURED to hurt -- the same inference pattern that turned out wrong
about punctuation. It also makes weighted MSP the only method on the ladder scored on a different
token set from the floor it is compared against (§12).

⚠️ OUTSIDE CHECK FIRST. The MASKED arms must reproduce pdl_master's wMSP-norm / shrink@2 / shrink@10
to 4 dp. Without that, a masked-vs-unmasked difference could be this driver rather than the mask.

Unit of analysis: the DATASET (n up to 8), OOD mean over the 4 OOD rungs. Trained methods, so cells
are not pseudo-replicated, but per-dataset is the level a deployment choice is made at.

    python scripts/checks/mask_ablation_verdict.py
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
VARIANTS = ["norm", "shrink2", "shrink10"]
MASTER_NAME = {"norm": "wMSP-norm", "shrink2": "wMSP-shrink@2", "shrink10": "wMSP-shrink@10"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / f"mask_ablation_verdict__{SLUG}.csv"))
    args = ap.parse_args()

    g = defaultdict(dict)
    for r in _csv.DictReader(open(RES / f"pdl_master__{SLUG}.csv")):
        if r.get("seed_regime") != "3seed":
            continue
        try:
            g[r["method"]][(r["rung"], r["eval"])] = float(r["prr"])
        except (ValueError, TypeError):
            continue

    got = defaultdict(dict)      # (variant, mode) -> (rung, eval) -> prr
    present = set()
    for f in sorted(glob.glob(str(RES / f"mask_ablation_*__{SLUG}.csv"))):
        if "SMOKE" in f or "verdict" in f:
            continue
        for r in _csv.DictReader(open(f)):
            if r["variant"] == "floor":
                continue
            present.add(r["eval"])
            try:
                got[(r["variant"], r["mode"])][(r["rung"], r["eval"])] = float(r["prr"])
            except (ValueError, TypeError):
                continue
    present = [e for e in LONG if e in present]
    missing = [e for e in LONG if e not in present]

    print("=" * 100)
    print("MASK ABLATION -- does excluding EOS from the weight logits help?")
    print("Population: widened cells_long, Llama-3.1-8B, 4 OOD rungs, 3 seeds.")
    print("=" * 100)
    print(f"\nCOVERAGE: {len(present)}/8 evals.")
    if missing:
        print(f"⚠️ MISSING, named not silently averaged over: {missing}")
    if not present:
        raise SystemExit("no mask-ablation CSVs yet")

    # ---- outside check ----
    bad = 0
    for v in VARIANTS:
        for k, val in got[(v, "masked")].items():
            ref = g[MASTER_NAME[v]].get(k)
            if ref is not None and abs(val - ref) > 1e-4:
                bad += 1
                print(f"  ⚠️ MISMATCH {v} {k}: mine {val:+.4f} vs master {ref:+.4f}")
    print(f"OUTSIDE CHECK (masked arms vs pdl_master): "
          f"{'ALL MATCH to 4dp' if bad == 0 else f'{bad} MISMATCHES -- STOP'}")
    if bad:
        raise SystemExit("masked arms do not reproduce the master; the driver is suspect")

    def dmean(src):
        out = {}
        for d in present:
            vals = [src.get((rg, d)) for rg in OOD]
            if any(v is None for v in vals):
                return None
            out[d] = float(np.mean(vals))
        return out

    rows = []
    print(f"\n{'eval':15s}" + "".join(f"{v + ' m':>10s}{v + ' u':>10s}{'diff':>8s}" for v in VARIANTS))
    per = {}
    for v in VARIANTS:
        pm, pu = dmean(got[(v, "masked")]), dmean(got[(v, "unmasked")])
        per[v] = (pm, pu)
    for d in present:
        line = ""
        for v in VARIANTS:
            pm, pu = per[v]
            line += f"{pm[d]:>+10.4f}{pu[d]:>+10.4f}{pu[d] - pm[d]:>+8.4f}" if pm and pu else " " * 28
        print(f"{d:15s}{line}")
    print(f"{'MEAN':15s}" + "".join(
        f"{np.mean(list(per[v][0].values())):>+10.4f}{np.mean(list(per[v][1].values())):>+10.4f}"
        f"{np.mean(list(per[v][1].values())) - np.mean(list(per[v][0].values())):>+8.4f}"
        if per[v][0] and per[v][1] else " " * 28 for v in VARIANTS))

    print("\n" + "=" * 100)
    print(f"UNMASKED MINUS MASKED, paired per dataset (n = {len(present)})")
    print("=" * 100)
    for v in VARIANTS:
        pm, pu = per[v]
        if not (pm and pu):
            continue
        d = np.array([pu[e] - pm[e] for e in present])
        p = _st.wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
        verdict = ("UNMASKED better" if d.mean() > 0 else "MASKED better") if p < 0.05 else "no difference"
        print(f"  {v:10s} {d.mean():+.4f}   unmasked wins {int((d > 0).sum())}/{len(present)}   "
              f"Wilcoxon p={p:.4f}   -> {verdict}")
        rows.append((v, f"{d.mean():.4f}", int((d > 0).sum()), len(present), f"{p:.4f}", verdict))

    print("\n  ⚠️ READING. A null here means the mask is COSTLESS, not that it is pointless -- it was")
    print("     introduced to stop the weighter degenerating into an EOS detector, and a null says")
    print("     that safeguard is free. A significant UNMASKED win would mean three headline numbers")
    print("     move AND the token-set asymmetry with the floor (§12) disappears at the same time.")

    outp = Path(args.out)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["variant", "unmasked_minus_masked", "unmasked_wins", "n_evals", "wilcoxon_p",
                    "verdict"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}")
    if missing:
        print(f"⚠️ PARTIAL: {len(present)}/8. Missing {missing}.")


if __name__ == "__main__":
    main()
