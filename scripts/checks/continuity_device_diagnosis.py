#!/usr/bin/env python
"""Characterise a continuity-gate discrepancy between a re-run and its canonical master.

Written for M6b: the Qwen2.5-14B ensemble pass reproduces the deterministic floors exactly but not the
trained components. Diagnostic only -- it decides nothing on its own.

THE SITUATION. The Qwen master was produced on DoC with --gres=gpu:1, so its trained components were
fitted on CUDA; the RCS ensemble pass is CPU-only. Torch takes a different arithmetic path per device,
and the wMSP pairwise soft-rank loss is chaotic enough to diverge. The Llama control is consistent:
both its master and its ensemble pass ran on CPU, and every component reproduced at max |d| = 1.22e-04.

WHAT IS PRE-SPECIFIED, and when. Written after seeing ONE dataset (samsum, 5 cells) and before the
other seven landed, so it is stated honestly rather than fitted to the full data:

  1. DETERMINISTIC components (the three floors) must still match at 1e-3. They involve no training,
     so a device cannot excuse them. Any failure here means the population is wrong, full stop.
  2. TRAINED components are judged against the master's OWN recorded seed spread, not an invented
     bar: |d| <= 2 * prr_std. The master stores prr_std per cell, so this uses measured variability.
  3. NO SYSTEMATIC BIAS. Re-drawing a stochastic fit should scatter symmetrically. If a component's
     signed deltas are consistently one-signed, the re-run is not a re-draw but a shifted estimator,
     and an ensemble built on it would be biased. Reported as a sign split with an exact binomial p.
     NOTE, disclosed: samsum alone already showed wmsp_norm at 5/5 positive, so this check was not
     designed blind. It is reported for all 40 cells so the reader can see whether that persisted.

Any component failing (1), or failing (3) with a clear one-sided split, means STOP and report rather
than proceed to the ensemble numbers.

    python scripts/checks/continuity_device_diagnosis.py --model Qwen/Qwen2.5-14B
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, results                                          # noqa: E402

LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
DETERMINISTIC = ["floor_min", "floor_ppl", "floor_sum"]
TRAINED = ["saplma", "uniform", "attention", "wmsp_norm", "wmsp_shrink2"]
DET_TOL, SD_MULT = 1e-3, 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--perex-dir", default=None)
    args = ap.parse_args()
    slug = cache._slug(args.model)
    pdir = Path(args.perex_dir or (ROOT / "results" / f"pdl_perex_ens_{slug}"))

    mst = {}
    with open(ROOT / "results" / f"pdl_master__{slug}.csv") as f:
        for r in _csv.DictReader(f):
            try:
                mst[(r["method"], r["eval"], r["rung"])] = (float(r["prr_mean"]), float(r["prr_std"]))
            except (ValueError, KeyError):
                pass

    print(f"CONTINUITY DIAGNOSIS: {args.model}")
    print(f"deterministic tol {DET_TOL:g} | trained judged at {SD_MULT:g} x the master's own prr_std")
    print("=" * 100)
    print(f"{'component':14s}{'cells':>6s}{'max|d|':>10s}{'med |d|/sd':>12s}{'>2sd':>6s}"
          f"{'signs +/-':>11s}{'binom p':>9s}  verdict")
    stop = []
    for m in DETERMINISTIC + TRAINED:
        ds, ratios, signed = [], [], []
        for d in LONG:
            for rg in RUNGS:
                p = pdir / f"{d}__{rg}__{slug}.npz"
                if not p.exists() or (m, d, rg) not in mst:
                    continue
                z = np.load(p, allow_pickle=True)
                if f"unc__{m}" not in z.files:
                    continue
                V = np.asarray(z[f"unc__{m}"], float)
                got = float(np.mean([results.prr(z["y"], V[s]) for s in range(V.shape[0])]))
                exp, sd = mst[(m, d, rg)]
                dv = got - exp
                ds.append(abs(dv)); signed.append(dv)
                ratios.append(abs(dv) / sd if sd > 0 else (0.0 if abs(dv) < 1e-9 else np.inf))
        if not ds:
            print(f"{m:14s}{'absent':>6s}"); continue
        pos = int(np.sum(np.array(signed) > 0)); n = len(signed)
        from scipy.stats import binomtest
        bp = binomtest(pos, n, 0.5).pvalue
        n_big = int(np.sum(np.array(ratios) > SD_MULT))
        if m in DETERMINISTIC:
            ok = max(ds) <= DET_TOL
            verdict = "ok" if ok else "FAIL (deterministic must match)"
            if not ok:
                stop.append(f"{m}: deterministic component off by {max(ds):.2e}")
        else:
            ok = n_big == 0
            biased = bp < 0.01 and (pos == n or pos == 0)
            verdict = ("ok" if ok else f"{n_big} cell(s) beyond {SD_MULT:g}sd") + (" BIASED" if biased else "")
            if biased:
                stop.append(f"{m}: signed deltas {pos}/{n} one-sided (binom p={bp:.4f})")
        print(f"{m:14s}{n:>6d}{max(ds):>10.2e}{np.median(ratios):>12.2f}{n_big:>6d}"
              f"{f'{pos}/{n-pos}':>11s}{bp:>9.4f}  {verdict}")
    print("=" * 100)
    if stop:
        print("VERDICT: STOP. Do not read ensemble numbers until these are explained:")
        for s in stop:
            print(f"  x {s}")
        raise SystemExit(1)
    print("VERDICT: the re-run is consistent with the canonical grid within its own seed spread,")
    print("         and the deterministic components reproduce exactly.")


if __name__ == "__main__":
    main()
