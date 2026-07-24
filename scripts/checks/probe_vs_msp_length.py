"""Extend the headline result (PART XV.6/XV.8): does the PROBE share MSP's length decay?

XV established: MSP's PRR decays monotonically with generation length across regimes (corr −0.896), while
the hidden-state PROBE does not (sciq Q4: MSP +0.38 vs probe +0.91; cnn/xsum: probe beats MSP at every band).
That asymmetry is the strongest single argument for the white-box bet. It was shown on sciq + cnn + xsum
only. This generalises it to every dataset with a pooled feature cache, ID, by length quartile.

Per dataset: train the mean-pool SAPLMA probe on the train split, then WITHIN each length quartile of the
test split score BOTH the probe and the best MSP-family floor (sum/perplexity/min), and report the gap.

GUARD (the check that cleared a false alarm in XV.2): also report the LABEL distribution per quartile
(mean correctness, std, frac@0, frac@1). If a quartile's PRR looks anomalous it must be checked against the
labels before it is believed -- a degenerate (all-correct or no-variance) subgroup produces a meaningless PRR.

CPU-only: reads the cached pooled features (all layers, we use L15) + records. No GPU, no new extraction.
    python scripts/checks/probe_vs_msp_length.py
"""
import csv as _csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import msp, probe, results  # noqa: E402

MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
LAYER = 15
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad"]


def load(dataset):
    fc = glob.glob(str(ROOT / "cache" / "features" / f"*{dataset}*saplma*.npz"))
    rc = glob.glob(str(ROOT / "cache" / "records" / f"*__{dataset}__ID.jsonl"))
    if not fc or not rc:
        return None
    feats = np.load(fc[0])["feats"][:, LAYER, :]                 # (n, 4096)
    recs = [json.loads(l) for l in open(rc[0])]
    if len(recs) != len(feats):
        print(f"  {dataset}: feats {len(feats)} != records {len(recs)} -> skip", flush=True)
        return None
    y = np.array([r.get("correctness", np.nan) for r in recs], float)
    split = np.array([r.get("split") for r in recs])
    length = np.array([len(r["token_logprobs"]) for r in recs], float)
    return feats, recs, y, split, length


def main():
    out_rows = []
    print(f"{'dataset':14s} {'quartile':>8s} {'n':>5s} {'medlen':>7s} {'meanY':>6s} "
          f"{'MSP':>7s} {'PROBE':>7s} {'gap':>7s}", flush=True)
    print("-" * 72)
    for d in DATASETS:
        loaded = load(d)
        if loaded is None:
            print(f"  {d}: no cache -> skip", flush=True)
            continue
        feats, recs, y, split, length = loaded
        ok = np.isfinite(y)
        tr = np.where((split == "train") & ok)[0]
        te = np.where((split == "test") & ok)[0]
        if len(tr) < 200 or len(te) < 200:
            print(f"  {d}: too few labelled rows (tr={len(tr)}, te={len(te)}) -> skip", flush=True)
            continue
        clf = probe.train_probe_mlp(feats[tr], y[tr], seed=1)
        unc = probe.uncertainty(clf, feats[te])                 # probe uncertainty on test
        qs = np.percentile(length[te], [25, 50, 75])
        bands = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e18)]
        for qi, (lo, hi) in enumerate(bands):
            m = (length[te] > lo) & (length[te] <= hi)
            if m.sum() < 60:
                continue
            yy = y[te][m]
            if yy.std() < 1e-6:                                 # degenerate subgroup -> PRR meaningless
                print(f"  {d} Q{qi+1}: no label variance (mean {yy.mean():.2f}) -> PRR undefined, skipped",
                      flush=True)
                continue
            sub = [recs[i] for i in te[m]]
            mspv = max(results.prr(yy, np.array([msp.msp_uncertainty(r["token_logprobs"], a) for r in sub]))
                       for a in ("sum", "perplexity", "min"))
            prbv = results.prr(yy, unc[m])
            medlen = float(np.median(length[te][m]))
            print(f"{d:14s} {'Q'+str(qi+1):>8s} {int(m.sum()):5d} {medlen:7.0f} {yy.mean():6.2f} "
                  f"{mspv:+7.3f} {prbv:+7.3f} {prbv-mspv:+7.3f}", flush=True)
            out_rows.append({"dataset": d, "quartile": qi + 1, "n": int(m.sum()),
                             "median_len": round(medlen, 1), "mean_correctness": round(float(yy.mean()), 3),
                             "label_std": round(float(yy.std()), 3),
                             "msp_prr": round(mspv, 4), "probe_prr": round(prbv, 4),
                             "gap_probe_minus_msp": round(prbv - mspv, 4)})

    out = ROOT / "results" / f"probe_vs_msp_length__{MODEL_SLUG}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["dataset", "quartile", "n", "median_len", "mean_correctness",
                                           "label_std", "msp_prr", "probe_prr", "gap_probe_minus_msp"])
        w.writeheader(); w.writerows(out_rows)
    # summary: does the probe beat MSP at every length band? does the gap grow with length?
    if out_rows:
        gaps = [r["gap_probe_minus_msp"] for r in out_rows]
        print(f"\nprobe beats MSP in {sum(1 for g in gaps if g > 0)}/{len(gaps)} (dataset,quartile) cells; "
              f"mean gap {np.mean(gaps):+.3f}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
