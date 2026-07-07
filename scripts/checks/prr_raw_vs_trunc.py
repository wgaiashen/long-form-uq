"""Step 5 of the answer-span plan: does truncating to the answer span move PRR?

Reads the step-2 `pertok_trunc` caches (raw and truncated pooled L15 vectors, side by side) and
recomputes PRR for the SAPLMA-mean-L15 probe RAW vs TRUNCATED, for ID and the two OOD pairs the
plan names (trivia_qa->pubmed_qa, xsum->med_quad). No model, no regeneration -- just retrains the
cheap probe on the two pooled representations. Averaged over seeds (the MLP is stochastic).

The append/P(True) method is NOT here: truncation moves the appended-question position, so it needs
a GPU re-run over [clean_text + question] -- flagged as a separate step, not part of this CPU diag.

Output per (cell, pooling): PRR_raw, PRR_trunc, delta. The step-6 decision reads the deltas:
|delta| < ~0.03 and OOD ordering unchanged => keep the compiled numbers (prefer truncated).

    python scripts/checks/prr_raw_vs_trunc.py --seeds 1 2 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, probe, results  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
TRUNC_DIR = ROOT / "cache" / "pertok_trunc"

# ID cells (each dataset vs itself). med_quad is train-only, so it has no ID/OOD EVAL target;
# its truncation test is the SameTask-SOURCE direction (train on med_quad, eval on a QA test set).
ID_DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad"]
OOD_PAIRS = [("trivia_qa", "pubmed_qa")]
# SameTask: does truncating med_quad's TRAINING features change the transfer PRR? (med_quad is a
# same-task QA neighbour, so it trains a probe evaluated on the QA eval sets.)
SAMETASK_PAIRS = [("med_quad", "pubmed_qa"), ("med_quad", "sciq"), ("med_quad", "trivia_qa")]


def load_trunc(dataset):
    p = TRUNC_DIR / f"{cache._slug(MODEL)}__{dataset}__ID__L15.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def prr_for(Xtr, ytr, Xte, yte, seeds):
    """Mean PRR over seeds for a SAPLMA MLP trained on (Xtr,ytr), scored on (Xte,yte)."""
    vals = []
    for sd in seeds:
        clf = probe.train_probe_mlp(Xtr, ytr, seed=sd)
        unc = 1.0 - clf.p_correct(Xte)          # higher = more uncertain
        vals.append(results.prr(yte, unc))
    return float(np.mean(vals)), float(np.std(vals))


def run_cell(name, train_d, eval_d, cache_map, seeds, pool):
    """pool in {'mean','last'}: which pooled vector to use. Trains on train_d's train split,
    evaluates on eval_d's test split (ID when train_d==eval_d)."""
    tr, ev = cache_map[train_d], cache_map[eval_d]
    raw_key, trunc_key = (f"raw_{pool}", f"trunc_{pool}")
    tr_mask = tr["split"] == "train"
    te_mask = ev["split"] == "test"
    if te_mask.sum() == 0:
        print(f"  [{name}] eval dataset has no test split -> skip", flush=True)
        return
    ytr, yte = tr["y"][tr_mask], ev["y"][te_mask]
    # drop any NaN-labelled rows (e.g. an unlabelled source)
    if np.isnan(ytr).any() or np.isnan(yte).any():
        print(f"  [{name}] has NaN labels -> skip", flush=True)
        return
    out = {}
    for tag, key in [("raw", raw_key), ("trunc", trunc_key)]:
        Xtr, Xte = tr[key][tr_mask], ev[key][te_mask]
        out[tag] = prr_for(Xtr, ytr, Xte, yte, seeds)
    draw, dtr = out["raw"][0], out["trunc"][0]
    delta = dtr - draw
    flag = "" if abs(delta) < 0.03 else "  <-- |delta|>=0.03"
    print(f"  [{name:26s} pool={pool:4s}] raw={draw:+.3f}±{out['raw'][1]:.3f}  "
          f"trunc={dtr:+.3f}±{out['trunc'][1]:.3f}  delta={delta:+.3f}{flag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--pools", nargs="+", default=["mean", "last"])
    args = ap.parse_args()

    cache_map = {}
    for d in ID_DATASETS:
        c = load_trunc(d)
        if c is not None:
            cache_map[d] = c
    print(f"loaded truncated caches for: {sorted(cache_map)}", flush=True)

    for pool in args.pools:
        print(f"\n===== SAPLMA-{pool}-L15  RAW vs TRUNCATED  (PRR, seeds={args.seeds}) =====")
        print("--- ID ---")
        for d in ID_DATASETS:
            if d in cache_map:
                run_cell(f"ID {d}", d, d, cache_map, args.seeds, pool)
        print("--- OOD ---")
        for a, b in OOD_PAIRS:
            if a in cache_map and b in cache_map:
                run_cell(f"OOD {a}->{b}", a, b, cache_map, args.seeds, pool)
            else:
                miss = [x for x in (a, b) if x not in cache_map]
                print(f"  [OOD {a}->{b}] missing cache: {miss} -> skip", flush=True)
        print("--- SameTask (train on truncated-vs-raw SOURCE, eval on QA test set) ---")
        for a, b in SAMETASK_PAIRS:
            if a in cache_map and b in cache_map:
                run_cell(f"SameTask {a}->{b}", a, b, cache_map, args.seeds, pool)
            else:
                miss = [x for x in (a, b) if x not in cache_map]
                print(f"  [SameTask {a}->{b}] missing cache: {miss} -> skip", flush=True)


if __name__ == "__main__":
    main()
