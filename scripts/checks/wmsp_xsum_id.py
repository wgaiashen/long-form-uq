"""Close the ID table: weighted-MSP (norm + Blondel) ID PRR on xsum.

WHY: every weighted-MSP result file so far covers sciq/trivia_qa/pubmed_qa only -- xsum was never run
(the wMSP driver reused the QA per-token caches). xsum's per-token L15 cache DOES exist, and records
carry token_logprobs, so this is a pure-CPU add: train the wMSP weight model on the xsum train split,
score the test split, report 3-seed mean +/- std. Anchored with the plain-MSP floor (sum + perplexity)
and, for context, the cached uniform/attention ID PRRs from the aggregation table.

Reuses the verified weighted_msp.weighted_msp_unc drop-in (same shape as attn_pool.attn_unc), so this
adds no new estimation code -- only wiring xsum through it.

    python scripts/checks/wmsp_xsum_id.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import cache, msp, results, weighted_msp  # noqa: E402
from attn_pool import load_per_token  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--length-normalise", default="yes", choices=["yes", "no"])
    ap.add_argument("--out", default=str(ROOT / "results" / f"wmsp_xsum_id__{cache._slug(MODEL)}.csv"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ln = args.length_normalise == "yes"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | length_normalise={ln}", flush=True)

    loaded = load_per_token(MODEL, "xsum", args.layer, LAB)
    if loaded is None:
        sys.exit("ERROR: no per-token cache for xsum -- run scripts/01h_pertoken.py --dataset xsum first.")
    states, split, y, layer, records = loaded
    if np.isnan(y).any():
        sys.exit("ERROR: xsum has NaN correctness labels -- cannot evaluate.")
    tr_idx = [i for i in range(len(states)) if split[i] == "train"]
    te_idx = [i for i in range(len(states)) if split[i] == "test"]
    yte = np.array([y[i] for i in te_idx], dtype=float)
    print(f"xsum: {len(tr_idx)} train, {len(te_idx)} test | label={LAB}", flush=True)

    # Plain-MSP floors (unsupervised -> seed-independent).
    msp_sum = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "sum") for i in te_idx])
    perpl = np.array([msp.msp_uncertainty(records[i]["token_logprobs"], "perplexity") for i in te_idx])
    prr_floor_sum = results.prr(yte, msp_sum)
    prr_floor_ppl = results.prr(yte, perpl)
    # the honest bar is the best of the three (msp_min was missing here); report which one won.
    _fv, _fname = msp.primary_floor([records[i] for i in te_idx])  # PRE-REGISTERED msp_min bar (2026-07-24)
    prr_floor_fair = results.prr(yte, _fv)
    print(f"  fair_floor = {_fname} {prr_floor_fair:+.3f}  (sum {prr_floor_sum:+.3f}, ppl {prr_floor_ppl:+.3f})",
          flush=True)

    # weighted-MSP, two losses, over the seeds.
    prr = {"weighted_msp_norm_pairwise": [], "weighted_msp_blondel": []}
    have_bl = weighted_msp._HAVE_TORCHSORT
    for sd in seeds:
        u_pair = weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                               weight_mode="normalised", length_normalise=ln,
                                               seed=sd, loss="pairwise")
        prr["weighted_msp_norm_pairwise"].append(results.prr(yte, np.asarray(u_pair, float)))
        if have_bl:
            u_bl = weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                                 weight_mode="normalised", length_normalise=ln,
                                                 seed=sd, loss="blondel")
            prr["weighted_msp_blondel"].append(results.prr(yte, np.asarray(u_bl, float)))
        print(f"  seed {sd} done", flush=True)

    rows = [{"eval": "xsum", "rung": "ID", "method": "msp_sum (floor)",
             "prr_mean": round(prr_floor_sum, 4), "prr_std": 0.0, "n_seeds": len(seeds)},
            {"eval": "xsum", "rung": "ID", "method": "perplexity (floor)",
             "prr_mean": round(prr_floor_ppl, 4), "prr_std": 0.0, "n_seeds": len(seeds)}]
    for m, v in prr.items():
        if v:
            rows.append({"eval": "xsum", "rung": "ID", "method": m,
                         "prr_mean": round(float(np.mean(v)), 4),
                         "prr_std": round(float(np.std(v)), 4), "n_seeds": len(v)})
    if not have_bl:
        print("  (torchsort not available -> Blondel row omitted)", flush=True)

    print("\n==== xsum ID ====", flush=True)
    for r in rows:
        print(f"  {r['method']:28s} {r['prr_mean']:+.4f} +/- {r['prr_std']:.4f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["eval", "rung", "method", "prr_mean", "prr_std", "n_seeds"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
