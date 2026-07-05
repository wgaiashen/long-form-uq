"""Verify + sanity-check the weighted-MSP method (Track 2).

Two things:
  1. GROUNDING ASSERT: the `constant` weight mode must reduce to plain MSP exactly (q = mean/sum of
     NLL == msp 'perplexity'/'sum'). Hard-asserted per dataset. If this drifts, the aggregation is
     wrong, not the method.
  2. ID SANITY: does the LEARNED weighting (normalised, length-normalised) beat the plain-MSP floor
     in-distribution? Reports weighted-MSP PRR vs msp_sum / perplexity / the constant-mode PRR (which
     must equal the perplexity floor -- a second cross-check).

Reuses the per-token cache loader from attn_pool.py, so it runs wherever attn_pool does (CPU or GPU).

    python scripts/checks/weighted_msp_check.py --datasets sciq
    python scripts/checks/weighted_msp_check.py --datasets sciq,trivia_qa,pubmed_qa,xsum --seeds 1,2,3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import msp, results, weighted_msp  # noqa: E402
from attn_pool import load_per_token  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def msp_floor_prr(records, idx, y, agg):
    """PRR of a plain MSP variant on the test split."""
    unc = [msp.msp_uncertainty(records[i]["token_logprobs"], agg) for i in idx]
    return results.prr([y[i] for i in idx], unc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sciq")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--label-field", default="correctness")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--length-normalise", default="yes", choices=["yes", "no"])
    ap.add_argument("--limit-train", type=int, default=None,
                    help="cap #train examples (fixed seed subsample) for a fast CPU smoke; "
                         "the full-data eval runs on GPU via the ladder.")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ln = args.length_normalise == "yes"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | length_normalise={ln} | label {args.label_field} | seeds {seeds}\n")

    for dataset in args.datasets.split(","):
        loaded = load_per_token(MODEL, dataset, args.layer, args.label_field)
        if loaded is None:
            print(f"==== {dataset}: no per-token cache, skip ====\n")
            continue
        states, split, y, layer, records = loaded
        if np.isnan(y).any():
            print(f"==== {dataset}: missing labels, skip ====\n")
            continue
        tr_idx = [i for i in range(len(states)) if split[i] == "train"]
        te_idx = [i for i in range(len(states)) if split[i] == "test"]
        if args.limit_train is not None and args.limit_train < len(tr_idx):
            rng = np.random.RandomState(0)
            tr_idx = [tr_idx[j] for j in rng.permutation(len(tr_idx))[:args.limit_train]]
        print(f"==== {dataset} (layer {layer}, train {len(tr_idx)}, test {len(te_idx)}) ====", flush=True)

        # 1) GROUNDING: constant mode == plain MSP (exact).
        md = weighted_msp.constant_equals_msp_maxdiff(records, te_idx, ln)
        assert md < 1e-4, f"{dataset}: constant-mode q deviates from plain MSP by {md:.2e} (should be ~0)"
        print(f"  [constant==MSP OK] max|q - msp| = {md:.2e}")

        # 2) The plain-MSP floor (what a learned weighting must beat).
        floor_sum = msp_floor_prr(records, te_idx, y, "sum")
        floor_ppl = msp_floor_prr(records, te_idx, y, "perplexity")
        print(f"  plain MSP floor:  msp_sum {floor_sum:+.3f}   perplexity {floor_ppl:+.3f}")

        # constant-mode PRR through OUR path must equal the matching floor (perplexity if length-norm).
        c_unc = weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                              weight_mode="constant", length_normalise=ln)
        c_prr = results.prr([y[i] for i in te_idx], c_unc)
        ref = floor_ppl if ln else floor_sum
        assert abs(c_prr - ref) < 1e-6, f"{dataset}: constant PRR {c_prr} != floor {ref}"
        print(f"  [constant path OK] weighted-MSP(constant) PRR {c_prr:+.3f} == floor {ref:+.3f}")

        # 3) LEARNED weighting (normalised) -- does it beat the floor ID?
        for mode in ("normalised", "unconstrained"):
            prrs = []
            for sd in seeds:
                unc = weighted_msp.weighted_msp_unc(states, records, y, tr_idx, te_idx, device,
                                                    weight_mode=mode, length_normalise=ln, seed=sd)
                prrs.append(results.prr([y[i] for i in te_idx], unc))
            prrs = np.array(prrs)
            spread = f" +/- {prrs.std():.3f}" if len(seeds) > 1 else ""
            delta = prrs.mean() - ref
            print(f"  weighted-MSP [{mode:13s}] {prrs.mean():+.3f}{spread}   (vs floor {delta:+.3f})")
        print()


if __name__ == "__main__":
    main()
