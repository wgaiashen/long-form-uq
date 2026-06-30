"""Diagnostic: recompute the pooled vector and compare ELEMENTWISE to the cached SAPLMA feature.

No theories. For a handful of records, recompute the layer-L hidden states teacher-forced (fp32 +
eager, matching the keystone), pool several candidate windows, and report max|diff| and cosine vs
the cached SAPLMA feature feat[idx, L]. Whichever window matches (~0) is the one the feature used;
if none match, the feature was produced by a different computation (e.g. generation-time pooling).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, generate  # noqa: E402
from luq.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting="ID")
    key = cache.run_key(args.model, args.dataset, "ID")
    records = cache.load_records(cfg.cache_dir, key)[: args.n]
    feat = cache.load_features(cfg.cache_dir, key, "saplma")  # (N, n_layers, hidden)
    L = args.layer

    model, tok = generate.load_model(args.model, attn_implementation="eager", dtype=torch.float32)

    # Candidate windows to test, as (name, lo_offset_from_P, hi expression).
    def windows(P, G):
        return {
            "[P-1:P+G]":   (P - 1, P + G),
            "[P:P+G]":     (P, P + G),
            "[P-1:P+G-1]": (P - 1, P + G - 1),
            "[P:P+G-1]":   (P, P + G - 1),
            "[P-1:P+G+1]": (P - 1, P + G + 1),
        }

    diffs = {k: [] for k in windows(2, 2)}
    cos = {k: [] for k in windows(2, 2)}
    for i, r in enumerate(records):
        p_ids, g_ids = list(r["prompt_token_ids"]), list(r["gen_token_ids"])
        P, G = len(p_ids), len(g_ids)
        states = generate.recompute_states(model, tok, p_ids + g_ids, [L])
        s = states[0]  # (P+G, hidden), layer L
        f = feat[i, L, :]
        for name, (lo, hi) in windows(P, G).items():
            lo2, hi2 = max(0, lo), min(s.shape[0], hi)
            pooled = s[lo2:hi2].mean(dim=0).numpy()
            diffs[name].append(float(np.abs(pooled - f).max()))
            denom = (np.linalg.norm(pooled) * np.linalg.norm(f)) or 1.0
            cos[name].append(float(np.dot(pooled, f) / denom))

    print(f"{args.dataset} L{L}, {len(records)} records: pooled-recompute vs cached SAPLMA feature")
    print(f"  {'window':14s}{'mean max|diff|':>16}{'mean cosine':>14}")
    for name in diffs:
        print(f"  {name:14s}{np.mean(diffs[name]):>16.4f}{np.mean(cos[name]):>14.5f}")
    print("cosine ~1.0 and max|diff| ~0 => that window reproduces the cached feature.")


if __name__ == "__main__":
    main()
