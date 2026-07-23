"""Orgad exact-answer overlay for weighted-MSP: does restricting the score to answer-bearing tokens
help, ID and OOD?

Joe's overlay (his `exact_answer_overlay.py`) only DREW the answer span for eyeballing; here we USE it to
mask the weighted-MSP sum to the exact-answer tokens (the token span where the gold answer appears in the
generation). It is a cheap, portable, task-agnostic prior on which tokens matter -- and it's short-form
only (long-form / unlocated rows fall back to all tokens = plain weighted-MSP).

For each ladder cell we run weighted-MSP (pairwise loss, to isolate the MASK from the loss) with and
without the Orgad mask, and report PRR + delta + the MSP floor. Masks are built once per source with
`weighted_msp.build_answer_masks` (gold substring match, CPU, no GPU/API). Reuses the Track B cell
machinery for identical splits/seeds.

    python scripts/checks/weighted_msp_orgad.py --seeds 1,2,3
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
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, msp, results, weighted_msp  # noqa: E402
from weighted_msp_blondel import cells, sampled_train_idx, CANDIDATE_SOURCES, LAB, LAYER  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"device {device} | seeds {seeds}", flush=True)

    PT, MASKS = {}, {}
    # Load EVAL TARGETS as well as sources: loading only --sources means an eval target absent from that
    # list is never loaded, yielding zero cells while still exiting 0 (the canonical_ladder/asqa silent
    # failure). Harmless when evals are already a subset. (2026-07-23)
    for d in sorted(set(args.sources.split(",")) | set(globals().get("EVALS", []))):
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None:
            continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            continue
        PT[d] = (states, split, y, records)
        masks, nloc = weighted_msp.build_answer_masks(tok, records)   # Orgad exact-answer span
        MASKS[d] = masks
        print(f"  {d}: {len(states)} rows | exact-answer located {nloc}/{len(records)} "
              f"({100*nloc/len(records):.0f}%)", flush=True)
    sources = set(PT)

    def cell_prr(spec, X, use_mask):
        vals, unc_seeds = [], []
        for sd in seeds:
            train_rows = [(d, i) for d, cap in spec for i in sampled_train_idx(PT[d][1], sd, cap)]
            test_rows = [(X, i) for i in np.where(PT[X][1] == "test")[0]]
            if not train_rows or not test_rows:
                return None, None
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = np.array([y[i] for i in te_idx], dtype=float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            masks = [MASKS[d][i] for d, i in allrows] if use_mask else None
            u = np.asarray(weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                length_normalise=True, seed=sd, loss="pairwise", masks=masks), dtype=float)
            vals.append(results.prr(yte, u)); unc_seeds.append(u)
        return (float(np.mean(vals)), float(np.std(vals))), np.mean(np.stack(unc_seeds), axis=0)

    print(f"\n{'rung':9s} {'eval':10s} {'no-mask':>9s} {'+orgad':>9s} {'delta':>8s} {'floor':>7s}  verdict", flush=True)
    rows = []
    for rung, X, spec in cells(sources):
        if X not in PT:
            continue
        base, ub = cell_prr(spec, X, False)
        org, uo = cell_prr(spec, X, True)
        if base is None or org is None:
            continue
        te = np.where(PT[X][1] == "test")[0]
        yte = PT[X][2][te]
        # FAIR floor (2026-07-22): best of {msp_sum, perplexity, msp_min}; msp_sum is the weakest on all 9.
        _fv, _fname = msp.fair_floor([PT[X][3][i] for i in te], yte, results.prr)
        floor = results.prr(yte, _fv)
        mg, lo, hi, p, sig = paired_bootstrap(yte, uo, ub)     # +orgad vs no-mask
        d = org[0] - base[0]
        print(f"{rung:9s} {X:10s} {base[0]:+9.3f} {org[0]:+9.3f} {d:+8.3f} {floor:+7.3f}  "
              f"CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}", flush=True)
        rows.append({"rung": rung, "eval": X, "no_mask": round(base[0], 4), "orgad": round(org[0], 4),
                     "delta": round(d, 4), "floor": round(floor, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "boot_p": round(p, 4), "significant": sig})

    out = ROOT / "results" / f"weighted_msp_orgad__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "no_mask", "orgad", "delta", "floor",
                                           "ci_lo", "ci_hi", "boot_p", "significant"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
