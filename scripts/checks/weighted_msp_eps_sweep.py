"""Understand (not replace) the Blondel result: sweep torchsort's smoothing strength ε.

The Blondel loss result was MIXED — it helps ID (esp. pubmed) and the harshest DiffTask, but HURTS the
LOO mixture. That help-ID / hurt-LOO pattern is the classic signature of OVERFITTING: at a small ε the
soft rank is nearly the exact (hard) rank, so the loss fits the training ordering very tightly and
generalises worse under shift. torchsort's `regularization_strength` (= ε) controls this: LARGER ε =
SMOOTHER soft rank = MORE regularised. So sweeping ε should trace an overfitting curve, and the question
is whether a smoother ε RECOVERS LOO without giving back the ID / DiffTask gains.

For each ladder cell we report the pairwise loss (fixed reference), the MSP floor (fixed), and the
blondel loss at every ε. Reuses the exact cell machinery from weighted_msp_blondel.py (same splits,
same seeds) so the ε=0.1 column reproduces the Track B result. CPU; needs torchsort.

    python scripts/checks/weighted_msp_eps_sweep.py --eps 0.03 0.1 0.3 1.0 3.0 10.0 --seeds 1,2,3
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
# reuse the verified cell assembly + splits from the Track B script (no duplication)
from weighted_msp_blondel import cells, sampled_train_idx, EVALS, CANDIDATE_SOURCES, LAB, LAYER  # noqa: E402
from aggregation_table import load_per_token  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sources", default=",".join(CANDIDATE_SOURCES))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not weighted_msp._HAVE_TORCHSORT:
        sys.exit("torchsort not installed -> cannot sweep eps. Run pbs/torchsort_setup.pbs first.")
    print(f"device {device} | seeds {seeds} | eps sweep {args.eps} | torchsort ok", flush=True)

    PT = {}
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
    sources = set(PT)
    print(f"sources: {sorted(sources)}", flush=True)

    def prr_cell(spec, X, loss, eps):
        """Mean PRR over seeds for weighted-MSP(normalised) at a given loss/eps on one cell."""
        vals = []
        for sd in seeds:
            train_rows = [(d, i) for d, cap in spec for i in sampled_train_idx(PT[d][1], sd, cap)]
            test_rows = [(X, i) for i in np.where(PT[X][1] == "test")[0]]
            if not train_rows or not test_rows:
                return None
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = np.array([y[i] for i in te_idx], dtype=float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            u = np.asarray(weighted_msp.weighted_msp_unc(
                states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                length_normalise=True, seed=sd, loss=loss, blondel_eps=eps), dtype=float)
            vals.append(results.prr(yte, u))
        return float(np.mean(vals))

    def floor_cell(X):
        te = np.where(PT[X][1] == "test")[0]
        yte = PT[X][2][te]
        # FAIR floor (2026-07-22): best of {msp_sum, perplexity, msp_min}; msp_sum is the weakest on all 9.
        u, _fname = msp.primary_floor([PT[X][3][i] for i in te])  # PRE-REGISTERED msp_min bar (2026-07-24)
        return float(results.prr(yte, u))

    eps_hdr = "".join(f" bl@{e:<5g}" for e in args.eps)
    print(f"\n{'rung':9s} {'eval':10s} {'pairwise':>9s} {'floor':>7s}{eps_hdr}", flush=True)
    rows = []
    for rung, X, spec in cells(sources):
        if X not in PT:
            continue
        pair = prr_cell(spec, X, "pairwise", 0.0)
        fl = floor_cell(X)
        bl = {e: prr_cell(spec, X, "blondel", e) for e in args.eps}
        cells_str = "".join(f" {bl[e]:+6.3f}" if bl[e] is not None else "   -   " for e in args.eps)
        print(f"{rung:9s} {X:10s} {pair:+9.3f} {fl:+7.3f}{cells_str}", flush=True)
        row = {"rung": rung, "eval": X, "pairwise": round(pair, 4), "floor": round(fl, 4)}
        for e in args.eps:
            row[f"blondel_eps_{e}"] = round(bl[e], 4) if bl[e] is not None else None
        rows.append(row)

    out = ROOT / "results" / f"weighted_msp_eps_sweep__{cache._slug(MODEL)}.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rung", "eval", "pairwise", "floor"] +
                            [f"blondel_eps_{e}" for e in args.eps])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)
    print("READ: does a LARGER eps recover the LOO cells toward pairwise WITHOUT losing the ID-pubmed / "
          "DiffTask gains? If so, the Blondel 'mixed' result was an eps (overfitting) artifact.", flush=True)


if __name__ == "__main__":
    main()
