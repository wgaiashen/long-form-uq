"""STAGE 1 — Multi-head MultiMax DIAGNOSTIC (the deciding number BEFORE any PRR).

The single-head MultiMax we tested is the degenerate n_head=1 case (one token per generation). The paper's eq 9 is
`f = Sum_h max_j (v_h . y_ij)`: K heads, K value projections, each free to pick a DIFFERENT token. Under a max what
matters is the per-head VALUE projections (the head weights W_h), NOT the attention distribution -- so the §D.4
"attention collapses to 0.999 correlation" result does NOT by itself imply the per-head argmaxes coincide.

⭐ THE DECIDING NUMBER (reported before any PRR): per example, how many DISTINCT tokens do the 4 heads' argmaxes
select? mean over examples (1.0 = full collapse -> eq-9 cannot work in our setting -> STOP; materially > 1.0 ->
multi-token coverage survived -> Stage 2). Also reports per-head ATTENTION correlation on the SAME cells, so we can
say whether attention-collapse and value-projection-collapse are the same phenomenon or different (the claim to test,
not assume).

Scope (TIGHT, per the request): 3 evals (pubmed=concentrated, cnn=spread, xsum=no-prob-signal -- one per §D.6
family) x {ID, DiffTask-long} x seed 1 = 6 cells. Trains a 4-head pooler and PERSISTS q_rest AND heads_rest (the
gap that blocked the post-hoc route). NO grid. Token set for the argmax = the FULL G+1 window (anchor included),
matching 3A's single-head `s = x @ W ; s.max()`.

    python scripts/checks/mh_multimax_diag.py --out results/mh_multimax_diag__meta-llama_Meta-Llama-3.1-8B.csv
"""
import argparse
import csv as _csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, results                                               # noqa: E402
from aggregation_table import attn_unc, load_per_token                       # noqa: E402
from attn_pool import train_attn, select_temperature, pad_batch              # noqa: E402
from xl_rungs import build_rows, eval_split, label_of                        # noqa: E402
import probedriftlong as pdl                                                 # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
EVALS = ["pubmed_qa", "cnn_dailymail", "xsum"]
RUNGS = {"ID", "DiffTask-long"}
SEED = 1
K = 4


def head_value_weights(model):
    """The K per-head VALUE projections W_h (each (d,)) = the linear head weights. heads_all[0] = primary head,
    [1..] = heads_rest. Under MultiMax eq 9 these are the v_h that score each token: s_hj = W_h . x_j."""
    heads = [model.head] + (list(model.heads_rest) if model.heads_rest is not None else [])
    return np.stack([h.weight.detach().cpu().numpy().ravel() for h in heads])   # (K, d)


def diagnose_cell(model, states, te_idx, device):
    """Returns (mean_distinct, hist[1..K], mean_attn_corr). distinct = # unique argmax tokens across the K value
    projections, per example (the full G+1 window, matching 3A). attn_corr = mean pairwise Pearson corr across the
    K attention distributions (tests whether the QUERIES collapsed, §D.4-style), on the same test examples."""
    W = head_value_weights(model)                     # (K, d)
    model.eval()
    distinct = []; corrs = []
    with torch.no_grad():
        for i in te_idx:
            x = states[i]                             # (T, d), full window incl. anchor row 0
            s = x @ W.T                               # (T, K) per-token per-head value score
            am = s.argmax(axis=0)                     # (K,) argmax token per head
            distinct.append(len(set(am.tolist())))
            # attention correlation across the K heads (the query-collapse measure)
            X, mask, pos = pad_batch([x], device)
            _, a = model(X, mask, pos)                # a: (1, T, K) in the multi-head path
            A = a[0, : x.shape[0]].cpu().numpy()      # (T, K)
            if A.shape[0] >= 2:
                c = np.corrcoef(A.T)                  # (K, K); constant columns -> nan
                corrs.append(np.nanmean(c[np.triu_indices(K, 1)]))
    distinct = np.array(distinct)
    hist = {k: float((distinct == k).mean()) for k in range(1, K + 1)}
    return float(distinct.mean()), hist, (float(np.nanmean(corrs)) if corrs else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / f"mh_multimax_diag__{SLUG}.csv"))
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probes = ROOT / "cache" / "probes"; probes.mkdir(parents=True, exist_ok=True)
    print(f"[mh-diag] device {device} | evals {EVALS} | rungs {sorted(RUNGS)} | K={K} | seed {SEED}", flush=True)

    # widened pool (DiffTask-long draws from other sources) + the 3 evals
    PT = {}
    for d in sorted(set(pdl.LONG_SRC) | set(EVALS)):
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))
        if loaded is None:
            print(f"  {d}: no pertok -> skip", flush=True); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(y)
        if not finite.any():
            print(f"  {d}: unlabelled -> skip", flush=True); continue
        if not finite.all():
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = y[keep]
        PT[d] = (states, split, y, records)
    sources = set(PT)

    rows = []
    for rung, X, spec in pdl.cells_long(sources, EVALS):
        if X not in EVALS or rung not in RUNGS or X not in PT:
            continue
        train_rows, test_rows = build_rows(X, spec, PT, SEED, pdl.sampled_train_idx)
        if not train_rows or not test_rows:
            print(f"  [{rung} {X}] empty rows -> skip", flush=True); continue
        n_tr = len(train_rows); tr_idx = list(range(n_tr)); te_idx = list(range(n_tr, n_tr + len(test_rows)))
        allrows = train_rows + test_rows
        y = np.array([PT[d][2][i] for d, i in allrows], float)
        yte = np.array([y[i] for i in te_idx], float)
        states = [PT[d][0][i] for d, i in allrows]

        # same seed/splits/val-temperature as armA (the control)
        best_T, _ = select_temperature(states, y, tr_idx, device, SEED, False, False)
        mA = train_attn(states, y, tr_idx, device, seed=SEED, temperature=best_T)          # single-head reference
        prr_armA = results.prr(yte, np.asarray(attn_unc(mA, states, te_idx, device), float))
        # the 4-head pooler (persists q_rest + heads_rest)
        mh = train_attn(states, y, tr_idx, device, seed=SEED, temperature=best_T, n_query=K, n_head=K)
        prr_mh = results.prr(yte, np.asarray(attn_unc(mh, states, te_idx, device), float))  # mean-of-sigmoids ensemble
        pk = probes / f"{SLUG}__{X}__{rung}_mh{K}_s{SEED}__L{args.layer}.pkl"
        with open(pk, "wb") as fh:
            pickle.dump({"model": mh, "best_T": float(best_T), "n_query": K, "n_head": K,
                         "eval": X, "rung": rung, "seed": SEED, "layer": args.layer}, fh)

        mean_distinct, hist, attn_corr = diagnose_cell(mh, states, te_idx, device)
        rows.append({"eval": X, "rung": rung, "n_test": len(te_idx), "mean_distinct": round(mean_distinct, 3),
                     "frac1": round(hist[1], 3), "frac2": round(hist[2], 3), "frac3": round(hist[3], 3),
                     "frac4": round(hist[4], 3), "attn_corr": round(attn_corr, 4),
                     "armA_prr": round(prr_armA, 4), "mh_ensemble_prr": round(prr_mh, 4)})
        print(f"  [{rung:14s} {X:13s}] ⭐ mean-distinct-argmax {mean_distinct:.3f}/4  "
              f"(1:{hist[1]:.2f} 2:{hist[2]:.2f} 3:{hist[3]:.2f} 4:{hist[4]:.2f})  | attn-corr {attn_corr:+.3f}  "
              f"| armA {prr_armA:+.3f}  mh-ens {prr_mh:+.3f}  [pooler saved]", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["eval", "rung", "n_test", "mean_distinct", "frac1", "frac2", "frac3",
                                            "frac4", "attn_corr", "armA_prr", "mh_ensemble_prr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[mh-diag] wrote {args.out} ({len(rows)} cells).", flush=True)
    print("[mh-diag] DECISION: mean-distinct near 1.0 across cells -> value-projection collapse, eq-9 dead, STOP.\n"
          "          Materially > 1.0 (esp. OOD) -> coverage survived -> Stage 2 (post-hoc + trained-with-max).", flush=True)


if __name__ == "__main__":
    main()
