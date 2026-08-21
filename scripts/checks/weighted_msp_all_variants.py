"""ALL weighted-MSP variants x ALL eval datasets x the full 5-rung ladder, in ONE tidy CSV.

The per-variant results were scattered across contribution_ladder (norm/unc), weighted_msp_smoothing
(shrink/kl/smooth, QA-only, ID/LOO/DiffTask only) and weighted_msp_eps_sweep (Blondel, QA-only). xsum had
only norm/blondel and cnn had none. This consolidates every variant onto EVERY eval dataset so the
presentation tables are complete and consistent (same seeds, same length-norm, same rungs).

Variants (all length-normalised, weight_mode as noted):
  wMSP-normalised       softmax*n weights (avg 1)                 the primary
  wMSP-unconstrained    raw weights (no softmax)                  ID-strong, OOD-fragile
  wMSP-Blondel          normalised + Blondel soft-rank loss (eps=0.1)
  shrink@2 / shrink@10  normalised + shrink-to-uniform penalty (lambda 2 / 10)   (P1.1 moderation)
  kl@2                  normalised + KL-to-uniform penalty (lambda 2)
  smooth_n3 / smooth_n5 normalised + neighbour-average logits over 3 / 5 tokens

Evals: sciq, trivia_qa, pubmed_qa, xsum, cnn_dailymail (whichever have a labelled per-token cache).
Rungs: ID, SameTask, LOO, OneDatasetDiffTask, DiffTask (get_training_spec, filtered to cached sources).
Output: results/weighted_msp_all_variants__<slug>.csv  (eval, rung, variant, prr_mean, prr_std, floor, n_seeds).

    python scripts/checks/weighted_msp_all_variants.py --seeds 1,2,3
"""
import argparse
import csv as _csv
import functools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402

from luq import cache, msp, results, weighted_msp, weighting  # noqa: E402
from aggregation_table import load_per_token  # noqa: E402
import xl_rungs  # noqa: E402  (shared organic-ProbeDriftXL rung machinery)
from xl_rungs import label_of  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
# Organic ProbeDriftXL: the 5 core datasets + the 3 XL long-form eval targets (med_quad/samsum/ExpertQA).
EVALS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa",
         "asqa"]
CANDIDATES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad", "samsum", "cnn_dailymail"]

# wMSP-normalised ID anchors (master table) -- a soft sanity check that the wiring is unchanged.
ID_ANCHOR = {"sciq": 0.820, "trivia_qa": 0.790, "pubmed_qa": 0.557}
GATE_TOL = 0.04

# (variant name, kwargs to weighted_msp_unc). length_normalise=True is added for all.
CONFIGS = [
    ("wMSP-normalised", {"weight_mode": "normalised"}),
    ("wMSP-unconstrained", {"weight_mode": "unconstrained"}),
    ("wMSP-Blondel", {"weight_mode": "normalised", "loss": "blondel"}),
    ("shrink@2", {"weight_mode": "normalised", "reg": weighting.shrink_to_uniform, "reg_lambda": 2.0}),
    ("shrink@10", {"weight_mode": "normalised", "reg": weighting.shrink_to_uniform, "reg_lambda": 10.0}),
    # W2: the Blondel differentiable-rank loss applied to the KEEP-and-develop shrink variants (it was
    # previously wired ONLY for plain `normalised`). Paired against shrink@2 / shrink@10 above, which are
    # identical except loss="pairwise" -> a clean loss-only comparison.
    ("shrink@2-blondel", {"weight_mode": "normalised", "reg": weighting.shrink_to_uniform,
                          "reg_lambda": 2.0, "loss": "blondel"}),
    ("shrink@10-blondel", {"weight_mode": "normalised", "reg": weighting.shrink_to_uniform,
                           "reg_lambda": 10.0, "loss": "blondel"}),
    ("kl@2", {"weight_mode": "normalised", "reg": weighting.kl_to_uniform, "reg_lambda": 2.0}),
    ("entropy_hinge@2", {"weight_mode": "normalised",
                         "reg": functools.partial(weighting.entropy_hinge, threshold=0.7), "reg_lambda": 2.0}),
    ("smooth_n3", {"weight_mode": "normalised", "smooth_n": 3}),
    ("smooth_n5", {"weight_mode": "normalised", "smooth_n": 5}),
]
SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"),
            ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("OneDatasetDiffTask", "OOD_ONE_DATASET_DIFF_TASK"),
            ("DiffTask", "OOD_DIFF_TASK")]


def sampled(split, seed, cap):
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def cells(sources):
    """Dispatching rung generator (keystone->get_training_spec faithful, XL->family taxonomy). Kept as a
    thin wrapper so the ladders that `from weighted_msp_all_variants import cells` pick up the XL evals."""
    return xl_rungs.cells(sources, EVALS)


def main():
    global EVALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--evals", default=",".join(EVALS),
                    help="restrict eval targets (e.g. med_quad,samsum,expertqa for a fast XL-only run).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    EVALS = args.evals.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    have_bl = weighted_msp._HAVE_TORCHSORT
    print(f"device {device} | seeds {seeds} | torchsort={have_bl} | variants={[c[0] for c in CONFIGS]}", flush=True)

    PT = {}
    for d in sorted(set(CANDIDATES) | set(EVALS)):    # load training SOURCES and EVAL TARGETS (e.g. ExpertQA)
        loaded = load_per_token(MODEL, d, args.layer, label_of(d))   # per-target label (ExpertQA=faithfulness)
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        states, split, y, _, records = loaded
        finite = np.isfinite(np.asarray(y, dtype=float))
        if not finite.any():
            print(f"  {d}: unlabelled ({label_of(d)}) -> skip"); continue
        if not finite.all():                          # ExpertQA faithfulness: keep the ~1724 labelled rows
            keep = np.where(finite)[0]
            states = [states[k] for k in keep]; records = [records[k] for k in keep]
            split = split[keep]; y = np.asarray(y)[keep]
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows (label={label_of(d)})", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells(sources):
        spec = [(d, c) for d, c in spec if d in PT]
        if not spec:
            continue
        _, te0 = xl_rungs.eval_split(PT[X][1])         # baked-in for core; deterministic carve for XL evals
        if len(te0) == 0:
            print(f"[{rung}/{X}] no test split -> skip", flush=True); continue
        yte = np.array([PT[X][2][i] for i in te0], dtype=float)
        # PRIMARY floor (2026-07-24 meeting): the PRE-REGISTERED msp_min bar, fixed across datasets
        # (replaces the rejected max-of-three). Dual-report vs the strongest free score is done at the
        # project record layer from the 3-variant rows.
        _fv, _fname = msp.primary_floor([PT[X][3][i] for i in te0])
        floor_prr = results.prr(yte, _fv)

        cfg_prr = {c[0]: [] for c in CONFIGS}
        for sd in seeds:
            train_rows, test_rows = xl_rungs.build_rows(X, spec, PT, sd, sampled)
            if not train_rows:
                continue
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            states = [PT[d][0][i] for d, i in allrows]
            recs = [PT[d][3][i] for d, i in allrows]
            for name, kw in CONFIGS:
                if name == "wMSP-Blondel" and not have_bl:
                    continue
                try:
                    u = np.asarray(weighted_msp.weighted_msp_unc(
                        states, recs, y, tr_idx, te_idx, device, length_normalise=True, seed=sd, **kw),
                        dtype=float)
                    cfg_prr[name].append(results.prr(yte, u))
                except Exception as e:  # noqa: BLE001 -- one variant failing must not sink the run
                    print(f"    !! {name} failed on {rung}/{X} seed {sd}: {type(e).__name__}: {e}", flush=True)

        print(f"\n[{rung:18s}] eval={X}  floor {floor_prr:+.3f}", flush=True)
        for name, _ in CONFIGS:
            if not cfg_prr[name]:
                continue
            mean, std = float(np.mean(cfg_prr[name])), float(np.std(cfg_prr[name]))
            out_rows.append({"eval": X, "rung": rung, "variant": name, "prr_mean": round(mean, 4),
                             "prr_std": round(std, 4), "floor": round(floor_prr, 4), "n_seeds": len(cfg_prr[name])})
            print(f"    {name:18s} {mean:+.3f}±{std:.3f}", flush=True)

        # soft ID-anchor sanity for the QA sets
        if rung == "ID" and X in ID_ANCHOR and cfg_prr["wMSP-normalised"]:
            got = float(np.mean(cfg_prr["wMSP-normalised"]))
            if abs(got - ID_ANCHOR[X]) >= GATE_TOL:
                print(f"    [WARN] wMSP-normalised ID {X} {got:.3f} != anchor {ID_ANCHOR[X]} (wiring drift?)", flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / f"weighted_msp_all_variants__{cache._slug(MODEL)}.csv")
    cols = ["eval", "rung", "variant", "prr_mean", "prr_std", "floor", "n_seeds"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out} ({len(out_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
