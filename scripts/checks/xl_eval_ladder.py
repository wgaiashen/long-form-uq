"""med_quad + samsum as FIRST-CLASS long-form eval targets (ProbeDrift-XL breadth).

Motivation: the ladder currently has only ONE long-form QA eval (pubmed) and one summarisation eval
(xsum, + cnn). med_quad (medical QA, long_qa) and samsum (dialogue summarisation) are already fully
generated AND judge-labelled (gpt-5-mini `correctness`, 1800 rows each, per-token L15 cached) -- they
have been TRAINING sources all along. This script promotes them to eval TARGETS, which needs no GPU
and no judge (everything is cached): a pure CPU re-slice + ladder run.

CLEAN same-label eval (unlike the ExpertQA ladder): med_quad/samsum carry `correctness`, the SAME label
every training source carries, so there is NO cross-label caveat -- the ID->OOD drop is a pure task shift.

XL divergence (documented): ProbeDrift ships med_quad/samsum as training/validation sources with NO
held-out eval split, so we CARVE a seeded ID train/test split from each 1800-row pool (a legitimate,
reproducible ID split; it is NOT a ProbeDrift keystone eval). The eval target is EXCLUDED from its own
OOD training sources (no leakage). Rungs are task-family aware:
  med_quad (long_qa):  SameTask=[pubmed_qa]           DiffTask=[xsum,samsum,cnn]  LOO=all others
  samsum   (summ):     SameTask=[xsum,cnn_dailymail]  DiffTask=[sciq,trivia,pubmed,med_quad]  LOO=all others

Methods per rung: msp_floor, saplma (mean-pool+MLP), uniform (frozen-q pooler), attention (pooler,
temperature selected ONCE on ID), wMSP-unconstrained (the OOD-winning variant) + wMSP-normalised.
3 seeds; MSP floor in every rung. CPU only, no GPU, no judge £.

    python scripts/checks/xl_eval_ladder.py --seeds 1,2,3
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

from luq import cache, msp, probe, results, weighted_msp  # noqa: E402
from aggregation_table import load_per_token, attn_unc  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
LAB = "correctness"
EVAL_TARGETS = ["med_quad", "samsum"]
ALL = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad", "samsum", "cnn_dailymail"]
# fine task label (for SameTask) and broad family (for DiffTask = the opposite family)
FINE = {"sciq": "short_qa", "trivia_qa": "short_qa", "pubmed_qa": "long_qa", "med_quad": "long_qa",
        "xsum": "summ", "samsum": "summ", "cnn_dailymail": "summ"}
BROAD = {"short_qa": "qa", "long_qa": "qa", "summ": "summ"}
TEST_FRAC = 0.30       # held-out test carved from the 1800 pool (per seed)
TOTAL = 1800           # matched training budget per OOD rung (split across its sources)


def rung_sources(X):
    """Task-family rung composition for eval target X, excluding X itself (no leakage)."""
    same = [d for d in ALL if d != X and FINE[d] == FINE[X]]
    diff = [d for d in ALL if d != X and BROAD[FINE[d]] != BROAD[FINE[X]]]
    loo = [d for d in ALL if d != X]
    return {"SameTask": same, "DiffTask": diff, "LOO": loo}


def carve(n, seed, test_frac=TEST_FRAC):
    """Seeded ID train/test split of n rows -> (train_idx, test_idx)."""
    perm = np.random.RandomState(seed).permutation(n)
    n_te = int(round(n * test_frac))
    return perm[n_te:], perm[:n_te]


def sampled(n, seed, cap):
    """Up to `cap` training rows sampled from a source of `n` rows (seeded)."""
    if cap >= n:
        return np.arange(n)
    return np.random.RandomState(seed).permutation(n)[:cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | XL eval targets {EVAL_TARGETS}", flush=True)

    # load every cached dataset (states, correctness, records); split is ignored (we carve our own).
    PT = {}
    for d in ALL:
        loaded = load_per_token(MODEL, d, LAYER, LAB)
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        st, _split, y, _, rc = loaded
        if np.isnan(np.asarray(y, float)).any():
            print(f"  {d}: NaN correctness -> skip"); continue
        PT[d] = (st, np.asarray(y, float), rc)
        print(f"  {d}: {len(st)} rows cached", flush=True)
    have = set(PT)

    METHODS = ["msp_floor", "saplma", "uniform", "attention", "wMSP-unconstrained", "wMSP-normalised"]
    out_rows = []
    for X in EVAL_TARGETS:
        if X not in have:
            print(f"!! {X} not cached -> skip eval target"); continue
        st_X, y_X, rc_X = PT[X]
        nX = len(st_X)
        rungs = rung_sources(X)
        # pooler temperature: selected ONCE on the ID split of the FIRST seed, reused across rungs.
        tr0, _ = carve(nX, seeds[0])
        best_T, _ = select_temperature(st_X, y_X, list(tr0), device, seeds[0], False, False)
        print(f"\n=== eval target {X} ({FINE[X]}) | n={nX} | pooler T={best_T} ===", flush=True)
        print(f"    rungs: " + " | ".join(f"{r}={[d for d in s if d in have]}" for r, s in rungs.items()), flush=True)

        for rung in ["ID"] + list(rungs):
            per = {m: [] for m in METHODS}
            for sd in seeds:
                tr_e, te_e = carve(nX, sd)
                test_rows = [(X, int(i)) for i in te_e]
                if rung == "ID":
                    train_rows = [(X, int(i)) for i in tr_e]
                else:
                    srcs = [d for d in rungs[rung] if d in have]
                    if not srcs:
                        continue
                    cap = max(1, TOTAL // len(srcs))
                    train_rows = [(d, int(i)) for d in srcs for i in sampled(len(PT[d][0]), sd, cap)]
                if not train_rows:
                    continue
                allrows = train_rows + test_rows
                n_tr = len(train_rows)
                tr_idx = list(range(n_tr))
                te_idx = list(range(n_tr, n_tr + len(test_rows)))
                st = [PT[d][0][i] for d, i in allrows]
                y = np.array([PT[d][1][i] for d, i in allrows], float)
                rc = [PT[d][2][i] for d, i in allrows]
                yte = y_X[te_e]

                # FAIR floor (fixed 2026-07-22): best of {msp_sum, perplexity, msp_min}, not bare msp_sum.
                _fv, _fname = msp.fair_floor([rc[i] for i in te_idx], yte, results.prr)
                per["msp_floor"].append(results.prr(yte, _fv))
                Xm = np.stack([np.asarray(s).mean(axis=0) for s in st])
                per["saplma"].append(results.prr(yte, probe.uncertainty(
                    probe.train_probe_mlp(Xm[tr_idx], y[tr_idx], seed=sd), Xm[te_idx])))
                per["uniform"].append(results.prr(yte, attn_unc(
                    train_attn(st, y, tr_idx, device, seed=sd, freeze_query=True), st, te_idx, device)))
                per["attention"].append(results.prr(yte, attn_unc(
                    train_attn(st, y, tr_idx, device, seed=sd, temperature=best_T), st, te_idx, device)))
                for wm, mode in (("wMSP-unconstrained", "unconstrained"), ("wMSP-normalised", "normalised")):
                    per[wm].append(results.prr(yte, np.asarray(weighted_msp.weighted_msp_unc(
                        st, rc, y, tr_idx, te_idx, device, weight_mode=mode,
                        length_normalise=True, seed=sd), float)))

            line = f"[{X:9s} {rung:9s}] n_test={len(te_e):4d}"
            for m in METHODS:
                if per[m]:
                    mean, std = float(np.mean(per[m])), float(np.std(per[m]))
                    out_rows.append({"eval": X, "rung": rung, "method": m, "prr_mean": round(mean, 4),
                                     "prr_std": round(std, 4), "n_test": int(len(te_e))})
                    line += f"  {m} {mean:+.3f}"
            print(line, flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / f"xl_eval_ladder__{cache._slug(MODEL)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["eval", "rung", "method", "prr_mean", "prr_std", "n_test"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out}", flush=True)
    print("NOTE: same-label eval (correctness throughout) -> clean task-shift, no cross-label caveat.\n"
          "XL divergence: med_quad/samsum test splits are CARVED (seeded) from their 1800-row training pool\n"
          "(ProbeDrift ships no eval split for them); they are NOT ProbeDrift keystone evals.", flush=True)


if __name__ == "__main__":
    main()
