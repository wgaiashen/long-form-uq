"""P1.1 — does moderating / smoothing the weighted-MSP weights buy OOD robustness?

The diagnosed failure (Joe's email): the learned per-token weighting spikes on single tokens that don't
generalise. This driver sweeps the P1.1 smoothing knobs (all from luq.weighting, shared with the pooler)
and asks whether a moderated weighting keeps the ID gain while surviving distribution shift:

  baseline     plain weighted-MSP normalised (the current method; the row every config is compared to)
  shrink@L     + reg_lambda*L * shrink_to_uniform(w)  (P1.1a: pull weights toward uniform / plain MSP)
  kl@L         + reg_lambda*L * kl_to_uniform(w)      (P1.1a: entropy / KL-to-uniform moderation)
  smooth_n=k   neighbour-smooth the weights over k tokens, at train AND score time (P1.1b)

Scored on ID + LOO + DiffTask (the key contrast: keep the ID gain? survive shift?) for the 3 QA evals,
3 seeds, vs the MSP floor. A baseline ID-anchor gate (wMSP-norm sciq 0.820 / trivia 0.790 / pubmed 0.557)
fails the job loudly if the smoothing wiring changed the default path.

    python scripts/checks/weighted_msp_smoothing.py --seeds 1,2,3
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

from luq import cache, msp, results, weighted_msp, weighting  # noqa: E402
from probe_drift.ood_settings import get_training_spec  # noqa: E402
from aggregation_table import load_per_token, paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
CANDIDATES = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "med_quad", "samsum"]
# baseline weighted-MSP-norm ID anchors (master table); the baseline config must reproduce them.
ID_ANCHOR = {"sciq": 0.820, "trivia_qa": 0.790, "pubmed_qa": 0.557}
GATE_TOL = 0.04

# (name, kwargs to weighted_msp_unc beyond the shared ones)
# Finer sweep (2026-07-12 follow-up): the J4 sweep found shrink is the lever and shrink@10 too strong,
# with shrink@2 raw-improving pubmed ID (0.557->0.631) but ns. So sweep FINE around the sweet spot to see
# if a lambda in [0.5,3] gives a SIGNIFICANT pubmed ID gain, and add the two OOD rungs J4 skipped.
CONFIGS = [
    ("baseline", {}),
    ("shrink@0.5", {"reg": weighting.shrink_to_uniform, "reg_lambda": 0.5}),
    ("shrink@1", {"reg": weighting.shrink_to_uniform, "reg_lambda": 1.0}),
    ("shrink@1.5", {"reg": weighting.shrink_to_uniform, "reg_lambda": 1.5}),
    ("shrink@2", {"reg": weighting.shrink_to_uniform, "reg_lambda": 2.0}),
    ("shrink@3", {"reg": weighting.shrink_to_uniform, "reg_lambda": 3.0}),
]
SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"), ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("OneDatasetDiffTask", "OOD_ONE_DATASET_DIFF_TASK"), ("DiffTask", "OOD_DIFF_TASK")]


def sampled(split, seed, cap):
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


def cells(sources):
    out = [("ID", X, [(X, None)]) for X in EVALS]
    for X in EVALS:
        for tag, setting in SETTINGS:
            spec = [(s, n) for s, n in get_training_spec(X, setting) if s in sources and s != X]
            if spec:
                out.append((tag, X, spec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | seeds {seeds} | configs {[c[0] for c in CONFIGS]}", flush=True)

    PT = {}
    for d in CANDIDATES:
        loaded = load_per_token(MODEL, d, args.layer, LAB)
        if loaded is None:
            print(f"  {d}: no pertok -> skip"); continue
        states, split, y, _, records = loaded
        if np.isnan(y).any():
            print(f"  {d}: unlabelled -> skip"); continue
        PT[d] = (states, split, y, records)
        print(f"  {d}: {len(states)} rows", flush=True)
    sources = set(PT)

    out_rows = []
    for rung, X, spec in cells(sources):
        if X not in PT:
            continue
        spec = [(d, c) for d, c in spec if d in PT]
        if not spec:
            continue
        # floor (unsupervised, seed/config-invariant)
        te0 = np.where(PT[X][1] == "test")[0]
        yte = np.array([PT[X][2][i] for i in te0], dtype=float)
        # FAIR floor (fixed 2026-07-22): best of {msp_sum, perplexity, msp_min}, not bare msp_sum.
        floor_unc, _fname = msp.fair_floor([PT[X][3][i] for i in te0], yte, results.prr)
        floor_prr = results.prr(yte, floor_unc)

        cfg_prr = {c[0]: [] for c in CONFIGS}
        cfg_unc = {c[0]: [] for c in CONFIGS}
        for sd in seeds:
            train_rows = [(d, i) for d, cap in spec for i in sampled(PT[d][1], sd, cap)]
            test_rows = [(X, i) for i in te0]
            if not train_rows:
                continue
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            for name, kw in CONFIGS:
                u = np.asarray(weighted_msp.weighted_msp_unc(
                    states, records, y, tr_idx, te_idx, device, weight_mode="normalised",
                    length_normalise=True, seed=sd, **kw), dtype=float)
                cfg_prr[name].append(results.prr(yte, u))
                cfg_unc[name].append(u)

        print(f"\n[{rung:8s}] eval={X}  floor(msp_sum) {floor_prr:+.3f}", flush=True)
        base_avg = np.mean(np.stack(cfg_unc["baseline"]), axis=0) if cfg_unc["baseline"] else None
        for name, _ in CONFIGS:
            if not cfg_prr[name]:
                continue
            mean, std = float(np.mean(cfg_prr[name])), float(np.std(cfg_prr[name]))
            row = {"rung": rung, "eval": X, "config": name, "prr_mean": round(mean, 4),
                   "prr_std": round(std, 4), "floor": round(floor_prr, 4), "n_seeds": len(cfg_prr[name])}
            # paired bootstrap vs baseline (does smoothing move it, and vs floor)
            if name != "baseline" and base_avg is not None:
                avg = np.mean(np.stack(cfg_unc[name]), axis=0)
                mg, lo, hi, p, sig = paired_bootstrap(yte, avg, base_avg)  # + => config better than baseline
                row.update(delta_vs_baseline=round(mg, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                           boot_p=round(p, 4), significant=sig)
                print(f"    {name:11s} {mean:+.3f}±{std:.3f}  Δvs-base {mg:+.3f}[{lo:+.3f},{hi:+.3f}] "
                      f"{'SIG' if sig else 'ns'}", flush=True)
            else:
                print(f"    {name:11s} {mean:+.3f}±{std:.3f}  (baseline)", flush=True)
            out_rows.append(row)

        if rung == "ID" and X in ID_ANCHOR and cfg_prr["baseline"]:
            got = float(np.mean(cfg_prr["baseline"]))
            d_ = abs(got - ID_ANCHOR[X])
            assert d_ < GATE_TOL, f"BASELINE ID-GATE FAIL {X}: {got:.3f} vs {ID_ANCHOR[X]} (|d|={d_:.3f}) " \
                                  f"-> smoothing wiring changed the default path!"
            print(f"    [ID-GATE OK] baseline reproduces wMSP-norm anchor {ID_ANCHOR[X]} within {GATE_TOL}",
                  flush=True)

    out = Path(args.out) if args.out else (ROOT / "results" / f"weighted_msp_smoothing__{cache._slug(MODEL)}.csv")
    cols = ["rung", "eval", "config", "prr_mean", "prr_std", "floor", "n_seeds",
            "delta_vs_baseline", "ci_lo", "ci_hi", "boot_p", "significant"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
