"""ONE-HARNESS OOD grid: every method x {ID, LOO, DiffTask} through Joe's PINNED ProbeDrift-light
pools, seed-averaged, PAIRED seeds, ID-diagonal-gated.

This retires the two ad-hoc grids (aggregation_ood_light = wrong 2-source pool; the transfer matrix
= single-source). Here ALL methods run in ONE harness on ONE OOD definition:
  * Pools = probe_drift.ood_settings.get_training_spec(eval, setting), restricted to the sources we
    have Llama features for {sciq, trivia_qa, pubmed_qa, xsum}, at Joe's per-source caps
    (LOO 200/source, DiffTask 600/source). ID = train on eval's own full train split.
  * PAIRED seeds: for each seed we draw the pool subsample ONCE and feed the SAME sampled examples
    to every method (poolers read per-token states; baselines read pooled vectors of the same rows).
    So the cross-method comparison is paired, closing the seed-protocol caveat.
  * ID-diagonal GATE: the ID cell must reproduce the ID anchors (aggregation table for mean-pool /
    attention; 04_eval for the baselines) within tol, else the wiring is wrong and the run aborts.

Methods: aggregation family (mean-pool+MLP, last-token, per-sentence #5, per-token #4, uniform,
attention) + supervised baselines (linear, ptrue_accurate, lookback). Judge label throughout.

    python scripts/checks/ood_onegrid.py --seeds 1,2,3,4,5     (GPU: per-token + attention training)
"""
import argparse
import os  # atomic replace in _flush_rows
import sys
import csv as _csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from luq import cache, probe, results          # noqa: E402
from luq.config import Config                   # noqa: E402
from luq.features import saplma as saplma_feat  # noqa: E402
from probe_drift.ood_settings import get_training_spec  # noqa: E402
from aggregation_table import (                 # noqa: E402
    load_per_token, build_arrays, conf_meanpool, conf_lasttoken, conf_persentence,
    conf_pertoken, attn_unc)
from attn_pool import train_attn, select_temperature  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAB = "correctness"
AVAIL = ("sciq", "trivia_qa", "pubmed_qa", "xsum")
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
# supervised baselines: name -> (pooled feature_method, layer, standardize)
BASE = {"linear": ("saplma", 15, True), "ptrue_accurate": ("ptrue_accurate", 15, True),
        "lookback": ("lookback", 0, False)}
POOLERS = ["mean-pool+MLP", "last-token", "per-sentence", "per-token", "uniform", "attention"]
# ID anchors (judge) the ID cells must reproduce.
ID_ANCHOR = {"sciq": {"mean-pool+MLP": 0.907, "attention": 0.932, "linear": 0.610,
                      "ptrue_accurate": 0.727, "lookback": 0.809},
             "trivia_qa": {"mean-pool+MLP": 0.807, "attention": 0.844, "linear": 0.732,
                           "ptrue_accurate": 0.420, "lookback": 0.631},
             "pubmed_qa": {"mean-pool+MLP": 0.713, "attention": 0.735, "linear": 0.550,
                           "ptrue_accurate": 0.532, "lookback": 0.600}}
GATE_TOL = 0.03


def cells():
    out = []
    for X in EVALS:
        out.append(("ID", X, [(X, None)]))                     # full train split
        for setting in ("OOD_LEAVE_ONE_OUT", "OOD_DIFF_TASK"):
            spec = [(s, n) for s, n in get_training_spec(X, setting) if s in AVAIL and s != X]
            out.append(("LOO" if "LEAVE" in setting else "DiffTask", X, spec))
    return out


def sampled_train_idx(split, seed, cap):
    """Indices of the train split, subsampled to `cap` with `seed` (cap=None -> all)."""
    tr = np.where(split == "train")[0]
    if cap is None or cap >= len(tr):
        return tr
    return tr[np.random.RandomState(seed).permutation(len(tr))[:cap]]


CSV_FIELDS = ["setting", "eval", "method", "prr_mean", "prr_std", "n_seeds"]


def _flush_rows(out, rows, fields):
    """Write everything accumulated SO FAR, atomically (temp + os.replace).

    CALLED PER CELL, NOT ONCE AT THE END (added 2026-08-03). This driver used to hold every row in
    memory and write the CSV only after the last cell, so a run killed at hour 15 of 16 -- walltime,
    OOM, or a node problem -- lost EVERYTHING and left nothing on disk saying which cells had already
    succeeded. A job was SIGTERM'd on cx3-14-9 the same day, and these ladders run 8-16 hours.

    temp-then-replace so a crash mid-write cannot leave a TRUNCATED csv, which would read as a
    short-but-valid grid. A half-written table is worse than no table: it looks complete.
    """
    tmp = Path(str(out) + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # Resolved BEFORE the loop so the crash-safety flush has a target.
    out_path = Path(args.out) if args.out else (
        ROOT / "results" / f"ood_onegrid__{cache._slug(MODEL)}.csv")
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    cd = Config(model_name=MODEL, dataset="sciq", ood_setting="ID").cache_dir
    print(f"device {device} | seeds {seeds} | PINNED pools | paired | one harness", flush=True)

    # per-token states (poolers) + pooled vectors (baselines), per dataset, positionally aligned.
    PT, POOLED = {}, {}
    for d in AVAIL:
        st, split, y, _, records = load_per_token(MODEL, d, args.layer, LAB)
        PT[d] = (st, split, y, records)
        key = cache.run_key(MODEL, d, "ID")
        POOLED[d] = {}
        for bm, (fm, layer, _std) in BASE.items():
            arr = cache.load_features(cd, key, fm)
            POOLED[d][bm] = np.ascontiguousarray(arr[:, layer, :]); del arr
        print(f"  loaded {d}: {len(st)} rows", flush=True)

    out_rows = []
    for setting, X, spec in cells():
        per_method = {m: [] for m in POOLERS + list(BASE)}
        for sd in seeds:
            # --- one shared subsample for THIS seed, used by every method (paired) ---
            train_rows = []   # (dataset, idx)
            for d, cap in spec:
                for i in sampled_train_idx(PT[d][1], sd, cap):
                    train_rows.append((d, i))
            test_rows = [(X, i) for i in np.where(PT[X][1] == "test")[0]]
            n_tr = len(train_rows)
            tr_idx, te_idx = list(range(n_tr)), list(range(n_tr, n_tr + len(test_rows)))
            allrows = train_rows + test_rows
            y = np.array([PT[d][2][i] for d, i in allrows], dtype=float)
            yte = [y[i] for i in te_idx]

            # poolers (per-token)
            states = [PT[d][0][i] for d, i in allrows]
            records = [PT[d][3][i] for d, i in allrows]
            Xmean, Xlast, sent_vecs = build_arrays(states, records, tok)
            best_T, _ = select_temperature(states, y, tr_idx, device, sd, False, False)
            res = {}
            res["mean-pool+MLP"] = results.prr(yte, 1.0 - conf_meanpool(Xmean, tr_idx, te_idx, y, sd))
            res["last-token"] = results.prr(yte, [1 - c for c in conf_lasttoken(Xlast, tr_idx, te_idx, y, sd)])
            res["per-sentence"] = results.prr(yte, [1 - c for c in conf_persentence(sent_vecs, tr_idx, te_idx, y, sd)])
            res["per-token"] = results.prr(yte, [1 - c for c in conf_pertoken(states, tr_idx, te_idx, y, sd)])
            res["uniform"] = results.prr(yte, attn_unc(train_attn(states, y, tr_idx, device, seed=sd, freeze_query=True), states, te_idx, device))
            res["attention"] = results.prr(yte, attn_unc(train_attn(states, y, tr_idx, device, seed=sd, temperature=best_T), states, te_idx, device))

            # baselines (pooled vectors, SAME sampled rows -> paired)
            for bm, (fm, layer, std) in BASE.items():
                Xtr = np.vstack([POOLED[d][bm][i] for d, i in train_rows])
                Xte = np.vstack([POOLED[d][bm][i] for d, i in test_rows])
                clf = probe.train_probe(Xtr, y[tr_idx], standardize=std, seed=sd)
                res[bm] = results.prr(yte, list(probe.uncertainty(clf, Xte)))

            for m, v in res.items():
                per_method[m].append(v)

        stats = {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_method.items()}
        print(f"\n[{setting:9s}] eval={X}", flush=True)
        for m in POOLERS + list(BASE):
            mu, sd_ = stats[m]
            print(f"    {m:16s} {mu:+.3f} ± {sd_:.3f}", flush=True)
        if setting == "ID":
            # HARD gate on the poolers (cross-dataset wiring check); WARN on baselines.
            for m in ("mean-pool+MLP", "attention"):
                d = abs(stats[m][0] - ID_ANCHOR[X][m])
                assert d < GATE_TOL, f"ID-GATE FAIL {X}/{m}: {stats[m][0]:.3f} vs anchor {ID_ANCHOR[X][m]} (|d|={d:.3f})"
            for m in BASE:
                d = abs(stats[m][0] - ID_ANCHOR[X][m])
                if d >= GATE_TOL:
                    print(f"    [ID-WARN] {m}: {stats[m][0]:.3f} vs anchor {ID_ANCHOR[X][m]} (|d|={d:.3f})", flush=True)
            print(f"    [ID-GATE OK] poolers reproduce anchors within {GATE_TOL}", flush=True)
        for m in POOLERS + list(BASE):
            out_rows.append({"setting": setting, "eval": X, "method": m,
                             "prr_mean": round(stats[m][0], 4), "prr_std": round(stats[m][1], 4),
                             "n_seeds": len(seeds)})

        # CRASH SAFETY: land this cell before the next one starts.
        _flush_rows(out_path, out_rows, CSV_FIELDS)
        print(f"    [saved] {len(out_rows)} rows -> {out_path.name}", flush=True)

    out = out_path
    _flush_rows(out, out_rows, CSV_FIELDS)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
