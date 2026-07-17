"""ExpertQA LOCO — the AGGREGATION methods (weighted-MSP + attention pooler) beside SAPLMA/MSP.

Extends expertqa_loco.py now that the per-token L15 cache exists
(cache/expertqa_rp12/pertok/...__L15.npz). Same domain-shift design: train on N-1 expert clusters, test on
the held-out cluster, per label (faithfulness / consistency, never merged). Adds, per fold:
  saplma       probe on the cached mean-pooled SAPLMA-L15 feature      (as in expertqa_loco)
  msp_floor    -log p(sequence)                                        (unsupervised floor, every table)
  uniform      frozen-query pooler on the per-token states (= mean-pool, the no-weighting control)
  attention    learned attention pooler (temperature selected on a val slice of train)
  wMSP-norm    our weighted-MSP (learned per-token weight on the NLL), normalised

INTEGRITY (verify-don't-conclude): before scoring, confirm the per-token cache reproduces the cached SAPLMA
feature via the pipeline's PRR-match check (mean-pool-pertok PRR ~ SAPLMA-feature PRR on an ID split) -- NOT
an element-wise match (fp32 sdpa-vs-eager attention differs ~0.2 element-wise; see the cnn gate saga).

CPU only. Reuses expertqa_loco's loaders + LOCO folds.

    python scripts/checks/expertqa_loco_agg.py --labels faithfulness consistency --seeds 1 2 3
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

from luq import cache, probe, results, weighted_msp  # noqa: E402
from aggregation_table import paired_bootstrap, attn_unc  # noqa: E402
from attn_pool import train_attn, select_temperature  # noqa: E402
from expertqa_loco import load_data, floor_unc, stratified_id_split, TEST_CLUSTERS  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15


def load_pertok():
    """Per-token L15 states for ExpertQA (positionally aligned to the records/features)."""
    p = ROOT / "cache" / "expertqa_rp12" / "pertok" / f"{cache._slug(MODEL)}__expertqa__ID__L15.npz"
    if not p.exists():
        sys.exit(f"ERROR: no ExpertQA per-token cache at {p}")
    z = np.load(p, allow_pickle=True)
    return list(z["states"]), z["idx"]


def wmsp_unc(states, recs, y, tr, te, device, seed):
    return np.asarray(weighted_msp.weighted_msp_unc(
        states, recs, y, list(tr), list(te), device, weight_mode="normalised",
        length_normalise=True, seed=seed), dtype=float)


def uniform_unc(states, y, tr, te, device, seed):
    m = train_attn(states, y, list(tr), device, seed=seed, freeze_query=True)
    return np.asarray(attn_unc(m, states, list(te), device), dtype=float)


def attention_unc(states, y, tr, te, device, seed):
    best_T, _ = select_temperature(states, y, list(tr), device, seed, False, False)
    m = train_attn(states, y, list(tr), device, seed=seed, temperature=best_T)
    return np.asarray(attn_unc(m, states, list(te), device), dtype=float)


def prr_seeds(fn, seeds):
    """Average PRR over seeds + the seed-averaged uncertainty (for the bootstrap). fn(seed)->(prr, unc)."""
    prrs, uncs = [], []
    for sd in seeds:
        p, u = fn(sd)
        prrs.append(p); uncs.append(u)
    return float(np.mean(prrs)), float(np.std(prrs)), np.mean(np.stack(uncs), axis=0)


def run_label(X, states, recs, label, seeds, out_rows, device):
    valid = [i for i in range(len(recs))
             if isinstance(recs[i].get(label), (int, float)) and np.isfinite(recs[i].get(label))]
    Xv = X[valid]
    stv = [states[i] for i in valid]
    yv = np.array([float(recs[i][label]) for i in valid], dtype=float)
    cl = np.array([recs[i]["cluster"] for i in valid])
    recs_v = [recs[i] for i in valid]
    print(f"\n===== {label} =====  kept {len(valid)}/{len(recs)}  label mean {yv.mean():.3f}", flush=True)

    # ---- integrity: mean-pool-pertok PRR vs cached-SAPLMA PRR on one ID split (PRR-match, not element-wise) ----
    tr0, te0 = stratified_id_split(cl, seeds[0])
    Xmean = np.stack([np.asarray(s).mean(axis=0) for s in stv])
    prr_pt = results.prr(yv[te0], probe.uncertainty(probe.train_probe_mlp(Xmean[tr0], yv[tr0], seed=seeds[0]), Xmean[te0]))
    prr_sap = results.prr(yv[te0], probe.uncertainty(probe.train_probe_mlp(Xv[tr0], yv[tr0], seed=seeds[0]), Xv[te0]))
    print(f"  [integrity] mean-pool-pertok PRR {prr_pt:+.4f} vs cached-SAPLMA PRR {prr_sap:+.4f} "
          f"(diff {abs(prr_pt-prr_sap):.4f})", flush=True)
    if abs(prr_pt - prr_sap) > 0.05:
        print(f"  [WARN] pertok vs SAPLMA PRR diverge by {abs(prr_pt-prr_sap):.3f} -- check the per-token cache", flush=True)

    METHODS = [
        ("saplma", lambda tr, te, sd: (results.prr(yv[te], probe.uncertainty(probe.train_probe_mlp(Xv[tr], yv[tr], seed=sd), Xv[te])),
                                       probe.uncertainty(probe.train_probe_mlp(Xv[tr], yv[tr], seed=sd), Xv[te]))),
        ("uniform", lambda tr, te, sd: (lambda u: (results.prr(yv[te], u), u))(uniform_unc(stv, yv, tr, te, device, sd))),
        ("attention", lambda tr, te, sd: (lambda u: (results.prr(yv[te], u), u))(attention_unc(stv, yv, tr, te, device, sd))),
        ("wMSP-norm", lambda tr, te, sd: (lambda u: (results.prr(yv[te], u), u))(wmsp_unc(stv, recs_v, yv, tr, te, device, sd))),
    ]

    folds = [("ID", None)] + [(C, C) for C in TEST_CLUSTERS]
    loco_acc = {m: [] for m, _ in METHODS}
    id_prr = {}
    for fold_name, C in folds:
        if C is None:
            splits = [stratified_id_split(cl, sd) for sd in seeds]  # per-seed ID splits
        else:
            te = np.where(cl == C)[0]; tr = np.where(cl != C)[0]
            if len(te) < 20:
                print(f"  [LOCO {C}] {len(te)} test rows -> skip", flush=True); continue
            splits = [(tr, te)] * len(seeds)  # fixed fold, seeds vary the probe only

        # floor (unsupervised, per fold uses the fold's test rows; for ID average over the per-seed splits)
        floor_prrs = [results.prr(yv[te], floor_unc([recs_v[i] for i in te])) for (_, te) in splits]
        f_prr = float(np.mean(floor_prrs))
        out_rows.append({"label": label, "fold": fold_name, "method": "msp_floor", "prr_mean": round(f_prr, 4),
                         "n_test": int(len(splits[0][1]))})

        line = f"  [{fold_name:18s}] n_te={len(splits[0][1]):4d}  floor {f_prr:+.3f}"
        for mname, fn in METHODS:
            prrs, uncs = [], []
            for (tr, te), sd in zip(splits, seeds):
                p, u = fn(tr, te, sd)
                prrs.append(p); uncs.append(u)
            m_mean, m_std = float(np.mean(prrs)), float(np.std(prrs))
            out_rows.append({"label": label, "fold": fold_name, "method": mname, "prr_mean": round(m_mean, 4),
                             "prr_std": round(m_std, 4), "n_test": int(len(splits[0][1]))})
            line += f"  {mname} {m_mean:+.3f}"
            if C is not None:
                loco_acc[mname].append(m_mean)
            else:
                id_prr[mname] = m_mean
        print(line, flush=True)

    # domain-shift drop summary (ID -> mean-LOCO) per method
    for m, _ in METHODS:
        if loco_acc[m] and m in id_prr:
            ls = float(np.mean(loco_acc[m]))
            out_rows.append({"label": label, "fold": "mean_LOCO", "method": m, "prr_mean": round(ls, 4)})
            print(f"  [SUMMARY {m:10s}] ID {id_prr[m]:+.3f} -> mean-LOCO {ls:+.3f} (drop {ls-id_prr[m]:+.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["faithfulness", "consistency"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, recs = load_data()
    states, idx = load_pertok()
    assert len(states) == len(recs) == X.shape[0], f"misaligned: {len(states)} states, {len(recs)} recs, {X.shape[0]} feats"
    print(f"loaded {X.shape[0]} records | pertok states {len(states)} | device {device}", flush=True)

    out_rows = []
    for label in args.labels:
        run_label(X, states, recs, label, args.seeds, out_rows, device)
        out = ROOT / "results" / "expertqa" / f"loco_agg_{label}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["label", "fold", "method", "prr_mean", "prr_std", "n_test"]
        with open(out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([r for r in out_rows if r["label"] == label])
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
