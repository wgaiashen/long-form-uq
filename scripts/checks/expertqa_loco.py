"""ExpertQA Role-C: leave-one-cluster-out (LOCO) SAPLMA probe vs the MSP floor, per label.

This is the DOMAIN-SHIFT rung of the ID -> domain-shift -> task-shift severity ladder. Train the probe
on N-1 expert clusters, test on the held-out cluster -- a topic shift MILDER than a task shift. CPU only:
the SAPLMA L15 features and both judge labels are already cached (cache/expertqa_rp12), no generation.

TWO LABELS, side by side, NEVER merged into `correctness`, never mixed (both gpt-5-mini):
  factuality  covered-claims three-state judge. 292 all-uncovered rows are None (no signal -> DROPPED).
                ~64% blind spot but cleaner. 237 SEVERE-derailed rows are quarantined to 0.0 (kept: a
                derailed answer IS untrustworthy).
  consistency   whole-answer non-contradiction judge. Defined for all 2016 (recovers the 292). Bimodal /
                noisier, 0% blind spot. 262 quarantined to 0.0.
  The two cover DIFFERENT row sets and have DIFFERENT quarantine sets, so each is compared against ITS OWN
  MSP floor -- never paired to each other.

SPLIT ON CLUSTER, NOT RAW FIELD (coverage is symmetric across clusters, spread 0.042, but drifts across
fields, 0.161 -- validated in results/expertqa/label_validation.txt). 'other' has only 9 records, far too
few for a meaningful held-out PRR, so it is NEVER a test fold -- it stays in the training pool.

Per label, per held-out cluster we report SAPLMA-L15 PRR and the MSP floor PRR (present in EVERY table,
Lihu's note), plus an ID reference (a stratified random pooled split) so the ID->LOCO drop = the
domain-shift penalty. A paired test-set bootstrap gives a CI on SAPLMA-vs-floor per cluster.

    python scripts/checks/expertqa_loco.py                 # both labels, seeds 1,2,3
    python scripts/checks/expertqa_loco.py --labels factuality --seeds 1
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import saplma  # noqa: E402
from expertqa_label_validation import load_labelled_with_group  # noqa: E402
from aggregation_table import paired_bootstrap  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
# 'other' (n=9) is never a held-out fold (PRR on 9 rows is noise); it stays in every training pool.
TEST_CLUSTERS = ["professional", "humanities_social", "natural_physical", "engineering", "arts"]


def load_data():
    """Return X (n, hidden) L15 SAPLMA features and the cluster-joined records, positionally aligned."""
    cfg = Config(model_name=MODEL, dataset="expertqa", ood_setting="ID", prompt_regime="expertqa_rp12")
    key = cache.run_key(MODEL, "expertqa", "ID")
    feats = cache.load_features(cfg.cache_dir, key, "saplma")   # (n, n_layers, hidden)
    X = saplma.select_layer(feats, LAYER)                       # (n, hidden)
    recs = load_labelled_with_group()                           # positional, same order as records
    assert len(recs) == X.shape[0], f"{len(recs)} recs != {X.shape[0]} feats -- misaligned"
    return X, recs


def saplma_unc(Xtr, ytr, Xte, seeds):
    """Per-example uncertainty (1-P(correct)) on Xte, averaged over seeds; higher = more uncertain."""
    accum = np.zeros(Xte.shape[0])
    for sd in seeds:
        clf = probe.train_probe_mlp(Xtr, ytr, seed=sd)
        accum += probe.uncertainty(clf, Xte)
    return accum / len(seeds)


def floor_unc(recs_te, y_te=None):
    """The unsupervised floor from the cached token_logprobs; higher = more uncertain.

    With `y_te` this returns the FAIR floor -- the best-scoring of {msp_sum, perplexity, msp_min} -- because
    bare msp_sum is not length-normalised and is the WEAKEST of the three on all 9 datasets, so comparing a
    probe against it overstates every margin (2026-07-22). Without labels it cannot choose, so it falls back
    to msp_sum and says so rather than pretending the bar is honest."""
    if y_te is None:
        print("  WARNING: floor_unc called without labels -> falling back to bare msp_sum (NOT the fair floor)",
              flush=True)
        return np.array([msp.msp_uncertainty(r["token_logprobs"], "sum") for r in recs_te], dtype=float)
    vec, _name = msp.primary_floor(recs_te)  # PRE-REGISTERED msp_min bar (2026-07-24)
    return vec


def stratified_id_split(clusters, seed, test_frac=0.2):
    """Random pooled split, stratified by cluster so every cluster appears in both train and test."""
    rng = np.random.RandomState(seed)
    te = []
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(test_frac * len(idx))))
        te.extend(idx[:k].tolist())
    te = np.array(sorted(te))
    tr = np.setdiff1d(np.arange(len(clusters)), te)
    return tr, te


def prr_over_seeds_saplma(X, y, tr, te, seeds):
    """Mean/std SAPLMA PRR over seeds, plus the seed-averaged test uncertainty (for the bootstrap)."""
    vals = []
    for sd in seeds:
        clf = probe.train_probe_mlp(X[tr], y[tr], seed=sd)
        vals.append(results.prr(y[te], probe.uncertainty(clf, X[te])))
    unc = saplma_unc(X[tr], y[tr], X[te], seeds)
    return float(np.mean(vals)), float(np.std(vals)), unc


def run_label(X, recs, label, seeds, out_rows):
    # valid rows = numeric label (drop None / NaN)
    valid = [i for i in range(len(recs))
             if isinstance(recs[i].get(label), (int, float)) and np.isfinite(recs[i].get(label))]
    Xv = X[valid]
    yv = np.array([float(recs[i][label]) for i in valid], dtype=float)
    cl = np.array([recs[i]["cluster"] for i in valid])
    recs_v = [recs[i] for i in valid]
    n_drop = len(recs) - len(valid)
    q_field = f"{label}_quarantined"
    n_quar = sum(1 for i in valid if recs[i].get(q_field))
    print(f"\n===== {label} =====  kept {len(valid)}/{len(recs)} (dropped {n_drop} None/NaN); "
          f"{n_quar} quarantined-0.0 kept; label spread [{yv.min():.2f},{yv.max():.2f}] mean {yv.mean():.3f}",
          flush=True)

    # ---- ID reference (stratified random pooled split) ----
    id_saplma, id_floor = [], []
    for sd in seeds:
        tr, te = stratified_id_split(cl, sd)
        clf = probe.train_probe_mlp(Xv[tr], yv[tr], seed=sd)
        id_saplma.append(results.prr(yv[te], probe.uncertainty(clf, Xv[te])))
        id_floor.append(results.prr(yv[te], floor_unc([recs_v[i] for i in te], yv[te])))
    # shuffled-label control: SAPLMA on shuffled y should give PRR ~ 0 (sanity that the pipeline is honest)
    tr, te = stratified_id_split(cl, seeds[0])
    yshuf = yv[tr].copy(); np.random.RandomState(0).shuffle(yshuf)
    clf = probe.train_probe_mlp(Xv[tr], yshuf, seed=seeds[0])
    ctrl = results.prr(yv[te], probe.uncertainty(clf, Xv[te]))
    id_s_m, id_f_m = float(np.mean(id_saplma)), float(np.mean(id_floor))
    print(f"  [ID pooled] SAPLMA {id_s_m:+.3f}  MSP {id_f_m:+.3f}  "
          f"(shuffled-label SAPLMA control {ctrl:+.3f}, should be ~0)", flush=True)
    for meth, v in [("saplma", id_s_m), ("msp_floor", id_f_m)]:
        out_rows.append({"label": label, "fold": "ID", "method": meth, "prr_mean": round(v, 4),
                         "n_test": int(len(te)), "n_train": int(len(tr))})

    # ---- LOCO folds ----
    loco_s, loco_f = [], []
    for C in TEST_CLUSTERS:
        te = np.where(cl == C)[0]
        tr = np.where(cl != C)[0]
        if len(te) < 20:
            print(f"  [LOCO {C}] only {len(te)} test rows -> skip", flush=True)
            continue
        s_m, s_sd, s_unc = prr_over_seeds_saplma(Xv, yv, tr, te, seeds)
        f_unc = floor_unc([recs_v[i] for i in te], yv[te])
        f_prr = results.prr(yv[te], f_unc)
        mg, lo, hi, p, sig = paired_bootstrap(yv[te], s_unc, f_unc)   # SAPLMA vs floor, per cluster
        loco_s.append(s_m); loco_f.append(f_prr)
        print(f"  [LOCO {C:18s}] n={len(te):4d}  SAPLMA {s_m:+.3f}±{s_sd:.3f}  MSP {f_prr:+.3f}  "
              f"| SAPLMA-vs-floor {mg:+.3f} CI[{lo:+.3f},{hi:+.3f}] p={p:.3f} {'SIG' if sig else 'ns'}",
              flush=True)
        out_rows.append({"label": label, "fold": C, "method": "saplma", "prr_mean": round(s_m, 4),
                         "prr_std": round(s_sd, 4), "n_test": int(len(te)), "n_train": int(len(tr))})
        out_rows.append({"label": label, "fold": C, "method": "msp_floor", "prr_mean": round(f_prr, 4),
                         "n_test": int(len(te)), "n_train": int(len(tr))})
        out_rows.append({"label": label, "fold": C, "method": "VERDICT:saplma_vs_floor",
                         "prr_mean": round(mg, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "boot_p": round(p, 4), "significant": sig, "n_test": int(len(te))})

    # ---- domain-shift summary ----
    if loco_s:
        ls, lf = float(np.mean(loco_s)), float(np.mean(loco_f))
        print(f"  [SUMMARY {label}] ID SAPLMA {id_s_m:+.3f} -> mean-LOCO {ls:+.3f}  (drop {ls-id_s_m:+.3f}) "
              f"| ID MSP {id_f_m:+.3f} -> mean-LOCO {lf:+.3f}  (drop {lf-id_f_m:+.3f})", flush=True)
        out_rows.append({"label": label, "fold": "mean_LOCO", "method": "saplma", "prr_mean": round(ls, 4)})
        out_rows.append({"label": label, "fold": "mean_LOCO", "method": "msp_floor", "prr_mean": round(lf, 4)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["factuality", "consistency"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    X, recs = load_data()
    # verification: idx->cluster join + question-match (reuse the validation invariant)
    print(f"loaded {X.shape[0]} records, feature dim {X.shape[1]}, layer {LAYER}", flush=True)
    clusters, counts = np.unique([r["cluster"] for r in recs], return_counts=True)
    print("clusters:", dict(zip(clusters.tolist(), counts.tolist())), flush=True)

    out_rows = []
    for label in args.labels:
        stamp = {r.get(f"{label}_model") for r in recs}
        print(f"\n### label={label}  judge={stamp}", flush=True)
        run_label(X, recs, label, args.seeds, out_rows)
        out = ROOT / "results" / "expertqa" / f"loco_{label}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["label", "fold", "method", "prr_mean", "prr_std", "n_test", "n_train",
                "ci_lo", "ci_hi", "boot_p", "significant"]
        with open(out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([r for r in out_rows if r["label"] == label])
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
