"""Validation-based tuning of OUR OWN probes' regularisation (linear, ptrue).

Only our own methods are tuned. SAPLMA and lookback are reproduced BASELINES anchored to
their papers (Azaria & Mitchell / Chuang et al.) and are NEVER tuned — a faithful baseline
that overfits is still reported as-is, not "rescued" by tuning.

Tunes ONLY on a validation split carved from the TRAIN split (never the test split, so no
leakage), selects by average validation PRR across the finished ID datasets, then reports the
held-out TEST PRR for the chosen config vs the current untuned default. CPU-only (cached
features).

    python scripts/checks/tune_probe.py --method linear

Honest success criterion: with 3584-dim hidden states and 1800 train examples (p >> n) the
train correlation will NOT fall to test levels — even a linear probe interpolates the train
set. Success = max validation/test PRR and a REDUCED train-test gap, not a zero gap.
"""
import argparse, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
from luq import cache, probe, results
from luq.features import saplma

M = "google/gemma-2-9b-it"
DS = ["sciq", "pubmed_qa"]   # add "xsum" once its labels are finished

# OUR methods only -> (cached feature set, architecture). Baselines are excluded by design.
SPEC = {
    "linear": ("saplma",         "linear"),   # our linear probe on the SAPLMA hidden states
    "ptrue":  ("ptrue_accurate", "linear"),   # our P(True) probe on the new-wording features
}
BASELINE = dict(epochs=300, weight_decay=1e-3)   # current untuned train_probe default


def load(ds, feature_method):
    key = cache.run_key(M, ds, "ID")
    feats = cache.load_features("cache", key, method=feature_method)
    recs = cache.load_records("cache", key)
    layer = feats.shape[1] // 2          # mirror 03_probe's default middle layer
    X = saplma.select_layer(feats, layer).astype(np.float64)
    y = np.array([r["correctness"] for r in recs], float)
    split = np.array([r["split"] for r in recs])
    return X[split == "train"], y[split == "train"], X[split == "test"], y[split == "test"], layer


def corr(p, y):
    return float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-9 else float("nan")


def fit(Xt, yt, ep, wd):
    # our own linear probe: standardised features (standardize=True is the default)
    return probe.train_probe(Xt, yt, epochs=ep, weight_decay=wd)


def kfold_prr(X, y, ep, wd, k=5, seed=1):
    """Mean validation PRR over k folds of the TRAIN split (never touches test).

    A single 20% holdout is too noisy on small/binary datasets (sciq: 360 val, 87% positive,
    val PRR swung 0.718 vs a 0.456 test) — k-fold averages that variance out, giving a far more
    reliable hyperparameter estimate. Also returns the mean train-val correlation gap.
    """
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    prrs, gaps = [], []
    for i in range(k):
        val = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        clf = fit(X[tr], y[tr], ep, wd)
        prrs.append(results.prr(y[val].tolist(), probe.uncertainty(clf, X[val]).tolist()))
        gaps.append(corr(clf.p_correct(X[tr]), y[tr]) - corr(clf.p_correct(X[val]), y[val]))
    return float(np.mean(prrs)), float(np.mean(gaps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="linear", choices=list(SPEC))
    args = ap.parse_args()
    feature_method, _ = SPEC[args.method]

    grid_ep, grid_wd = [200, 300, 500], [1e-3, 1e-2, 1e-1, 3e-1, 1.0]
    data = {ds: load(ds, feature_method) for ds in DS}
    print(f"method={args.method} (features={feature_method}) 5-fold CV selection "
          f"layers={ {ds: data[ds][4] for ds in DS} }", flush=True)
    head = "  ".join(f"{ds}_cv" for ds in DS)
    print(f"{'epochs':>6} {'wd':>6} | {head} {'AVG':>7} | " +
          "  ".join(f"{ds}_gap" for ds in DS), flush=True)

    best = None
    for ep in grid_ep:
        for wd in grid_wd:
            vprr, gap = {}, {}
            for ds in DS:
                Xtr, ytr, Xte, yte, _ = data[ds]
                vprr[ds], gap[ds] = kfold_prr(Xtr, ytr, ep, wd)
            avg = float(np.mean([vprr[ds] for ds in DS]))
            row = "  ".join(f"{vprr[ds]:7.3f}" for ds in DS)
            grow = "  ".join(f"{gap[ds]:7.2f}" for ds in DS)
            print(f"{ep:>6} {wd:>6.3g} | {row} {avg:7.3f} | {grow}", flush=True)
            if best is None or avg > best[0]:
                best = (avg, ep, wd)

    _, ep, wd = best
    print(f"\nBEST by avg val PRR: epochs={ep} weight_decay={wd} (avg_val={best[0]:.3f})", flush=True)
    print(f"\nHeld-out TEST PRR: tuned (ep={ep}, wd={wd}) vs untuned baseline "
          f"(ep={BASELINE['epochs']}, wd={BASELINE['weight_decay']}):", flush=True)
    for ds in DS:
        Xtr, ytr, Xte, yte, _ = data[ds]
        t = fit(Xtr, ytr, ep, wd)
        b = fit(Xtr, ytr, BASELINE["epochs"], BASELINE["weight_decay"])
        tprr = results.prr(yte.tolist(), probe.uncertainty(t, Xte).tolist())
        bprr = results.prr(yte.tolist(), probe.uncertainty(b, Xte).tolist())
        g = corr(t.p_correct(Xtr), ytr) - corr(t.p_correct(Xte), yte)
        print(f"  {ds:10s}: tuned {tprr:.3f} (train-test gap {g:.2f}) | baseline {bprr:.3f}",
              flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
