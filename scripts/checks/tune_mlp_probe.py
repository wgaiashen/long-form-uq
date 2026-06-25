"""Validation-based tuning of the SAPLMA-MLP probe knobs (epochs, weight_decay).

Tunes ONLY on a validation split carved from the TRAIN split (never the test split, so no
leakage). Selects by average validation PRR across the finished ID datasets, then reports the
held-out TEST PRR for the chosen config. Run on CPU (cached features); writes a table to stdout.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import torch
from luq import cache, probe, results
from luq.features import saplma

torch.set_num_threads(max(1, torch.get_num_threads()))
M = "google/gemma-2-9b-it"
DS = [("sciq", 21), ("pubmed_qa", 21)]


def load(ds, layer):
    key = cache.run_key(M, ds, "ID")
    feats = cache.load_features("cache", key, method="saplma")
    recs = cache.load_records("cache", key)
    X = saplma.select_layer(feats, layer).astype(np.float64)
    y = np.array([r["correctness"] for r in recs], float)
    split = np.array([r["split"] for r in recs])
    return X[split == "train"], y[split == "train"], X[split == "test"], y[split == "test"]


def split_val(Xtr, ytr, frac=0.2, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    nv = int(len(Xtr) * frac)
    v, t = idx[:nv], idx[nv:]
    return Xtr[t], ytr[t], Xtr[v], ytr[v]


def corr(p, y):
    return float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-9 else float("nan")


data = {ds: load(ds, L) for ds, L in DS}
grid_ep = [100, 200, 300]
grid_wd = [1e-3, 1e-2, 1e-1, 3e-1]

print(f"{'epochs':>6} {'wd':>6} | {'sciq_val':>8} {'pub_val':>8} {'AVG_val':>8} "
      f"| {'sciq_gap':>8} {'pub_gap':>8}", flush=True)
best = None
for ep in grid_ep:
    for wd in grid_wd:
        vprr, gap = {}, {}
        for ds, _ in DS:
            Xtr, ytr, Xte, yte = data[ds]
            Xt, yt, Xv, yv = split_val(Xtr, ytr)
            clf = probe.train_probe_mlp(Xt, yt, epochs=ep, weight_decay=wd)
            vprr[ds] = results.prr(yv.tolist(), probe.uncertainty(clf, Xv).tolist())
            gap[ds] = corr(clf.p_correct(Xt), yt) - corr(clf.p_correct(Xv), yv)
        avg = float(np.mean(list(vprr.values())))
        print(f"{ep:>6} {wd:>6.3g} | {vprr['sciq']:>8.3f} {vprr['pubmed_qa']:>8.3f} "
              f"{avg:>8.3f} | {gap['sciq']:>8.2f} {gap['pubmed_qa']:>8.2f}", flush=True)
        if best is None or avg > best[0]:
            best = (avg, ep, wd)

_, ep, wd = best
print(f"\nBEST by avg val PRR: epochs={ep} weight_decay={wd} (avg_val={best[0]:.3f})", flush=True)

# Retrain on FULL train with the chosen config; report held-out TEST PRR (once).
print("\nHeld-out TEST PRR with chosen config vs current defaults (ep=500, wd=1e-3):", flush=True)
for ds, _ in DS:
    Xtr, ytr, Xte, yte = data[ds]
    tuned = probe.train_probe_mlp(Xtr, ytr, epochs=ep, weight_decay=wd)
    base = probe.train_probe_mlp(Xtr, ytr, epochs=500, weight_decay=1e-3)
    tuned_prr = results.prr(yte.tolist(), probe.uncertainty(tuned, Xte).tolist())
    base_prr = results.prr(yte.tolist(), probe.uncertainty(base, Xte).tolist())
    g_t = corr(tuned.p_correct(Xtr), ytr) - corr(tuned.p_correct(Xte), yte)
    print(f"  {ds:10s}: tuned PRR {tuned_prr:.3f} (train-test corr gap {g_t:.2f}) "
          f"| default PRR {base_prr:.3f}", flush=True)
print("DONE", flush=True)
