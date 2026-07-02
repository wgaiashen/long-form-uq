"""Bug-hunt anchors against Joe's Table 11 (per-dataset AlignScore signal) + AlignScore liveness.

Two rerunnable checks, both from cached features/labels (no model, no cache writes):

A. AlignScore LIVENESS per dataset (item-1 housekeeping): SAPLMA trained+eval on the AlignScore label
   (self-eval), judge-eval too. Refutes the "AlignScore is a dead yardstick / PRR ~0 for any predictor"
   phrasing where it is false: pubmed self-eval is ~0.24 (concentrated but LIVE), xsum may be genuinely
   deader. Report per-dataset so the claim is checked, not asserted.

B. Table-11 ANCHORS (item-3): our restricted-pool sciq-LOO in Joe's two AlignScore configs —
   AlignScore-train/AlignScore-eval (Joe Table 11 = 0.57) and AlignScore-train/Judge-eval (Joe = 0.73).
   Short-form AlignScore is healthy, so these add two independent verification anchors on a different
   label axis. Caveats as always: restricted 3-of-9 pool, seed-averaged, ranges not cells.

    PYTHONPATH=src python scripts/checks/joe_table_anchors_check.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache, probe, results          # noqa: E402
from luq.config import Config                   # noqa: E402
from luq.features import saplma                 # noqa: E402
from probe_drift.ood_settings import get_training_spec  # noqa: E402

M = "meta-llama/Meta-Llama-3.1-8B"
L, NS = 15, 5
cd = Config(model_name=M, dataset="sciq", ood_setting="ID").cache_dir
FE = {d: (saplma.select_layer(cache.load_features(cd, cache.run_key(M, d, "ID"), "saplma"), L),
          cache.load_records(cd, cache.run_key(M, d, "ID")))
      for d in ("sciq", "trivia_qa", "pubmed_qa", "xsum")}


def rows(r, s):
    return np.array([i for i, x in enumerate(r) if x["split"] == s])


def selfeval(d, label):
    f, r = FE[d]
    tr, te = rows(r, "train"), rows(r, "test")
    y = np.array([x[label] for x in r], float)
    m = np.isfinite(y)
    tr = tr[m[tr]]; te = te[m[te]]
    clf = probe.train_probe_mlp(f[tr], y[tr], batch_size=1, seed=1)
    return results.prr(list(y[te]), list(probe.uncertainty(clf, f[te])))


def loo(E, train_label, eval_label, seed):
    kept = [(s, n) for s, n in get_training_spec(E, "OOD_LEAVE_ONE_OUT")
            if s in FE and s != E]
    Xs, ys = [], []
    for d, n in kept:
        f, r = FE[d]; tr = rows(r, "train")
        pick = tr[np.random.RandomState(seed).permutation(len(tr))[:n]]
        yy = np.array([r[i][train_label] for i in pick], float)
        Xs.append(f[pick]); ys.append(yy)
    X, y = np.vstack(Xs), np.concatenate(ys)
    keep = np.isfinite(y)
    clf = probe.train_probe_mlp(X[keep], y[keep], batch_size=1, seed=seed)
    f, r = FE[E]; te = rows(r, "test")
    yte = np.array([r[i][eval_label] for i in te], float)
    return results.prr(list(yte), list(probe.uncertainty(clf, f[te])))


print("A. AlignScore LIVENESS (SAPLMA self-eval on the AlignScore label; is it dead or just concentrated?)")
print(f"{'dataset':10s} {'align self-eval PRR':>20} {'label mean':>11} {'frac<0.05':>10}")
for d in ("sciq", "trivia_qa", "pubmed_qa", "xsum"):
    _, r = FE[d]
    a = np.array([x.get("correctness_alignscore", np.nan) for x in r], float)
    a = a[np.isfinite(a)]
    print(f"{d:10s} {selfeval(d, 'correctness_alignscore'):>20.3f} {a.mean():>11.3f} {np.mean(a < 0.05):>10.2f}")

print("\nB. Table-11 ANCHORS — restricted-pool sciq-LOO, 5-seed mean±std (Joe: align/align=0.57, align/judge=0.73):")
for tl, el, joe in [("correctness_alignscore", "correctness_alignscore", 0.57),
                    ("correctness_alignscore", "correctness", 0.73)]:
    vals = [loo("sciq", tl, el, s) for s in range(NS)]
    print(f"  train={tl.split('_')[-1]:10s} eval={el.split('_')[-1]:10s} "
          f"ours {np.mean(vals):+.3f} ± {np.std(vals):.3f}   Joe {joe:.2f}")
print("NOTE: restricted 3-of-9-source pool, seed-averaged — directional anchors on a different label "
      "axis, not cell reproductions.")
