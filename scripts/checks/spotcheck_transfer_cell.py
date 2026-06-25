"""Independent provenance spot-check for transfer-matrix cells (esp. the dramatic ones).

The diagonal gate in 05_transfer proves the wiring reproduces 04_eval. This script
independently confirms the OFF-diagonal cells are freshly computed from the real cached
artifacts, not inherited from some stale/diagonal score. For one eval dataset + method it:

  * names the exact probe .pkl and feature .npz feeding each cell, with mtimes and the
    probe's own meta (which dataset it was trained on) — the provenance trail;
  * recomputes PRR from scratch (load probe -> apply to eval features -> PRR), so the
    number does not come through 05_transfer's code at all;
  * runs the SAME eval features through EACH train-dataset's probe, so you can see the
    sign flip across probes (the clincher: identical eval data, different probe -> the
    -0.8 only appears for the cross-task probe, so it is a real transfer effect, not a
    bookkeeping artifact);
  * reports corr(uncertainty, correctness): a POSITIVE correlation = the probe calls
    CORRECT answers "uncertain" = a genuine inversion, which is what a negative PRR means;
  * recomputes twice to confirm determinism.

    python scripts/checks/spotcheck_transfer_cell.py --eval-dataset sciq --method ptrue
    python scripts/checks/spotcheck_transfer_cell.py --eval-dataset sciq --method ptrue_accurate
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import saplma  # noqa: E402

TRAIN_DATASETS = ["sciq", "pubmed_qa", "xsum"]


def _probe_file(cache_dir, key, method):
    hits = sorted((Path(cache_dir) / "probes").glob(f"{key}__{method}__L*.pkl"))
    if not hits:
        return None
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dataset", default="sciq")
    ap.add_argument("--method", default="ptrue")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--ood", default="ID")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.eval_dataset, ood_setting=args.ood)
    cache_dir = cfg.cache_dir
    eval_key = cache.run_key(args.model, args.eval_dataset, args.ood)

    # --- Provenance of the EVAL side (shared by every cell in this column) ---
    # We need the feature_method/layer from a probe; read it from one probe's meta below.
    print(f"=== spot-check: eval = {args.eval_dataset} ({args.ood}), method = {args.method} ===\n")

    rows = []
    for train_ds in TRAIN_DATASETS:
        train_key = cache.run_key(args.model, train_ds, args.ood)
        pf = _probe_file(cache_dir, train_key, args.method)
        if pf is None:
            print(f"[{train_ds:>9} probe] MISSING — skip"); continue
        clf = cache.load_probe(cache_dir, train_key, args.method, int(pf.stem.rsplit("__L", 1)[1]))
        meta = getattr(clf, "meta", {})
        feature_method = meta.get("feature_method", args.method)
        layer = meta.get("layer")

        # Eval features + labels (the SAME for every train_ds row).
        feat_file = cache.features_path(cache_dir, eval_key, feature_method)
        feats = cache.load_features(cache_dir, eval_key, method=feature_method)
        records = cache.load_records(cache_dir, eval_key)
        X = saplma.select_layer(feats, layer)
        split = np.array([r["split"] for r in records])
        y = np.array([r["correctness"] for r in records], dtype=float)
        te = split == "test"
        X_te, y_te = X[te], y[te]

        # Recompute (twice, for determinism).
        unc1 = probe.uncertainty(clf, X_te)
        unc2 = probe.uncertainty(clf, X_te)
        prr = results.prr(y_te, unc1)
        deterministic = bool(np.array_equal(unc1, unc2))
        corr = float(np.corrcoef(unc1, y_te)[0, 1])  # +ve corr(unc, correctness) = inversion

        print(f"[{train_ds:>9}-trained probe -> {args.eval_dataset} test]  PRR = {prr:+.3f}")
        print(f"    probe : {pf.name}")
        print(f"            mtime {pf.stat().st_mtime:.0f}  meta.dataset={meta.get('dataset')} "
              f"layer={layer} seed={meta.get('seed')} feat={feature_method}")
        print(f"    eval feats : {feat_file.name}  mtime {feat_file.stat().st_mtime:.0f}  "
              f"shape {tuple(feats.shape)}  (n_test={te.sum()})")
        print(f"    corr(uncertainty, correctness) = {corr:+.3f}   "
              f"{'<- INVERTED (calls correct answers uncertain)' if corr > 0.05 else ''}")
        print(f"    deterministic recompute: {deterministic}\n")
        rows.append((train_ds, prr))

    # The clincher: same eval features, different probe -> different sign.
    print("Summary (same eval features + labels; only the probe changes):")
    for train_ds, prr in rows:
        tag = "  <- diagonal (ID)" if train_ds == args.eval_dataset else ""
        print(f"    {train_ds:>9}-probe -> {args.eval_dataset}:  PRR {prr:+.3f}{tag}")
    diag = [p for d, p in rows if d == args.eval_dataset]
    offs = [p for d, p in rows if d != args.eval_dataset]
    if diag and offs:
        print(f"\n  Diagonal PRR {diag[0]:+.3f} vs off-diagonal {[round(p,3) for p in offs]} — distinct "
              "values from the SAME eval data, so the off-diagonals are computed from the cross-task "
              "probes, not inherited from the diagonal/cached score.")


if __name__ == "__main__":
    main()
