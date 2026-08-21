"""Stage 6: POOLED multi-task training (train on a MIXTURE, test on a held-out task).

The transfer matrix (05) asks "does a probe trained on ONE task work on another?". This
asks the complementary question: "if you train on SEVERAL tasks at once, do you get a
probe that generalises to an UNSEEN task?" By default it runs leave-one-out over the
three datasets: for each held-out dataset, pool the other two datasets' TRAIN splits into
one training set, fit one probe, and score the held-out dataset's TEST split.

Cheap by construction — it self-pools the already-cached, judge-labelled ID features
(same label semantics across all datasets after the standardisation gate), so there is no
GPU, no relabelling, and no ProbeDrift OOD re-extraction. (The protocol-faithful ProbeDrift
OOD_LEAVE_ONE_OUT route, which would need that re-extraction, is the optional later refinement.)

    python scripts/06_pool.py --model google/gemma-2-9b-it --ood ID
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import saplma  # noqa: E402

# Mirror of scripts/03_probe.py METHOD_SPEC: method -> (feature set, probe arch, hparams).
# Kept in sync by hand; if 03_probe's spec changes, update here too.
METHOD_SPEC = {
    "saplma":         ("saplma",         "mlp",    {}),
    "linear":         ("saplma",         "linear", {}),
    "ptrue":          ("ptrue",          "linear", {}),
    "ptrue_accurate": ("ptrue_accurate", "linear", {}),
    "lookback":       ("lookback",       "linear", {"standardize": False}),
}
DATASETS = ["sciq", "pubmed_qa", "xsum"]


def _split_features(cache_dir, model, dataset, ood, feature_method, layer_arg, memo):
    """Return (X_train, y_train, X_test, y_test) for a dataset at `layer_arg` (0 for the
    single-layer lookback feature). Memoised so each big all-layer feature file is loaded once.
    layer_arg MUST match the layer 03_probe/04_eval/the ID table used (Llama middle = 15, the
    ceil(N/2)-1), NOT n_layers//2 = 16: the ptrue_accurate feature is stored L15-ONLY (other layers
    NaN), so layer 16 is NaN, and saplma at 16 would not be apples-to-apples with the ID table."""
    mk = (dataset, feature_method, layer_arg)
    if mk in memo:
        return memo[mk]
    key = cache.run_key(model, dataset, ood)
    feats = cache.load_features(cache_dir, key, method=feature_method)  # (n, n_layers, hidden)
    records = cache.load_records(cache_dir, key)
    if len(records) != len(feats):
        raise SystemExit(f"{dataset}: records and features out of step — rerun 01/03.")
    layer = 0 if feats.shape[1] == 1 else layer_arg   # lookback = single combined layer -> 0
    X = saplma.select_layer(feats, layer)
    if not np.isfinite(X).all():
        raise SystemExit(f"{dataset}/{feature_method} layer {layer}: NaN/inf features -- wrong layer "
                         f"(e.g. an L15-only feature read at 16). Pass the correct --layer.")
    split = np.array([r["split"] for r in records])
    y = np.array([r["correctness"] for r in records], dtype=float)
    tr, te = split == "train", split == "test"
    res = (X[tr], y[tr], X[te], y[te])
    memo[mk] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--methods", default=",".join(METHOD_SPEC))
    ap.add_argument("--layer", type=int, default=15,
                    help="hidden layer for the multi-layer features (Llama middle = 15, matching "
                         "03_probe --layer 15 / the ID table; lookback's single layer is forced to 0)")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset="sciq", ood_setting=args.ood)
    cache_dir = cfg.cache_dir
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    memo, out_rows = {}, []

    print("Leave-one-out pooled training: train on all-but-the-held-out, score the held-out.\n")
    for method in methods:
        feature_method, arch, hparams = METHOD_SPEC[method]
        print(f"================ method: {method} ================")
        print(f"{'held-out (eval)':>16}{'pooled-PRR':>12}   (trained on)")
        for held_out in datasets:
            train_dss = [d for d in datasets if d != held_out]
            try:
                Xtr_parts, ytr_parts = [], []
                for d in train_dss:
                    Xtr, ytr, _, _ = _split_features(
                        cache_dir, args.model, d, args.ood, feature_method, args.layer, memo)
                    Xtr_parts.append(Xtr)
                    ytr_parts.append(ytr)
                X_train = np.concatenate(Xtr_parts, axis=0)
                y_train = np.concatenate(ytr_parts, axis=0)
                _, _, X_te, y_te = _split_features(
                    cache_dir, args.model, held_out, args.ood, feature_method, args.layer, memo)

                if np.ptp(y_train) < 1e-6:
                    raise SystemExit("pooled train labels are constant — nothing to learn")
                if arch == "mlp":
                    clf = probe.train_probe_mlp(X_train, y_train, **hparams)
                else:
                    clf = probe.train_probe(X_train, y_train, **hparams)
                prr = results.prr(y_te, probe.uncertainty(clf, X_te))
                print(f"{held_out:>16}{prr:>12.3f}   (train: {'+'.join(train_dss)})")
                out_rows.append({"method": method, "held_out": held_out,
                                 "train": "+".join(train_dss), "prr": round(prr, 4)})
            except Exception as e:
                print(f"{held_out:>16}{'n/a':>12}   ({type(e).__name__}: {e})")
        print()

    out_path = Path(cfg.results_dir) / f"pooled_loo__{cache._slug(args.model)}__{args.ood}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(out_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["method", "held_out", "train", "prr"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path}")
    print("Compare each pooled-PRR to (a) the held-out's ID diagonal in 04_eval/05 and "
          "(b) the best single-source column in the 05 transfer matrix: does training on a "
          "mixture beat training on one other task?")


if __name__ == "__main__":
    main()
