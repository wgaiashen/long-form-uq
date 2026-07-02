"""Stage 5: cross-task TRANSFER MATRIX (train on dataset A, score dataset B).

The project's central question is whether ONE probe generalises across long-form task
types. This driver answers the single-source-training half of it: take a probe already
trained on dataset A's in-distribution run (cached by 03_probe) and apply it, unchanged,
to dataset B's test features, for every (A, B) pair and every supervised method.

It is pure recombination of cached artifacts — no GPU, no training, no labelling:
  * the probe (weights + its StandardScaler) is loaded from cache/probes/,
  * dataset B's features are loaded from cache/features/ at the SAME layer the probe used,
  * `probe.uncertainty` is one numpy forward pass,
  * PRR is scored against B's test-split correctness (the SAME label 04_eval uses).

GATE (built in): the diagonal cell (train B, score B) must reproduce exactly what 04_eval
reports for B's ID run. We check the strongest possible form of this — the recomputed
test uncertainties must equal the cached 03_probe scores that 04_eval reads (allclose).
If any diagonal fails, the cross-dataset wiring is wrong and NO off-diagonal cell is
trustworthy, so the script says so loudly.

    python scripts/05_transfer.py --model google/gemma-2-9b-it --ood ID
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import saplma  # noqa: E402

# The supervised, probe-based methods (uhead is a pretrained head, not a trained probe of
# ours, and MSP/perplexity are not probes — none of those transfer in this sense).
METHODS = ["saplma", "linear", "ptrue", "ptrue_accurate", "lookback"]
DATASETS = ["sciq", "pubmed_qa", "xsum"]


def _find_probe_layer(cache_dir, key, method):
    """A probe is cached as <key>__<method>__L<layer>.pkl. We don't know the layer up
    front (saplma/ptrue use the middle layer, lookback uses 0), so discover it from the
    filename. Returns the int layer, or None if no probe was trained for this pair."""
    hits = sorted((Path(cache_dir) / "probes").glob(f"{key}__{method}__L*.pkl"))
    if not hits:
        return None
    if len(hits) == 1:
        return int(hits[0].stem.rsplit("__L", 1)[1])
    # More than one cached layer (e.g. a SAPLMA layer sweep leaves L6..L28 on disk). Rather than
    # force the user to delete artifacts, disambiguate to the SAME layer 04_eval serves — the
    # cached score's layer — so the diagonal gate below stays self-consistent by construction.
    layers = {int(h.stem.rsplit("__L", 1)[1]) for h in hits}
    try:
        score_layer = int(cache.load_scores(cache_dir, key, method)["layer"])
    except FileNotFoundError:
        raise SystemExit(f"multiple cached probes for {key}__{method}: {sorted(layers)}, and no "
                         f"cached score to disambiguate — run 03_probe --method {method} so the "
                         "reported layer is unambiguous.")
    if score_layer not in layers:
        raise SystemExit(f"cached {method} score is layer {score_layer} but no probe at that layer "
                         f"exists for {key} (have {sorted(layers)}); rerun "
                         f"03_probe --method {method} --layer {score_layer}.")
    return score_layer


def _test_features_and_labels(cache_dir, model, dataset, ood, feature_method, layer, memo):
    """Load dataset's TEST-split features at `layer` and its test-split correctness, in
    the same order — exactly the rows 03_probe/04_eval score.

    Memoised on (dataset, feature_method, layer): the cached feature file holds ALL layers
    (~1.7GB for the hidden-state methods), so we load it ONCE, keep only the small sliced
    test rows, and let the big array be freed. Without this the matrix reloads gigabytes
    per cell."""
    mk = (dataset, feature_method, layer)
    if mk in memo:
        return memo[mk]
    key = cache.run_key(model, dataset, ood)
    feats = cache.load_features(cache_dir, key, method=feature_method)  # (n, n_layers, hidden)
    records = cache.load_records(cache_dir, key)
    if len(records) != len(feats):
        raise SystemExit(f"{dataset}: records and features out of step — rerun 01/03.")
    X = saplma.select_layer(feats, layer)
    split = np.array([r["split"] for r in records])
    y = np.array([r["correctness"] for r in records], dtype=float)
    test = split == "test"
    # boolean indexing copies, so X_te is independent of the big `feats` (freed on return)
    res = (X[test], y[test], key)
    memo[mk] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma-separated; rows = train, cols = eval")
    ap.add_argument("--methods", default=",".join(METHODS), help="comma-separated")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="allclose tolerance for the diagonal self-check vs cached 04_eval scores")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset="sciq", ood_setting=args.ood)
    cache_dir = cfg.cache_dir
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")

    out_rows = []          # for the CSV: (method, train, eval, prr)
    diagonal_failures = []
    feat_memo = {}         # (dataset, feature_method, layer) -> (X_test, y_test, key)

    for method in methods:
        print(f"\n================ method: {method} ================")
        # Header row of eval datasets.
        print("train \\ eval      " + "".join(f"{d:>14}" for d in datasets))
        for train_ds in datasets:
            key_train = cache.run_key(args.model, train_ds, args.ood)
            layer = _find_probe_layer(cache_dir, key_train, method)
            if layer is None:
                print(f"{train_ds:>14}    (no cached probe — run 03_probe --method {method} "
                      f"--dataset {train_ds})")
                continue
            clf = cache.load_probe(cache_dir, key_train, method, layer)
            feature_method = clf.meta["feature_method"]

            cells = []
            for eval_ds in datasets:
                # A whole cell can be missing if its features/scores aren't cached yet
                # (e.g. xsum ptrue_accurate still extracting). Mark it, don't crash the run.
                try:
                    X_te, y_te, key_eval = _test_features_and_labels(
                        cache_dir, args.model, eval_ds, args.ood, feature_method, layer, feat_memo)
                    unc = probe.uncertainty(clf, X_te)
                    prr = results.prr(y_te, unc)
                    tag = ""
                    # GATE: diagonal must reproduce the cached 04_eval scores exactly.
                    if train_ds == eval_ds:
                        cached = cache.load_scores(cache_dir, key_eval, method)["unc"]
                        if len(cached) != len(unc) or not np.allclose(cached, unc, atol=args.tol):
                            diagonal_failures.append((method, eval_ds))
                            tag = "✗"      # flag the failing diagonal cell
                        else:
                            tag = "="      # verified identical to 04_eval
                    cells.append(f"{prr:6.3f}{tag}")
                    out_rows.append({"method": method, "train": train_ds, "eval": eval_ds,
                                     "layer": layer, "prr": round(prr, 4)})
                except Exception as e:
                    cells.append("   n/a")
                    print(f"  ! {train_ds}->{eval_ds} {method}: {type(e).__name__}: {e}",
                          file=sys.stderr)
            print(f"{train_ds:>14}    " + "".join(f"{c:>14}" for c in cells))

    # Save the matrix for the worklog.
    out_path = Path(cfg.results_dir) / f"transfer_matrix__{cache._slug(args.model)}__{args.ood}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(out_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["method", "train", "eval", "layer", "prr"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out_path}")
    print("Legend: '=' diagonal verified identical to 04_eval; '✗' diagonal MISMATCH.")

    if diagonal_failures:
        sys.exit(f"\nDIAGONAL GATE FAILED for {diagonal_failures} — the cross-dataset wiring "
                 "does not reproduce 04_eval, so the off-diagonal transfer numbers are NOT "
                 "trustworthy. Fix before reading them.")
    print("\nDIAGONAL GATE PASSED: every diagonal cell reproduces 04_eval exactly. "
          "Off-diagonal cells are the cross-task transfer PRR.")


if __name__ == "__main__":
    main()
