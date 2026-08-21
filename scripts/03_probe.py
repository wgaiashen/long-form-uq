"""Cheap step: train the probe from cached features + labels. Rerun freely (no GPU).

    python scripts/03_probe.py --dataset sciq --ood ID --layer 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, probe  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import saplma  # noqa: E402


# Each reported supervised method is a (cached feature set, probe architecture, hparams) triple.
# hparams are passed straight into the train fn, so each method carries its own recipe.
#
# BASELINES (anchored to their papers, NOT tuned by us — a faithful baseline that overfits is
# still reported as-is):
#   saplma   = mean-pooled hidden states + the 4-layer 256/128/64 MLP. Faithful Azaria & Mitchell
#              (train_probe_mlp's defaults ARE the A&M recipe: 5 epochs, batch 32, no wd, raw),
#              so hparams = {} (use those defaults).
#   lookback = lookback-ratio features + logistic regression, RAW features (standardize=False),
#              default-strength L2 — Chuang et al.'s probe. Not tuned.
#
# OUR OWN methods (standardised features, single linear logit = logistic regression):
#   linear         = the SAME mean-pooled hidden states as SAPLMA — architecture ablation vs the MLP.
#   ptrue          = P(True) verdict-position state, OLD "Is the above answer true?" wording.
#   ptrue_accurate = same probe, NEW task-agnostic "Is the above response accurate?" wording.
# ptrue and ptrue_accurate are kept as SEPARATE methods on purpose so the old vs new wording can be
# compared head-to-head per dataset (each reads its own cached feature set: 'ptrue' vs 'ptrue_accurate').
# We tried to validation-tune the linear/ptrue probes (scripts/checks/tune_probe.py, 5-fold CV) but in
# this p>>n regime (3584 dims, 1800 examples) the CV PRR does NOT transfer to test — every "tuned"
# config just trades sciq for pubmed and none beats the standard default. So we keep the standard
# regularized logistic-regression default (hparams={}) and report the train/test gap honestly. (PCA-
# before-probe is the principled p>>n lever if a real improvement is wanted later — the "Linear+PCA".)
# Score files are named by the reported method, so 04_eval prints them directly.
METHOD_SPEC = {
    "saplma":         ("saplma",         "mlp",    {}),                     # faithful A&M (5 ep / batch 32 / no wd / raw)
    "linear":         ("saplma",         "linear", {}),                     # standard logistic regression
    "ptrue":          ("ptrue",          "linear", {}),                     # P(True), OLD "is this true?" wording
    "ptrue_accurate": ("ptrue_accurate", "linear", {}),                     # P(True), NEW "is this accurate?" wording
    "lookback":       ("lookback",       "linear", {"standardize": False}), # faithful Chuang (raw features, default L2)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--method", default="saplma", choices=list(METHOD_SPEC),
                    help="supervised method: saplma (A&M MLP) | linear (linear probe on the "
                         "same hidden states) | ptrue | lookback")
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden layer index to probe (default: the middle layer). "
                         "NOTE: index 0 = embeddings, so for Llama-3.1-8B the middle "
                         "is 15 (ceil(32/2)-1); our default n_layers//2 is 16 — pass "
                         "--layer 15 to match Hidden Failures Table 14.")
    ap.add_argument("--saplma-batch", type=int, default=None,
                    help="override the SAPLMA MLP batch size (default 32). Pass 1 to match "
                         "Hidden Failures Table 14 (full_seq_head_saplma.py fits batch_size=1). Only "
                         "affects --method saplma.")
    ap.add_argument("--label-field", default="correctness",
                    help="which correctness field to TRAIN the probe on (e.g. "
                         "correctness_alignscore for AlignScore-trained, or correctness for the "
                         "judge). Pair with 04_eval --label-field for the train x eval matrix.")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    feature_method, arch, hparams = METHOD_SPEC[args.method]
    # Copy so we never mutate the shared METHOD_SPEC dict, then apply per-run overrides.
    hparams = dict(hparams)
    if args.method == "saplma" and args.saplma_batch is not None:
        hparams["batch_size"] = args.saplma_batch

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    # saplma and linear share the same cached hidden-state features (feature_method).
    feats = cache.load_features(cfg.cache_dir, key, method=feature_method)  # (n, n_layers, hidden)
    records = cache.load_records(cfg.cache_dir, key)
    assert len(records) == len(feats), "records and features are out of step — rerun 01"

    # Default to the middle layer: usually more informative than the last one.
    n_layers = feats.shape[1]
    layer = args.layer if args.layer is not None else n_layers // 2

    # FAIL LOUDLY on a configuration this feature set cannot produce, instead of letting
    # select_layer raise a raw IndexError mid-loop and a stale score linger downstream.
    # (This is the lookback `--layer 21` footgun: lookback is a single combined layer.)
    if not 0 <= layer < n_layers:
        sys.exit(f"ERROR: --layer {layer} is out of range for method '{args.method}' "
                 f"(features '{feature_method}' have {n_layers} layer(s): 0..{n_layers - 1}). "
                 f"{'lookback is a single combined layer -> use --layer 0. ' if n_layers == 1 else ''}"
                 f"Nothing was written; the previous score for this method is left untouched "
                 f"and must NOT be treated as this configuration's result.")

    # Records and features share index order, so boolean masks built from the
    # records' "split" tags select the matching feature rows.
    y = np.array([r[args.label_field] for r in records])
    split = np.array([r["split"] for r in records])
    train_mask, test_mask = split == "train", split == "test"

    # The probe trains on the graded label directly (no 0.5 threshold), so the only
    # degenerate case is a train split with no variation to learn from.
    if np.ptp(y[train_mask]) < 1e-6:
        sys.exit("all train labels are identical — the probe has nothing to learn "
                 "(tiny --limit run, or a labelling bug)")

    X = saplma.select_layer(feats, layer)  # (n, hidden)
    # Same soft-label objective either way; the method's spec fixes architecture + hparams.
    if arch == "mlp":
        clf = probe.train_probe_mlp(X[train_mask], y[train_mask], **hparams)
    else:
        clf = probe.train_probe(X[train_mask], y[train_mask], **hparams)
    unc = probe.uncertainty(clf, X[test_mask])

    # Quick diagnostic: correlation between predicted P(correct) and the soft label.
    # A big train/test gap means the probe memorised rather than learned. (PRR in
    # stage 04 is the real score; this is just an at-a-glance fit/overfit check.)
    def _corr(p, t):
        if np.std(p) < 1e-9 or np.std(t) < 1e-9:
            return float("nan")  # no variation -> correlation undefined
        return float(np.corrcoef(p, t)[0, 1])
    corr_tr = _corr(clf.p_correct(X[train_mask]), y[train_mask])
    corr_te = _corr(clf.p_correct(X[test_mask]), y[test_mask])
    print(f"{args.method} layer {layer} [{arch}]: P(correct) vs soft-label corr "
          f"train {corr_tr:.3f} | test {corr_te:.3f}")

    # Scores are named by the reported method (saplma | linear | ptrue | lookback), so
    # 04_eval lists them directly and saplma vs linear sit side by side.
    # Provenance stamp: record WHICH feature file (and its mtime) produced these scores, so
    # 04_eval can refuse a score whose features have since been regenerated (stale) instead of
    # serving it silently.
    # Stamp BOTH provenance signals: the feature file (to catch re-extraction) AND the records
    # file (to catch a RELABEL — the probe was trained against this label, so if correctness
    # changes later the probe is stale even though its features didn't move).
    feat_file = cache.features_path(cfg.cache_dir, key, feature_method)
    rec_file = cache.records_path(cfg.cache_dir, key)
    path = cache.save_scores(unc, cfg.cache_dir, key, method=args.method, layer=layer,
                             feat_method=feature_method, feat_mtime=feat_file.stat().st_mtime,
                             rec_mtime=rec_file.stat().st_mtime)
    print(f"saved {len(unc)} test uncertainties -> {path}")

    # Persist the trained probe (with provenance) so it can be re-applied to another
    # dataset's features without retraining — this is what the OOD cross-task matrix needs.
    clf.meta = {"method": args.method, "feature_method": feature_method, "arch": arch,
                "dataset": cfg.dataset, "ood": cfg.ood_setting, "layer": layer,
                "hparams": hparams, "seed": 1, "label_field": args.label_field}
    ppath = cache.save_probe(clf, cfg.cache_dir, key, method=args.method, layer=layer)
    print(f"saved trained probe -> {ppath}")


if __name__ == "__main__":
    main()
