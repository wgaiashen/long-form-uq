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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--method", default="saplma",
                    help="which cached feature set to probe: saplma | ptrue")
    ap.add_argument("--layer", type=int, default=None,
                    help="hidden layer index to probe (default: the middle layer)")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    feats = cache.load_features(cfg.cache_dir, key, method=args.method)  # (n, n_layers, hidden)
    records = cache.load_records(cfg.cache_dir, key)
    assert len(records) == len(feats), "records and features are out of step — rerun 01"

    # Default to the middle layer: usually more informative than the last one.
    layer = args.layer if args.layer is not None else feats.shape[1] // 2

    # Records and features share index order, so boolean masks built from the
    # records' "split" tags select the matching feature rows.
    y = np.array([r["correctness"] for r in records])
    split = np.array([r["split"] for r in records])
    train_mask, test_mask = split == "train", split == "test"

    # The probe trains on the graded label directly (no 0.5 threshold), so the only
    # degenerate case is a train split with no variation to learn from.
    if np.ptp(y[train_mask]) < 1e-6:
        sys.exit("all train labels are identical — the probe has nothing to learn "
                 "(tiny --limit run, or a labelling bug)")

    X = saplma.select_layer(feats, layer)  # (n, hidden)
    clf = probe.train_probe(X[train_mask], y[train_mask])
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
    print(f"layer {layer}: P(correct) vs soft-label corr "
          f"train {corr_tr:.3f} | test {corr_te:.3f}")

    path = cache.save_scores(unc, cfg.cache_dir, key, method=args.method, layer=layer)
    print(f"saved {len(unc)} test uncertainties -> {path}")


if __name__ == "__main__":
    main()
