"""Cheap step: train the probe from cached features + labels. Rerun freely (no GPU).

    python scripts/03_probe.py --dataset sciq --ood ID --layer 12
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, probe  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import saplma  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--layer", type=int, default=-1, help="hidden layer index to probe")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    feats = cache.load_features(cfg.cache_dir, key, method="saplma")  # (n, n_layers, hidden)

    # TODO:
    #   - Split feats + labels into train / test using each record's "split" field.
    #   - X = saplma.select_layer(feats, args.layer) for each split.
    #   - clf = probe.train_probe(X_train, y_train)
    #   - unc = probe.uncertainty(clf, X_test)
    #   - Hand (test correctness, unc) to scripts/04_eval.py (or import and score here).
    raise NotImplementedError("train + score the probe — follow the TODO above")


if __name__ == "__main__":
    main()
