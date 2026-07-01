"""SAPLMA seed variance: how much does the L15 PRR move just from the training seed?

SAPLMA (the A&M MLP) has two sources of run-to-run randomness: the weight init and the minibatch
shuffle, both driven by `seed` in train_probe_mlp. Every reported number and every +/- delta in this
project is a single-seed run (seed=1). This measures the seed noise directly: train the SAPLMA MLP at
layer 15 with N seeds on the same cached features + labels, and report PRR mean / std / min / max per
dataset. It sets the bar for what counts as a real difference (e.g. a +0.012 attention-vs-MLP gap is
only meaningful if it clears this std).

Reads the frozen cache and trains in memory, writing nothing. Pure CPU.

    python scripts/checks/saplma_seed_variance.py
    python scripts/checks/saplma_seed_variance.py --seeds 20 --datasets pubmed_qa
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (1..N)")
    ap.add_argument("--label-field", default="correctness",
                    help="eval + train label (default correctness = the canonical judge label)")
    ap.add_argument("--prompt-regime", default="", help="namespace (default = frozen old prompts)")
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))
    lf = args.label_field
    L = args.layer

    print(f"SAPLMA L{L} PRR variance over {len(seeds)} seeds, label='{lf}'\n")
    print(f"  {'dataset':12s} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'range':>7}   per-seed")
    for dataset in args.datasets.split(","):
        cfg = Config(model_name=args.model, dataset=dataset, ood_setting="ID",
                     prompt_regime=args.prompt_regime)
        key = cache.run_key(args.model, dataset, "ID")
        try:
            feat = cache.load_features(cfg.cache_dir, key, "saplma")
            records = cache.load_records(cfg.cache_dir, key)
        except FileNotFoundError:
            print(f"  {dataset:12s}   (features/records missing)")
            continue
        y = np.array([r.get(lf, np.nan) for r in records], dtype=float)
        split = np.array([r["split"] for r in records])
        tr, te = split == "train", split == "test"
        if np.isnan(y[tr]).any() or np.isnan(y[te]).any():
            print(f"  {dataset:12s}   (missing '{lf}' labels)")
            continue
        prrs = []
        for s in seeds:
            clf = probe.train_probe_mlp(feat[tr, L, :], y[tr], seed=s)
            prrs.append(results.prr(list(y[te]), probe.uncertainty(clf, feat[te, L, :])))
        a = np.array(prrs)
        perseed = " ".join(f"{v:.3f}" for v in a)
        print(f"  {dataset:12s} {a.mean():+7.3f} {a.std():7.3f} {a.min():+7.3f} {a.max():+7.3f} "
              f"{a.max() - a.min():7.3f}   {perseed}")


if __name__ == "__main__":
    main()
