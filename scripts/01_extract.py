"""GPU step: generate, then cache the Tier-1 records and Tier-2 pooled features.

Run under Slurm (see slurm/extract.sbatch). Download the model on the login node
first, because compute nodes have no internet.

    python scripts/01_extract.py --dataset sciq --ood ID
"""
import argparse
import sys
from pathlib import Path

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, data, generate  # noqa: E402
from luq.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on #examples for a quick run")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    train_ds, eval_ds = data.load(cfg.dataset, cfg.ood_setting, cfg.seed)
    model, tok = generate.load_model(cfg.model_name)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)

    # TODO: the extract loop.
    #   for split, ds in [("train", train_ds), ("test", eval_ds)]:
    #       for idx, (xb, yb, mnt) in enumerate(ds):
    #           prompt, target, mnt = xb[0], yb[0], mnt[0]   # batch_size=1
    #           record, pooled = generate.generate(model, tok, prompt, mnt)
    #           record |= {"idx": idx, "split": split, "target": target}
    #           ...collect record into records[], pooled into a list...
    #   stack pooled -> (n, n_layers, hidden); then:
    #   cache.save_records(records, cfg.cache_dir, key)
    #   cache.save_features(pooled_array, cfg.cache_dir, key, method="saplma")
    raise NotImplementedError("wire up the extract loop — follow the TODO above")


if __name__ == "__main__":
    main()
