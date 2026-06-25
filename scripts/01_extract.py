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

    # Few-shot short-form QA: the answer ends at the first newline; after that the
    # model just imitates the prompt format. Long-form output keeps its newlines.
    truncate = cfg.dataset in data.SHORT_FORM

    # Resume: at 7-9B a job can hit the Slurm time limit before finishing, so we
    # reload whatever a previous run already cached and skip those examples instead
    # of regenerating them. `pooled_list` holds one (n_layers, hidden) array per
    # example, in the SAME order as `records`, so Tier 1 and Tier 2 stay aligned;
    # 03_probe.py separates train/test by each record's split tag.
    records, pooled_list = cache.load_checkpoint(cfg.cache_dir, key)
    done = {(r["split"], r["idx"]) for r in records}
    if done:
        print(f"resuming: {len(done)} examples already cached", flush=True)

    # Checkpoint every CKPT_EVERY new examples. A kill loses at most this many.
    CKPT_EVERY = 200
    n_new = 0
    for split, ds in [("train", train_ds), ("test", eval_ds)]:
        for idx, (xb, yb, mnt) in enumerate(ds):
            if args.limit is not None and idx >= args.limit:
                break
            if (split, idx) in done:
                continue  # already cached by a previous run
            prompt, target = xb[0], yb[0]  # batch_size=1: unwrap the lists
            budget = min(int(mnt[0]), cfg.max_new_tokens_cap)

            record, pooled = generate.generate(model, tok, prompt, budget,
                                               truncate_at_newline=truncate)
            record |= {"idx": idx, "split": split, "target": target}
            records.append(record)
            pooled_list.append(pooled.numpy())  # float32 array, (n_layers, hidden)
            n_new += 1

            if n_new % CKPT_EVERY == 0:
                cache.save_checkpoint(records, pooled_list, cfg.cache_dir, key)
                print(f"[{split}] {idx} done (checkpointed {len(records)})", flush=True)
            elif idx % 10 == 0:
                # flush=True: Slurm buffers stdout, so unflushed prints make a
                # healthy job look hung.
                print(f"[{split}] {idx} done", flush=True)

    records_path, features_path = cache.save_checkpoint(
        records, pooled_list, cfg.cache_dir, key)
    print(f"saved {len(records)} records  ({n_new} new this run) -> {records_path}")
    print(f"saved features ({len(pooled_list)}, ...) -> {features_path}")


if __name__ == "__main__":
    main()
