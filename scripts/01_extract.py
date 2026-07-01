"""GPU step: generate, then cache the Tier-1 records and Tier-2 pooled features.

Run under Slurm (see slurm/extract.sbatch). Download the model on the login node
first, because compute nodes have no internet.

    python scripts/01_extract.py --dataset sciq --ood ID
"""
import argparse
import sys
from pathlib import Path

import torch

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, data, generate  # noqa: E402
from luq.config import Config  # noqa: E402

# Map the --dtype flag to a torch dtype. "auto" -> None lets load_model pick its
# per-model default (bf16 for Gemma, fp16 otherwise).
_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                    help="model load dtype. auto = load_model's per-model default "
                         "(bf16 for Gemma, fp16 otherwise). Use fp32 to match Joe's "
                         "Hidden Failures Llama runs (he passes no torch_dtype).")
    ap.add_argument("--attn", default="auto", choices=["auto", "eager", "sdpa"],
                    help="attention backend. auto = load_model's default (eager for "
                         "Gemma, HF default otherwise). Use eager to match Joe's runs.")
    ap.add_argument("--truncate-long", action="store_true",
                    help="ALSO truncate long-form generations at the first newline (D1). "
                         "Matches Joe's generate_until=['\\n'] for every dataset; off by "
                         "default since our convention leaves long-form untruncated. Use for "
                         "the pubmed_qa keystone reproduction.")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on #examples for a quick run")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag for one ProbeDrift prompt set. Empty = the "
                         "frozen original cache; use e.g. 'pdnew' for the updated ProbeDrift "
                         "(different prompts) so the two never share records/features.")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    train_ds, eval_ds = data.load(cfg.dataset, cfg.ood_setting)
    # auto -> None so load_model keeps its per-model defaults; otherwise override.
    dtype = None if args.dtype == "auto" else _DTYPE[args.dtype]
    attn = None if args.attn == "auto" else args.attn
    model, tok = generate.load_model(cfg.model_name, attn_implementation=attn, dtype=dtype)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)

    # Guard against silently extending a cache built from different prompts (e.g. a
    # ProbeDrift change). Stamp the hash of the exact prompts+targets. If a stored hash
    # exists and differs, stop loudly instead of mixing two prompt sets in one cache.
    digest = cache.prompt_hash(list(train_ds.x) + list(eval_ds.x),
                               list(train_ds.y) + list(eval_ds.y))
    stored = cache.load_prompt_hash(cfg.cache_dir, key)
    if stored is not None and stored != digest:
        sys.exit(f"PROMPT MISMATCH for {key} in {cfg.cache_dir}:\n"
                 f"  cached prompts hash {stored}\n  current prompts hash {digest}\n"
                 f"The prompts changed under this cache. Use a fresh --prompt-regime "
                 f"or clear this namespace; do not mix prompt sets.")
    cache.save_prompt_hash(digest, cfg.cache_dir, key)

    # Few-shot short-form QA: the answer ends at the first newline; after that the
    # model just imitates the prompt format. Long-form output keeps its newlines by
    # default, UNLESS --truncate-long is set (Joe's generate_until=['\n'] for the
    # pubmed_qa keystone reproduction; see D1 in the analysis).
    truncate = (cfg.dataset in data.SHORT_FORM) or args.truncate_long

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
    budget = min(data.MAX_NEW_TOKENS[cfg.dataset], cfg.max_new_tokens_cap)
    for split, ds in [("train", train_ds), ("test", eval_ds)]:
        for idx, batch in enumerate(ds):
            # The updated ProbeDrift yields (x, y); the old one yielded (x, y, mnt). Take
            # the first two either way, and use our own per-dataset budget (data.MAX_NEW_TOKENS).
            xb, yb = batch[0], batch[1]
            if args.limit is not None and idx >= args.limit:
                break
            if (split, idx) in done:
                continue  # already cached by a previous run
            prompt, target = xb[0], yb[0]  # batch_size=1: unwrap the lists

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
