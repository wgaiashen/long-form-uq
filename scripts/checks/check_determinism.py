"""Check 6 (optional): confirm greedy generation reproduces the cached token IDs. (GPU)

Re-generates a few cached prompts with the SAME budget + truncation 01_extract used, and asserts
the new gen_token_ids equal the cached ones. Generation is greedy (do_sample=False, no sampling,
no beams), so it should be deterministic given the same model + libraries + GPU -- which is what
makes the Tier-1 cache reproducible.

    srun --partition=t4 --gres=gpu:1 python scripts/check_determinism.py --dataset sciq --n 3

NOTE: this regenerates from the LIVE ProbeDrift prompts, so run it with --prompt-regime matching the
cache you are validating (i.e. the ProbeDrift version currently installed). It cannot re-validate a
frozen cache built with a ProbeDrift version that is no longer installed (the live prompts differ).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache, data, generate
from luq.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag; must match the installed ProbeDrift's prompts.")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = [r for r in cache.load_records(cfg.cache_dir, key) if r["split"] == "test"][:args.n]

    train_ds, eval_ds = data.load(cfg.dataset, cfg.ood_setting)
    truncate = cfg.dataset in data.SHORT_FORM
    budget = min(data.MAX_NEW_TOKENS[cfg.dataset], cfg.max_new_tokens_cap)
    model, tok = generate.load_model(cfg.model_name)

    all_match = True
    for r in records:
        i = r["idx"]
        new_rec, _ = generate.generate(model, tok, eval_ds.x[i], budget,
                                       truncate_at_newline=truncate)
        same = new_rec["gen_token_ids"] == r["gen_token_ids"]
        all_match = all_match and same
        print(f"idx {i}: match={same}  (cached len {len(r['gen_token_ids'])}, "
              f"regen len {len(new_rec['gen_token_ids'])})")

    print(f"\nall {len(records)} reproduced exactly: {all_match}")
    print("True => greedy generation is deterministic and the Tier-1 cache is reproducible.")


if __name__ == "__main__":
    main()
