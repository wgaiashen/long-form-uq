"""Label cached records with AlignScore (FREE, local correctness signal).

Writes record["correctness_alignscore"] (+ _model provenance), resumable + checkpointed.
the runnable pipeline evaluates long-form PRR against AlignScore (not the judge), so this
is the free, like-for-like eval label AND our Gemma/Llama cross-check. AlignScore is a
RoBERTa model -> prefers a GPU (works on CPU, just slower). No API cost.

    python scripts/01f_alignscore.py --model meta-llama/Meta-Llama-3.1-8B --dataset pubmed_qa --ood ID
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import alignscore  # noqa: E402

FIELD = "correctness_alignscore"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match the one used by 01_extract).")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    new = n_fail = 0
    for i, r in enumerate(records):
        if isinstance(r.get(FIELD), (int, float)):
            continue  # resume: skip already-scored
        s = alignscore.score(r, cfg.dataset)
        r[FIELD] = s
        new += 1
        if s is None:
            n_fail += 1
        else:
            r[FIELD + "_model"] = "yzha/AlignScore-large"
        if new % args.save_every == 0:
            cache.save_records(records, cfg.cache_dir, key)
            print(f"alignscore {new} new (at {i + 1}/{len(records)})", flush=True)

    cache.save_records(records, cfg.cache_dir, key)
    scored = [r[FIELD] for r in records if isinstance(r.get(FIELD), (int, float))]
    mean = sum(scored) / len(scored) if scored else float("nan")
    print(f"done: {len(scored)}/{len(records)} alignscore-labelled, mean {mean:.3f} (fail {n_fail})")


if __name__ == "__main__":
    main()
