"""Label the cached records for correctness.

Short-form (sciq, trivia_qa, qa): string match, no model, runs anywhere.
Long-form (pubmed_qa, xsum, cnn_dailymail): Joe's LLM judge, LOGIN NODE only.

    python scripts/02_label.py --dataset sciq --ood ID
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import cache, data  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import string_match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # TODO:
    #   if cfg.dataset in data.SHORT_FORM:
    #       for r in records: r["correctness"] = string_match.match(r["gen_text"], r["target"])
    #   else:
    #       use luq.labels.llm_judge (login node) and attach the 0-1 scores.
    #   Then re-save the records (or write a separate labels file) so 03/04 can read them.
    raise NotImplementedError("attach correctness to records — follow the TODO above")


if __name__ == "__main__":
    main()
