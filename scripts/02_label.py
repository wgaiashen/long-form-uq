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

    if cfg.dataset in data.SHORT_FORM:
        # String match: does the gold answer (or any alias) appear in the output?
        for r in records:
            r["correctness"] = string_match.match(r["gen_text"], r["target"])
    else:
        # Long-form: Joe's LLM judge (login node only; needs OPENAI_API_KEY).
        raise NotImplementedError(
            "long-form labelling comes with luq.labels.llm_judge — fill that first"
        )

    # Re-save the records in place: correctness becomes part of the Tier-1 record,
    # so stages 03/04 read one file and labels can never drift out of line with
    # their examples. Relabelling is cheap (no GPU), so overwriting is safe.
    cache.save_records(records, cfg.cache_dir, key)

    scores = [r["correctness"] for r in records]
    print(f"labelled {len(records)} records: mean correctness {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
