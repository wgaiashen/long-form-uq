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
from luq.labels import llm_judge, string_match  # noqa: E402


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
        # Long-form: Joe's GPT-5 judge. LOGIN NODE only; needs OPENAI_API_KEY; each
        # call costs money. Resumable: skip records already scored, and checkpoint to
        # disk every few, so a crash or rate-limit partway through never re-spends on
        # work already done (just rerun this command to pick up where it stopped).
        save_every = 25
        new_count = n_failed = 0
        for i, r in enumerate(records):
            if r.get("correctness") is not None:
                continue  # already judged on a previous (possibly partial) run
            r["correctness"] = llm_judge.judge(r, cfg.dataset)
            new_count += 1
            if r["correctness"] is None:
                n_failed += 1
            if new_count % save_every == 0:
                cache.save_records(records, cfg.cache_dir, key)  # checkpoint
                print(f"judged {new_count} new (at {i + 1}/{len(records)})", flush=True)
        if n_failed:
            print(f"WARNING: {n_failed} records returned no valid score (None)")

    # Re-save the records in place: correctness becomes part of the Tier-1 record,
    # so stages 03/04 read one file and labels can never drift out of line with
    # their examples. Relabelling is cheap for string match; for the judge this is
    # the final checkpoint that captures the tail since the last periodic save.
    cache.save_records(records, cfg.cache_dir, key)

    scores = [r["correctness"] for r in records if r["correctness"] is not None]
    mean = sum(scores) / len(scores) if scores else float("nan")
    print(f"labelled {len(scores)}/{len(records)} records: mean correctness {mean:.3f}")


if __name__ == "__main__":
    main()
