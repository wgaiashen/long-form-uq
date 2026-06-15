"""Check 2: eyeball the LLM-judge labels on pubmed_qa (read-only, CPU, login node).

Samples ~20 judged records SPREAD ACROSS the score range and prints, per record: the trimmed
Abstract+Question the judge saw, the gold answer, the model's answer, and the 0-1 correctness
score. Read them by eye -- low scores should be wrong/unfaithful answers, high scores good ones.
No agreement metric, just a sanity read of the long-form ground truth. (No OpenAI calls: the
labels are already cached.)

    python scripts/inspect_judge_labels.py --n 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from luq import cache
from luq.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--n", type=int, default=20, help="how many records to show")
    ap.add_argument("--split", default="test", help="train | test | all")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)
    if args.split != "all":
        records = [r for r in records if r["split"] == args.split]
    records = [r for r in records if r.get("correctness") is not None]
    records.sort(key=lambda r: r["correctness"])

    # Evenly-spaced picks across the sorted-by-score list => spread over the score range.
    n = min(args.n, len(records))
    picks = sorted(set(int(round(i)) for i in np.linspace(0, len(records) - 1, n)))

    print(f"{cfg.dataset} {cfg.ood_setting} | {len(picks)} of {len(records)} judged records "
          f"(split={args.split}), spread across the score range\n")
    for i in picks:
        r = records[i]
        p = r["prompt"]
        # Show the QUESTION first (it sits between the last 'Question:' and the final 'Answer:'),
        # then a short tail of the source for context -- the full abstract is long and would
        # otherwise bury the question.
        if "Question:" in p and "Answer:" in p:
            question = p[p.rfind("Question:"):p.rfind("Answer:")].strip()
        else:
            question = "(no Question marker -- summarisation task; see source)"
        source_tail = (p[:p.rfind("Question:")] if "Question:" in p else p).strip()[-500:]
        print("=" * 92)
        print(f"correctness = {r['correctness']:.2f}    (record idx {r.get('idx')})")
        print(f"\n{question}")
        print(f"\n-- source (last 500 chars, for context) --\n...{source_tail}")
        print(f"\n-- GOLD answer (what the judge compares against) --\n{str(r['target'])[:600]}")
        print(f"\n-- MODEL answer (the model's free-text answer, up to 128 tokens) --\n{r['gen_text'][:600]}")
    print("=" * 92)
    print("\nThe MODEL writes a free-text answer; the JUDGE scores it 0-1 vs the GOLD answer.")
    print("Read by eye: do LOW scores match wrong/unfaithful model answers, HIGH scores good ones?")


if __name__ == "__main__":
    main()
