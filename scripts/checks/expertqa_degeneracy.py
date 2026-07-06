"""ExpertQA generation-quality (degeneracy) scan — the Stage-2 go/no-go check.

This measures whether the base-Llama generations loop/repeat, WITHOUT needing the judge
labels (it reads only the cached generations). It reproduces the ad-hoc scan from the
2026-07-05 worklog entries so the full-set can be compared to the pilot on the same yardstick.

Degeneracy modes (EXPERTQA_PLAN Stage-2, item 3):
  - REPETITION: the base model, with no natural stop on open-ended prompts, runs to the token
    budget and loops. Measured by distinct-n (unique n-grams / total) and by any sentence
    repeated >= 3x.
  - CAP-COUPLING: repetition is tightly coupled to hitting the max_new_tokens cap, so we also
    report P(degenerate | capped) vs P(degenerate | not capped) — the single decisive number
    in the worklog (64.4% of capped pilot generations looped).
  - HEDGE / REFUSAL: generic non-answers ("I don't know", "It depends"). A keyword + short-length
    heuristic, reported separately (a different failure mode from looping).

Run (reads only records; no GPU, no judge):
    python scripts/checks/expertqa_degeneracy.py --ood ID --prompt-regime expertqa_reppen

Self-validation: run it on the ORIGINAL no-reppen pilot and it reproduces the worklog's
73% cap / 48.5% repetition-degenerate (verified 2026-07-06):
    python scripts/checks/expertqa_degeneracy.py --ood pilot --prompt-regime expertqa
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ -> repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq.config import Config          # noqa: E402
from luq import cache                  # noqa: E402

CAP = 384  # ExpertQA Stage-0 locked max_new_tokens (a capped generation ran the whole budget)

# Hedge / refusal cues (case-insensitive substring match). Kept deliberately small and explicit.
HEDGE_CUES = [
    "i don't know", "i do not know", "it depends", "i'm not able", "i am not able",
    "i cannot", "i can't", "as an ai", "i'm sorry", "i am sorry", "i'm unable", "i am unable",
]


def distinct_n(tokens, n):
    """Fraction of DISTINCT n-grams among all n-grams: 1.0 = no repetition, low = looping."""
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def max_sentence_repeat(text):
    """Largest count of any single (normalised) sentence — >=3 is the worklog's loop flag."""
    # split on sentence enders; normalise whitespace/case so near-identical loops collapse
    sents = [re.sub(r"\s+", " ", s).strip().lower() for s in re.split(r"[.!?\n]+", text)]
    sents = [s for s in sents if len(s) > 3]  # ignore trivial fragments
    if not sents:
        return 0
    counts = {}
    for s in sents:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="expertqa")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--prompt-regime", default="")
    args = ap.parse_args()

    cfg = Config(dataset=args.dataset, ood_setting=args.ood, model_name=args.model,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(args.model, args.dataset, args.ood)
    recs = cache.load_records(cfg.cache_dir, key)
    print(f"loaded {len(recs)} records from {cfg.cache_dir}/records/{key}.jsonl\n")

    capped, d2, d3, srepeat, hedge, glen = [], [], [], [], [], []
    for r in recs:
        toks = r["gen_text"].split()
        n_gen = len(r["gen_token_ids"])
        glen.append(n_gen)
        capped.append(n_gen >= CAP)
        d2.append(distinct_n(toks, 2))
        d3.append(distinct_n(toks, 3))
        srepeat.append(max_sentence_repeat(r["gen_text"]))
        low = r["gen_text"].lower()
        hedge.append(any(c in low for c in HEDGE_CUES) or len(toks) < 8)

    capped = np.array(capped)
    d3 = np.array(d3)
    srepeat = np.array(srepeat)
    # worklog definition of repetition-degenerate: distinct-3 < 0.5 OR a sentence repeated >= 3x
    degen = (d3 < 0.5) | (srepeat >= 3)

    n = len(recs)
    print(f"cap rate (len>= {CAP}):        {capped.mean():6.1%}  ({capped.sum()}/{n})")
    print(f"median gen length (tokens): {np.median(glen):6.0f}")
    print(f"distinct-2 median:          {np.median(d2):6.2f}")
    print(f"distinct-3 median:          {np.median(d3):6.2f}")
    print(f"repetition-degenerate:      {degen.mean():6.1%}  ({degen.sum()}/{n})")
    print(f"  (distinct-3 < 0.5:        {(d3 < 0.5).mean():6.1%})")
    print(f"  (sentence repeated >=3x:  {(srepeat >= 3).mean():6.1%};  worst = {srepeat.max()}x)")
    print(f"hedge/refusal (heuristic):  {np.array(hedge).mean():6.1%}")
    if capped.any() and (~capped).any():
        print(f"\nP(degenerate | capped):     {degen[capped].mean():6.1%}")
        print(f"P(degenerate | not capped): {degen[~capped].mean():6.1%}   <- the decisive coupling number")


if __name__ == "__main__":
    main()
