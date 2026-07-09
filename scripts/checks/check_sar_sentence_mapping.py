"""STEP 5: validate the sentence-level SAR token->sentence mapping (`sar._token_sentence_ids`).

This mapping is OUR OWN invention (no reference exists), so it needs its own check. Per record we verify:
  1. every generated token maps to exactly one sentence id in [0, n_sent);
  2. sentence ids are NON-DECREASING (tokens are in reading order, so sentences must be too) -- a
     mis-ordered id means a token was assigned to the wrong sentence;
  3. reconstructing each sentence from its grouped token ids ~ an INDEPENDENT regex sentence split of
     gen_text (sentence COUNT match + content coverage).
Until this passes we cannot distinguish "SAR uninformative on long-form" from "our mapping is broken".

    python scripts/checks/check_sar_sentence_mapping.py --n 40
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.features import sar  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"


def regex_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def jaccard(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(len(wa | wb), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--datasets", nargs="+", default=["pubmed_qa", "med_quad"])
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    all_valid = all_monotone = True
    for d in args.datasets:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID")
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))[: args.n]
        n_valid = n_mono = 0
        count_match = []
        jac = []
        for r in recs:
            g = r["gen_token_ids"]
            sid, n_sent = sar._token_sentence_ids(tok, g, r.get("gen_text", tok.decode(g, skip_special_tokens=True)))
            valid = (len(sid) == len(g)) and (sid.min() >= 0) and (sid.max() < n_sent)
            mono = all(sid[i] <= sid[i + 1] for i in range(len(sid) - 1))
            n_valid += valid; n_mono += mono
            all_valid &= valid; all_monotone &= mono
            # reconstruct sentences from grouped tokens vs regex split
            regex = regex_sentences(r.get("gen_text", ""))
            grouped = [tok.decode([t for t, s in zip(g, sid) if s == j], skip_special_tokens=True).strip()
                       for j in range(n_sent)]
            grouped = [s for s in grouped if s]
            count_match.append(len(grouped) == len(regex))
            # best-match jaccard of each regex sentence to the grouped set (content coverage)
            for rs in regex:
                jac.append(max((jaccard(rs, gs) for gs in grouped), default=0.0))
        print(f"  {d}: valid {n_valid}/{len(recs)} | monotone {n_mono}/{len(recs)} | "
              f"sentence-count match {100*np.mean(count_match):.0f}% | mean content jaccard {np.mean(jac):.2f}",
              flush=True)
    ok = all_valid and all_monotone
    print(f"\nMAPPING {'PASS' if ok else 'FAIL'} (every token -> one sentence, ids non-decreasing)", flush=True)
    print("  (count-match / jaccard are quality signals vs a regex split, not hard pass/fail -- sentence "
          "segmentation is inherently fuzzy; low jaccard would mean our grouping loses content.)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
