#!/usr/bin/env python
"""Quantify few-shot continuation junk in the generations, per model x dataset.

Occasioned 2026-08-11 by the author's visual audit of the Qwen gen-quality pages: xsum/cnn/samsum
carry "Here's the text/dialogue and its short summary…" continuations, expertqa/factscore a
multiple-choice "Available choices" artifact. This script (a) counts marker rates in gen_text for
both populations, (b) for the affected cells compares judge label and generation length on
junk-bearing vs clean rows. Findings recorded in results/analysis/QWEN_REPLICATION_VERDICT.md
(caveat block) and STOCKTAKE_qwen §6: rates are strongly model-asymmetric (samsum 37.5% Qwen vs
7.1% Llama; the MC artifact is Qwen-only; med_quad's next-Question leak is LLAMA-heavier at 47%),
labels are NOT depressed on junk rows (completion correlate), lengths are cap-inflated.
answer_span has no cut rules for samsum/cnn/expertqa/factscore — flagged, not silently "fixed".

    python scripts/checks/continuation_junk_scan.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MARKERS = {
    "heres_text_summary": "Here's the text and its short summary",
    "heres_dialogue": "Here's the dialogue and its short summary",
    "avail_choices": "Available choices",
    "single_select": "single-select problem",
    "next_Text:": "\nText:",
    "next_Question:": "\nQuestion:",
}
ROOTS = {"pubmed_qa": "cache", "med_quad": "cache", "xsum": "cache", "cnn_dailymail": "cache",
         "samsum": "cache", "asqa": "cache/asqa_rp12", "expertqa": "cache/expertqa_rp12",
         "factscore": "cache/factscore_rp12"}
LABEL = {"expertqa": "factuality", "factscore": "factuality"}
MODELS = [("Llama", "meta-llama_Meta-Llama-3.1-8B"), ("Qwen", "Qwen_Qwen2.5-14B")]
SPLIT_CASES = [  # (slug-prefix, dataset, marker key) for the junk-vs-clean comparison
    ("Qwen", "samsum", "heres_dialogue"), ("Llama", "samsum", "heres_dialogue"),
    ("Qwen", "cnn_dailymail", "heres_text_summary"),
    ("Qwen", "expertqa", "avail_choices"), ("Qwen", "factscore", "avail_choices"),
    ("Llama", "med_quad", "next_Question:"), ("Qwen", "med_quad", "next_Question:"),
]


def records(slug, d):
    p = ROOT / ROOTS[d] / "records" / f"{slug}__{d}__ID.jsonl"
    return [json.loads(l) for l in open(p)]


def main():
    for name, slug in MODELS:
        print(f"\n=== {name} — marker rates (% of generations containing) ===")
        print(f"{'dataset':15s}{'n':>6s}" + "".join(f"{k[:14]:>16s}" for k in MARKERS))
        for d in ROOTS:
            rs = records(slug, d)
            counts = {k: sum(1 for r in rs if m in (r.get("gen_text") or ""))
                      for k, m in MARKERS.items()}
            print(f"{d:15s}{len(rs):>6d}"
                  + "".join(f"{100 * counts[k] / max(len(rs), 1):>15.1f}%" for k in MARKERS))

    print("\n=== junk-bearing vs clean rows (labelled rows only) ===")
    print(f"{'model':7s}{'dataset':15s}{'n_junk':>7s}{'n_clean':>8s}{'label junk':>11s}"
          f"{'label clean':>12s}{'medlen junk':>12s}{'medlen clean':>13s}")
    slug_of = dict(MODELS)
    for mname, d, mk in SPLIT_CASES:
        rs = records(slug_of[mname], d)
        lf = LABEL.get(d, "correctness")
        j, c = [], []
        for r in rs:
            y = r.get(lf)
            if y is None:
                continue
            (j if MARKERS[mk] in (r.get("gen_text") or "") else c).append(
                (float(y), len(r["gen_token_ids"])))
        if not j:
            print(f"{mname:7s}{d:15s}   no junk rows")
            continue
        jy, jl = zip(*j); cy, cl = zip(*c)
        print(f"{mname:7s}{d:15s}{len(j):>7d}{len(c):>8d}{np.mean(jy):>11.3f}"
              f"{np.mean(cy):>12.3f}{np.median(jl):>12.0f}{np.median(cl):>13.0f}")


if __name__ == "__main__":
    main()
