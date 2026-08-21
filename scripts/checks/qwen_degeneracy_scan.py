#!/usr/bin/env python
"""§4 of the Qwen validity audit: the STRONG detector over all 18,464 Qwen generations.

PRIMARY DETECTOR: `luq.degeneracy.classify`, IMPORTED AND UNCHANGED. That module was written
2026-07-06, after the Llama ExpertQA false-GO, and its docstring is explicit about why a
repetition-only scan must not be the gate: forbidden from repeating n-grams, the base model drops
into NON-repetitive degeneration (word salad, function-word-dropped rambling, code emission,
enumeration explosions) that distinct-n is structurally blind to. Its thresholds were hand-validated
against legitimate numbered lists and reference sections, which it must NOT flag.

SECONDARY, CLEARLY LABELLED: distinct-2 / distinct-3 / repeated-sentence, the definitions from
`scripts/checks/expertqa_degeneracy.py`. These are the WEAKER scan that produced the false GO. They
are reported for continuity with the Llama numbers, never as the gate.

ONE ADDED DIAGNOSTIC, DEFINITION FIXED BEFORE ANY PER-DATASET RESULT WAS SEEN.
The canonical detector cannot see the failure mode found in §1's expertqa example: a fluent
one-sentence answer followed by the model inventing a multiple-choice quiz about its own answer.
That is neither repetition nor salad -- every existing signal passes it -- but it is exactly what
the audit brief names, "text that merely continues until the token cap without behaving like the
requested task". §4 of the brief permits adding such a diagnostic provided its definition is fixed
in advance, so it is written out here, in full, and was committed before the scan was run:

    TASK ABANDONMENT / PROMPT-FORMAT BLEED
    The generation re-opens the prompt's own scaffolding or starts a fresh task. Conservative and
    purely lexical: a line (after optional whitespace) matching any of
        Question: | Answer: | Available choices: | (N). | Summary: | Article: | Story: |
        Abstract: | Text: | Document: | Dialogue: | Conversation: | A single-select problem |
        Is the question answered
    It is REPORTED SEPARATELY and never merged into severe/degraded. It is also EXPECTED to fire
    legitimately on some sets -- a few-shot QA continuation is the designed behaviour that
    `--truncate-long` and `--truncate-answer-span` exist to cut -- so a high rate is a prompt to
    look, not a verdict.

Read-only: reads records, writes one CSV and prints a table. No GPU.

    python scripts/checks/qwen_degeneracy_scan.py
    python scripts/checks/qwen_degeneracy_scan.py --model meta-llama/Meta-Llama-3.1-8B
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, degeneracy                        # noqa: E402  (UNCHANGED detector)
from luq.config import Config                            # noqa: E402
from attn_pool import PROMPT_REGIME                      # noqa: E402

QWEN = "Qwen/Qwen2.5-14B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]

# Budgets as executed (slurm/qwen_generate_full.sbatch). Identical for Llama.
BUDGET = {"pubmed_qa": 128, "med_quad": 128, "asqa": 256, "xsum": 56,
          "cnn_dailymail": 128, "samsum": 56, "expertqa": 384, "factscore": 256}

# The added diagnostic, fixed above. Compiled once so the definition cannot drift per dataset.
BLEED = re.compile(
    r"^\s*(?:Question:|Answer:|Available choices:|\(\d+\)\.|Summary:|Article:|Story:|Abstract:"
    r"|Text:|Document:|Dialogue:|Conversation:|A single-select problem|Is the question answered)",
    re.MULTILINE)


def distinct_n(text, n):
    t = text.split()
    if len(t) < n:
        return 1.0
    g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return len(set(g)) / len(g)


def max_sentence_repeat(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    if not sents:
        return 0
    c = {}
    for s in sents:
        c[s] = c.get(s, 0) + 1
    return max(c.values())


def load(model, dataset):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    key = cache.run_key(model, dataset, "ID")
    p = Path(cfg.cache_dir) / "records" / f"{key}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    slug = cache._slug(args.model)
    out = ROOT / (args.out or f"results/analysis/degeneracy_scan__{slug}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print(f"STRONG DEGENERACY SCAN   model={args.model}")
    print("PRIMARY = luq.degeneracy.classify, imported UNCHANGED (severe: content-run>=25 OR")
    print("  code-density>=0.005 OR whitespace-gap>=8;  degraded: severe OR content-run>=15).")
    print("SECONDARY (the weaker scan that gave the Llama false-GO): distinct-2/3, repeated sentence.")
    print("ADDED, definition fixed BEFORE any result was seen: task-abandonment / prompt-format bleed.")
    print("=" * 118)

    rows, per_ex = [], []
    hdr = (f"{'dataset':15s}{'n':>6s}{'sev%':>7s}{'degr%':>7s}{'bleed%':>8s}{'cap%':>7s}"
           f"{'d2':>7s}{'d3':>7s}{'rep-sent%':>10s}"
           f"{'P(sev|cap)':>11s}{'P(sev|not)':>11s}{'P(bld|cap)':>11s}{'P(bld|not)':>11s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for d in LONG:
        recs = load(args.model, d)
        if recs is None:
            print(f"{d:15s}  RECORDS MISSING -- reported absent, not zero")
            continue
        bud = BUDGET[d]
        sev, deg, bld, cap, d2, d3, rs = [], [], [], [], [], [], []
        for r in recs:
            t = r.get("gen_text", "") or ""
            c = degeneracy.classify(t)
            n_gen = len(r.get("gen_token_ids", []))
            b = bool(BLEED.search(t))
            sev.append(c["severe"]); deg.append(c["degraded"]); bld.append(b)
            cap.append(n_gen >= bud)
            d2.append(distinct_n(t, 2)); d3.append(distinct_n(t, 3))
            rs.append(max_sentence_repeat(t) >= 3)
            per_ex.append({"model": args.model, "dataset": d, "idx": r.get("idx"),
                           "n_gen": n_gen, "capped": int(n_gen >= bud),
                           "severe": int(c["severe"]), "degraded": int(c["degraded"]),
                           "bleed": int(b), "max_content_run": c["max_content_run"],
                           "code_density": c["code_density"],
                           "max_whitespace_gap": c["max_whitespace_gap"],
                           "distinct2": round(distinct_n(t, 2), 4),
                           "repeated_sentence": int(max_sentence_repeat(t) >= 3)})
        sev, deg, bld, cap = map(np.array, (sev, deg, bld, cap))

        def cond(flag, mask):
            return 100 * flag[mask].mean() if mask.any() else float("nan")

        print(f"{d:15s}{len(recs):>6d}{100*sev.mean():>7.1f}{100*deg.mean():>7.1f}"
              f"{100*bld.mean():>8.1f}{100*cap.mean():>7.1f}"
              f"{np.median(d2):>7.3f}{np.median(d3):>7.3f}{100*np.mean(rs):>10.1f}"
              f"{cond(sev, cap):>11.1f}{cond(sev, ~cap):>11.1f}"
              f"{cond(bld, cap):>11.1f}{cond(bld, ~cap):>11.1f}")
        rows.append({"model": args.model, "dataset": d, "n": len(recs), "budget": bud,
                     "severe_pct": round(100 * sev.mean(), 2),
                     "degraded_pct": round(100 * deg.mean(), 2),
                     "bleed_pct": round(100 * bld.mean(), 2),
                     "cap_pct": round(100 * cap.mean(), 2),
                     "distinct2_median": round(float(np.median(d2)), 4),
                     "distinct3_median": round(float(np.median(d3)), 4),
                     "repeated_sentence_pct": round(100 * float(np.mean(rs)), 2),
                     "P_severe_given_capped": round(cond(sev, cap), 2),
                     "P_severe_given_notcapped": round(cond(sev, ~cap), 2),
                     "P_degraded_given_capped": round(cond(deg, cap), 2),
                     "P_degraded_given_notcapped": round(cond(deg, ~cap), 2),
                     "P_bleed_given_capped": round(cond(bld, cap), 2),
                     "P_bleed_given_notcapped": round(cond(bld, ~cap), 2)})

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    pe = out.parent / f"degeneracy_perexample__{slug}.csv"
    with open(pe, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ex[0].keys()))
        w.writeheader(); w.writerows(per_ex)
    print(f"\nwrote {out}\nwrote {pe}  ({len(per_ex):,} rows)")
    print("\nREAD THESE AS DIAGNOSTICS, NOT VERDICTS. The detector's thresholds were tuned on")
    print("   ExpertQA prose; a whitespace or code signal may be legitimate formatting on a")
    print("   summarisation or medical set, and `bleed` fires by design on few-shot continuation.")
    print("   Every triggered mode gets surfaced in the HTML for a human to look at.")


if __name__ == "__main__":
    main()
