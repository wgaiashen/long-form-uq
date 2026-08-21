"""Per-dataset generation-quality report: the v1-vs-v2 gate table, from the record cache only.

Plan item A.3. The 97.7 / 65.9 / 30.0 / 28.4 capping figures were produced by an ad-hoc in-session
analysis with no committed script, so v1-vs-v2 could only ever be a retelling rather than a diff. This
is that script.

Emits, per dataset: budget, generation length p50/p90, % capped, % severe / % degraded (via the
validated `luq.degeneracy` detector, which is deliberately NOT repetition-based), % EMPTY, and the gold
reference length p50/p90 beside them -- because the reference length is what decides whether a budget
is genuinely too small or whether the model is simply over-generating.

% EMPTY is the med_quad failure mode: a repetition penalty applied over a long few-shot context can
force immediate EOS. It produced ~35% empty generations there, and no other quality check catches it.

Read-only, CPU, no GPU, no API, no £.

    python scripts/checks/generation_quality.py --datasets samsum,xsum,cnn_dailymail,med_quad
    python scripts/checks/generation_quality.py --datasets samsum --regime v2pilot   # the v2 side
"""
import argparse
import csv as _csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, data, degeneracy  # noqa: E402
from luq.config import Config  # noqa: E402

# DEFAULT only. Kept as the previous hard-coded value so every existing invocation stays
# byte-identical; a second model is passed with --model. (Was a bare constant until 2026-08-08,
# which made the script silently SKIP every dataset when pointed at a non-Llama cache namespace --
# it reported "no datasets" rather than "wrong model", which reads like missing data.)
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def gold_text(r):
    t = r.get("target")
    return str(t[0] if isinstance(t, list) and t else t)


# Base Llama is not instruction-tuned, so under a few-shot prompt it does not stop when the answer
# ends -- it writes the NEXT "Question: / Answer:" pair itself, inventing both. `luq.answer_span`
# documents the per-dataset shapes; this is the cheap population-level rate of the med_quad/QA one.
_FABRICATED = re.compile(r"\bQuestion\s*:")

# ADDED 2026-08-15. `_FABRICATED` only sees an invented "Question:" restart -- the BASE-model
# few-shot failure. It is structurally blind to the INSTRUCT-model failure: the model answers
# correctly and then continues in its assistant persona ("Let me know if you'd like me to analyze
# anything else!"). Measured on gemma-2-9b-it samsum: 80% of generations, with the real answer only
# ~63% of the saved text, while pct_fabricated read 0.0%.
# This matters because the judge scores the WHOLE saved output, so the filler is graded as if it
# were the summary -- the same harm pct_fabricated exists to measure, arriving by a different route.
# Kept as a SEPARATE column: pct_fabricated's semantics are load-bearing for the existing Llama and
# Qwen numbers and must not shift under them.
_CHATTER = re.compile(
    r"(let me know|i hope (this|that) helps|hope this helps|would you like|feel free to|"
    r"anything else|if you'd like|shall i|do you want me to|is there anything)", re.I)


def chatter(text):
    """(has_assistant_chatter, fraction of the text that precedes it)."""
    m = _CHATTER.search(text or "")
    if not m:
        return False, 1.0
    return True, m.start() / max(len(text), 1)


def fabrication(text):
    """(has_fabricated_continuation, fraction of the text that is the REAL answer).

    This is the metric the degeneracy detector cannot see, and it is the one that matters for
    labelling: the judge scores the whole saved output, so a generation that answers correctly and
    then invents an unrelated Q&A gets marked down for text the model was never asked to produce.
    Measured on med_quad: 47.8% of the old (cap 128) generations and 92.6% of the n-gram-only 768
    arm carry it, and in the latter ~66% of the average generation is the invented part.
    """
    m = _FABRICATED.search(text or "")
    if not m:
        return False, 1.0
    return True, m.start() / max(len(text), 1)


def report(dataset, regime, budget_override, tok=None, model=DEFAULT_MODEL):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=regime if regime is not None else DEFAULT_REGIME.get(dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(model, dataset, "ID"))
    budget = budget_override if budget_override else data.MAX_NEW_TOKENS[dataset]
    glen = np.array([len(r["gen_token_ids"]) for r in recs])
    texts = [(r.get("gen_text") or "") for r in recs]
    # `capped` matches truncation_confound.py's rule (budget-1) so the two agree by construction
    capped = glen >= budget - 1
    empty = np.array([len(t.strip()) == 0 for t in texts])
    sev = np.array([degeneracy.is_severe(t) for t in texts])
    deg = np.array([degeneracy.is_degraded(t) for t in texts])
    golds = [gold_text(r) for r in recs]
    gl = [len(g.split()) for g in golds]   # words; token count needs a tokeniser
    # Gold length in TOKENS, via the `tok` hook that had been declared but never wired (2026-08-08).
    # THIS IS THE COLUMN THE BUDGET DECISION NEEDS. `budget` is a token count, so budget/gold_words
    # is a units mismatch. The words->tokens factor is NOT a safe constant: measured on this gold with
    # the Qwen tokeniser it ranges 1.24 (samsum) to 1.45 (asqa), and using a single ~1.31 factor flips
    # asqa's verdict from 0.96x (below the gate) to 1.06x (above it) -- i.e. the approximation fails
    # precisely on the one marginal dataset the gate exists to catch. Measure, do not scale.
    # Without a tokeniser this stays None -- NOT zero. A not-measured cell that reads as a number is
    # how a budget gets declared adequate on evidence nobody produced.
    if tok is not None:
        gt = [len(tok(g, add_special_tokens=False)["input_ids"]) for g in golds]
        gold_tok = {"gold_tok_p50": float(np.percentile(gt, 50)),
                    "gold_tok_p90": float(np.percentile(gt, 90))}
    else:
        gold_tok = {"gold_tok_p50": None, "gold_tok_p90": None}
    fab = [fabrication(t) for t in texts]
    has_fab = np.array([f[0] for f in fab])
    answer_frac = np.array([f[1] for f in fab])
    cht = [chatter(t) for t in texts]
    has_cht = np.array([c[0] for c in cht])
    cht_frac = np.array([c[1] for c in cht])
    return {"dataset": dataset, "regime": cfg.prompt_regime or "(v1 default)", "n": len(recs),
            "pct_fabricated": round(100 * float(has_fab.mean()), 1),
            "mean_answer_frac": round(float(answer_frac.mean()), 3),
            "pct_chatter": round(100 * float(has_cht.mean()), 1),
            "mean_prechatter_frac": round(float(cht_frac.mean()), 3),
            "budget": int(budget),
            "gen_p50": float(np.percentile(glen, 50)), "gen_p90": float(np.percentile(glen, 90)),
            "pct_capped": round(100 * float(capped.mean()), 1),
            "pct_empty": round(100 * float(empty.mean()), 2),
            "pct_severe": round(100 * float(sev.mean()), 2),
            "pct_degraded": round(100 * float(deg.mean()), 2),
            "gold_words_p50": float(np.percentile(gl, 50)), "gold_words_p90": float(np.percentile(gl, 90)),
            **gold_tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="med_quad,samsum,xsum,cnn_dailymail")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model slug whose record cache to read. Defaults to the Llama keystone, so "
                         "existing calls are unchanged. Pass Qwen/Qwen2.5-14B for the second model.")
    ap.add_argument("--regime", default=None, help="cache namespace; omit for the v1 default per dataset")
    ap.add_argument("--budget", type=int, default=None, help="override the budget used for %%capped")
    ap.add_argument("--gold-tokenizer", default=None, metavar="MODEL",
                    help="tokeniser used to measure the GOLD reference length in TOKENS, e.g. "
                         "Qwen/Qwen2.5-14B. Needed to evaluate the budget gate, since budget is a "
                         "token count and gold_words is the wrong unit. Omitted: the gold_tok_* "
                         "columns stay BLANK (not measured), never 0.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = None
    if args.gold_tokenizer:
        from transformers import AutoTokenizer     # imported lazily: the default path needs no ML stack
        tok = AutoTokenizer.from_pretrained(args.gold_tokenizer)

    rows = []
    for d in args.datasets.split(","):
        try:
            rows.append(report(d.strip(), args.regime, args.budget, tok=tok, model=args.model))
        except Exception as e:                      # a missing cache is reported, never silently skipped
            print(f"  {d}: SKIPPED ({type(e).__name__}: {e})")
    if not rows:
        raise SystemExit("no datasets could be reported")

    hdr = f"{'dataset':<14} {'regime':<12} {'n':>5} {'bud':>5} {'g50':>5} {'g90':>5} " \
          f"{'%cap':>6} {'%empty':>7} {'%sev':>6} {'%deg':>6} {'%fabr':>7} {'ansfrac':>8} " \
          f"{'gold50':>7} {'gold90':>7}" \
          + ("  |{:>7} {:>7} {:>7}".format('gtok50', 'gtok90', 'ratio') if tok is not None else "")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<14} {r['regime']:<12} {r['n']:>5} {r['budget']:>5} {r['gen_p50']:>5.0f} "
              f"{r['gen_p90']:>5.0f} {r['pct_capped']:>6.1f} {r['pct_empty']:>7.2f} {r['pct_severe']:>6.2f} "
              f"{r['pct_degraded']:>6.2f} {r['pct_fabricated']:>7.1f} {r['mean_answer_frac']:>8.3f} "
              f"{r['gold_words_p50']:>7.0f} {r['gold_words_p90']:>7.0f}"
              # The gate columns, printed ONLY when a tokeniser was supplied, so default output is
              # unchanged. ratio = budget / gold_tok_p90, both TOKENS.
              + (f"  |{r['gold_tok_p50']:>7.0f} {r['gold_tok_p90']:>7.0f} "
                 f"{r['budget'] / r['gold_tok_p90']:>6.2f}x"
                 if r.get("gold_tok_p90") else ""))
    if tok is not None:
        print(f"\ngold_tok_* measured with {args.gold_tokenizer}. ratio = budget / gold_tok_p90 (TOKENS/TOKENS).")
        print("ratio < 1.00 = DEFECT (the reference cannot fit the budget) -> gate trips, stop and report.")
        print("ratio >= 1.00 with a high %cap = the model over-generates -> NOT a reason to change the budget.")
        print("the ratio is meaningless where gold is not a reference GENERATION (factscore's gold is")
        print("   the entity NAME, ~7 tokens, so its ratio is not-applicable rather than 'lots of room').")
    print("\ngold_* are WORDS (tokeniser-free); gen_* are TOKENS -- do not compare the two columns directly.")
    print("%fabr   = generations that invent a follow-up 'Question:' -- the few-shot continuation the")
    print("         degeneracy detector CANNOT see, and the one the judge is scored over.")
    print("ansfrac = mean fraction of the text that is the REAL answer (1.000 = no fabrication).")

    out = Path(args.out) if args.out else ROOT / "results" / "generation_quality.csv"
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
