"""Per-dataset generation-quality report: the v1-vs-v2 gate table, from the record cache only.

Plan item A.3. The 97.7 / 65.9 / 30.0 / 28.4 capping figures were produced by an ad-hoc in-session
analysis with no committed script, so v1-vs-v2 could only ever be a retelling rather than a diff. This
is that script.

Emits, per dataset: budget, generation length p50/p90, % capped, % severe / % degraded (via the
validated `luq.degeneracy` detector, which is deliberately NOT repetition-based), % EMPTY, and the gold
reference length p50/p90 beside them -- because the reference length is what decides whether a budget
is genuinely too small or whether the model is simply over-generating.

⚠️ % EMPTY is the med_quad failure mode: a repetition penalty applied over a long few-shot context can
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

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def gold_text(r):
    t = r.get("target")
    return str(t[0] if isinstance(t, list) and t else t)


# Base Llama is not instruction-tuned, so under a few-shot prompt it does not stop when the answer
# ends -- it writes the NEXT "Question: / Answer:" pair itself, inventing both. `luq.answer_span`
# documents the per-dataset shapes; this is the cheap population-level rate of the med_quad/QA one.
_FABRICATED = re.compile(r"\bQuestion\s*:")


def fabrication(text):
    """(has_fabricated_continuation, fraction of the text that is the REAL answer).

    ⚠️ This is the metric the degeneracy detector cannot see, and it is the one that matters for
    labelling: the judge scores the whole saved output, so a generation that answers correctly and
    then invents an unrelated Q&A gets marked down for text the model was never asked to produce.
    Measured on med_quad: 47.8% of the old (cap 128) generations and 92.6% of the n-gram-only 768
    arm carry it, and in the latter ~66% of the average generation is the invented part.
    """
    m = _FABRICATED.search(text or "")
    if not m:
        return False, 1.0
    return True, m.start() / max(len(text), 1)


def report(dataset, regime, budget_override, tok=None):
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=regime if regime is not None else DEFAULT_REGIME.get(dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))
    budget = budget_override if budget_override else data.MAX_NEW_TOKENS[dataset]
    glen = np.array([len(r["gen_token_ids"]) for r in recs])
    texts = [(r.get("gen_text") or "") for r in recs]
    # `capped` matches truncation_confound.py's rule (budget-1) so the two agree by construction
    capped = glen >= budget - 1
    empty = np.array([len(t.strip()) == 0 for t in texts])
    sev = np.array([degeneracy.is_severe(t) for t in texts])
    deg = np.array([degeneracy.is_degraded(t) for t in texts])
    gl = [len(g.split()) for g in (gold_text(r) for r in recs)]   # words; token count needs a tokeniser
    fab = [fabrication(t) for t in texts]
    has_fab = np.array([f[0] for f in fab])
    answer_frac = np.array([f[1] for f in fab])
    return {"dataset": dataset, "regime": cfg.prompt_regime or "(v1 default)", "n": len(recs),
            "pct_fabricated": round(100 * float(has_fab.mean()), 1),
            "mean_answer_frac": round(float(answer_frac.mean()), 3),
            "budget": int(budget),
            "gen_p50": float(np.percentile(glen, 50)), "gen_p90": float(np.percentile(glen, 90)),
            "pct_capped": round(100 * float(capped.mean()), 1),
            "pct_empty": round(100 * float(empty.mean()), 2),
            "pct_severe": round(100 * float(sev.mean()), 2),
            "pct_degraded": round(100 * float(deg.mean()), 2),
            "gold_words_p50": float(np.percentile(gl, 50)), "gold_words_p90": float(np.percentile(gl, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="med_quad,samsum,xsum,cnn_dailymail")
    ap.add_argument("--regime", default=None, help="cache namespace; omit for the v1 default per dataset")
    ap.add_argument("--budget", type=int, default=None, help="override the budget used for %%capped")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for d in args.datasets.split(","):
        try:
            rows.append(report(d.strip(), args.regime, args.budget))
        except Exception as e:                      # a missing cache is reported, never silently skipped
            print(f"  {d}: SKIPPED ({type(e).__name__}: {e})")
    if not rows:
        raise SystemExit("no datasets could be reported")

    hdr = f"{'dataset':<14} {'regime':<12} {'n':>5} {'bud':>5} {'g50':>5} {'g90':>5} " \
          f"{'%cap':>6} {'%empty':>7} {'%sev':>6} {'%deg':>6} {'%fabr':>7} {'ansfrac':>8} " \
          f"{'gold50':>7} {'gold90':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<14} {r['regime']:<12} {r['n']:>5} {r['budget']:>5} {r['gen_p50']:>5.0f} "
              f"{r['gen_p90']:>5.0f} {r['pct_capped']:>6.1f} {r['pct_empty']:>7.2f} {r['pct_severe']:>6.2f} "
              f"{r['pct_degraded']:>6.2f} {r['pct_fabricated']:>7.1f} {r['mean_answer_frac']:>8.3f} "
              f"{r['gold_words_p50']:>7.0f} {r['gold_words_p90']:>7.0f}")
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
