"""What will the v2 judge run cost, measured rather than guessed (plan item 0.5).

Committed version of an analysis that was originally an inline one-off, so the figures quoted in the
stocktake had no reproducible source. It gates the regeneration spend, so it needs one.

WHAT IT MEASURES, AND WHAT IT ASSUMES. Input tokens are MEASURED: the judge prompt is built with
`llm_judge.build_prompt`, the real code, and tokenised with the model's own encoding rather than a
chars/4 rule. Everything else is an assumption and is labelled as one:
  - the PRICE is a parameter, not a fact. Confirm it against the current price page.
  - OUTPUT tokens are NOT priced here. The judge emits a single number, but gpt-5-mini is a reasoning
    model and reasoning tokens bill as output while being invisible. This is the main unknown.
  - ⚠️ THE BEST ANCHOR IS NOT THIS SCRIPT. Every one of these datasets has already been judged once, so
    the previous spend on the OpenAI dashboard is a better estimate than any token count -- and it is a
    FLOOR rather than a ceiling, because v1 covered SHORTER generations than v2 will.

The range is honest rather than a point estimate: LOW keeps today's generation lengths, HIGH assumes
every row runs to the new cap. Cost is dominated by the SOURCE text, so the budget choice moves it
much less than one might expect -- except on med_quad, whose answers genuinely grow.

Read-only. No API call is made and nothing is spent.

    python scripts/checks/judge_cost_estimate.py
    python scripts/checks/judge_cost_estimate.py --budgets med_quad=768,samsum=96 --price-in 0.25
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, data  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="med_quad,samsum,xsum,cnn_dailymail")
    ap.add_argument("--budgets", default="med_quad=768,samsum=96,xsum=96,cnn_dailymail=192",
                    help="the v2 budget per dataset, dataset=N")
    ap.add_argument("--sample", type=int, default=300, help="prompts tokenised per dataset")
    ap.add_argument("--price-in", type=float, default=0.25,
                    help="$ per 1M INPUT tokens. ⚠️ ASSUMED -- confirm against the current price page.")
    ap.add_argument("--rejudge-n", type=int, default=200,
                    help="rows per dataset for the judge-effect / noise-floor re-judge")
    ap.add_argument("--out", default=str(ROOT / "results" / "judge_cost_estimate.csv"))
    args = ap.parse_args()

    budgets = {}
    for item in args.budgets.split(","):
        k, v = item.split("="); budgets[k.strip()] = int(v)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        ntok = lambda s: len(enc.encode(s))                                    # noqa: E731
        tok_note = "tiktoken o200k_base (measured)"
    except Exception:                       # never silently swap in a worse estimator without saying so
        ntok = lambda s: max(1, len(s) // 4)                                   # noqa: E731
        tok_note = "⚠️ tiktoken UNAVAILABLE -- falling back to a chars/4 APPROXIMATION"
    print(f"tokeniser: {tok_note}\n")

    rows = []
    for d in [x.strip() for x in args.datasets.split(",")]:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID", prompt_regime=REGIME.get(d, ""))
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        idx = np.linspace(0, len(recs) - 1, min(args.sample, len(recs))).astype(int)
        tin = np.array([ntok(llm_judge.build_prompt(recs[i], d)) for i in idx], float)
        tgen = np.array([ntok(recs[i].get("gen_text") or "") for i in idx], float)
        base = float(tin.mean() - tgen.mean())          # the prompt minus the generation it contains
        bud = budgets.get(d, data.MAX_NEW_TOKENS[d])
        lo = len(recs) * (base + float(tgen.mean())) / 1e6
        hi = len(recs) * (base + bud) / 1e6
        rj_lo = args.rejudge_n * (base + float(tgen.mean())) / 1e6
        rj_hi = args.rejudge_n * (base + bud) / 1e6
        rows.append({"dataset": d, "n_rows": len(recs), "v2_budget": bud,
                     "in_per_call_now": round(float(tin.mean()), 1),
                     "gen_per_call_now": round(float(tgen.mean()), 1),
                     "M_in_low": round(lo, 3), "M_in_high": round(hi, 3),
                     "rejudge_M_low": round(rj_lo, 3), "rejudge_M_high": round(rj_hi, 3)})

    n_calls = sum(r["n_rows"] for r in rows) + args.rejudge_n * len(rows)
    lo_t = sum(r["M_in_low"] + r["rejudge_M_low"] for r in rows)
    hi_t = sum(r["M_in_high"] + r["rejudge_M_high"] for r in rows)

    print(f"{'dataset':<15} {'rows':>6} {'budget':>7} {'in/call':>8} {'M in LOW':>9} {'M in HIGH':>10}")
    for r in rows:
        print(f"{r['dataset']:<15} {r['n_rows']:>6} {r['v2_budget']:>7} {r['in_per_call_now']:>8.0f} "
              f"{r['M_in_low']:>9.3f} {r['M_in_high']:>10.3f}")
    print(f"\n  judge calls (incl. {args.rejudge_n}/dataset re-judge): {n_calls:,}")
    print(f"  input tokens: {lo_t:.2f}M .. {hi_t:.2f}M")
    print(f"  INPUT cost @ ${args.price_in}/M (ASSUMED): ${lo_t*args.price_in:.2f} .. ${hi_t*args.price_in:.2f}")
    print("  output/reasoning tokens NOT priced -- the main unknown.")
    print("  ⚠️ Better anchor: these sets were all judged once already. Use the previous dashboard spend")
    print("     x >=1.15, and treat that as a FLOOR, since v1 covered shorter generations.")

    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
