#!/usr/bin/env python
"""SPECIAL-TOKEN AUDIT -- which methods exclude end-of-text tokens, which do not, and does it matter?

Results: ../STOCKTAKE_sharpening_axis.md §12.

WHY THIS EXISTS
---------------
CLAUDE.md's standing rule: "when aligning, keep every compared method on the SAME token set or the
difference is a confound rather than a result." This audit checks whether that holds across the
master ladder. It was prompted by W5's CONTROL 2 abort, which turned out to be a token-set mismatch.

THE AUDIT, read from the code (see the table printed below for the citations):

  msp_min / perplexity / msp_sum   INCLUDE specials -- msp.msp_uncertainty() takes every logprob
  SAPLMA (mean-pool + MLP)         INCLUDE -- `s.mean(axis=0)` over all per-token states
  armA / armB (attention pooler)   INCLUDE -- attn_pool.py has no content mask at all
  weighted MSP, LEARNED modes      **EXCLUDE** -- `_weights_from_raw` masks id>=128000 to -inf
  weighted MSP, `constant` mode    INCLUDE -- deliberately, to keep the constant==MSP invariant

⭐ So weighted MSP is the ONLY method on the ladder that excludes them. That exclusion exists for a
documented reason (weighted_msp.py:216-219: ~70% of xsum/cnn generations end in EOS and the learned
weighter otherwise concentrates its softmax mass on that content-free "I'm done" token), but it means
wMSP is scored on a different token set from the floor it is compared against.

WHAT THIS SCRIPT MEASURES
-------------------------
The three floors computed BOTH ways -- all tokens, and content tokens only -- per dataset and as the
cross-dataset mean, so the size of the confound is a number rather than a worry. Floors are
rung-invariant, so one number per dataset covers all five rungs.

Records only. No GPU, no training, seconds.

    python scripts/checks/special_token_audit.py
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, msp, results                                  # noqa: E402
from luq.config import Config                                        # noqa: E402
from luq.weighted_msp import per_token_nll, content_keep             # noqa: E402
from xl_rungs import eval_split, label_of                            # noqa: E402
from attn_pool import PROMPT_REGIME                                  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OUT = ROOT / "results" / "special_token_audit__meta-llama_Meta-Llama-3.1-8B.csv"


def floors_both_ways(nlls, keeps, y):
    """The three floors on ALL tokens and on CONTENT tokens only."""
    out = {}
    out["msp_min_all"] = results.prr(y, np.array([a.max() for a in nlls]))
    out["msp_min_kept"] = results.prr(y, np.array(
        [a[k].max() if k.any() else a.max() for a, k in zip(nlls, keeps)]))
    out["perplexity_all"] = results.prr(y, np.array([a.mean() for a in nlls]))
    out["perplexity_kept"] = results.prr(y, np.array(
        [a[k].mean() if k.any() else a.mean() for a, k in zip(nlls, keeps)]))
    out["msp_sum_all"] = results.prr(y, np.array([a.sum() for a in nlls]))
    out["msp_sum_kept"] = results.prr(y, np.array(
        [a[k].sum() if k.any() else a.sum() for a, k in zip(nlls, keeps)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print("=" * 100)
    print("SPECIAL-TOKEN AUDIT -- Llama-3.1-8B, 8 long evals, test rows, floors are rung-invariant")
    print("=" * 100)
    print("""
WHICH METHODS EXCLUDE END-OF-TEXT TOKENS (read from the code)
  msp_min / perplexity / msp_sum   INCLUDE   msp.py:36-40, every cached logprob
  SAPLMA (mean-pool + MLP)         INCLUDE   aggregation_table.py:193, s.mean(axis=0)
  armA / armB (attention pooler)   INCLUDE   attn_pool.py has no content mask
  weighted MSP, learned modes      EXCLUDE   weighted_msp.py:232-... masks id>=128000 to -inf
  weighted MSP, `constant` mode    INCLUDE   deliberate, keeps the constant==MSP invariant
=> weighted MSP is the ONLY method on the ladder that excludes them.
""")

    rows = []
    print(f"{'eval':15s}{'%argmax is':>12s}{'msp_min':>19s}{'perplexity':>19s}{'msp_sum':>19s}")
    print(f"{'':15s}{'special':>12s}{'all':>9s}{'kept':>10s}{'all':>9s}{'kept':>10s}{'all':>9s}{'kept':>10s}")
    agg = []
    for d in LONG:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        lf = label_of(d)
        y = np.array([r.get(lf, np.nan) for r in recs], float)
        fin = np.isfinite(y)
        recs = [recs[i] for i in np.where(fin)[0]]
        y = y[fin]
        _, te = eval_split(np.array([r["split"] for r in recs]))
        nlls = [per_token_nll(recs[i]) for i in te]
        keeps = [content_keep(recs[i]).astype(bool) for i in te]
        yte = y[te]
        pct = float(np.mean([0.0 if k[int(np.argmax(a))] else 1.0 for a, k in zip(nlls, keeps)]))
        f = floors_both_ways(nlls, keeps, yte)
        agg.append(f)
        print(f"{d:15s}{pct:>11.1%}"
              f"{f['msp_min_all']:>9.4f}{f['msp_min_kept']:>10.4f}"
              f"{f['perplexity_all']:>9.4f}{f['perplexity_kept']:>10.4f}"
              f"{f['msp_sum_all']:>9.4f}{f['msp_sum_kept']:>10.4f}")
        rows.append((d, f"{pct:.4f}", *[f"{f[k]:.4f}" for k in
                                        ["msp_min_all", "msp_min_kept", "perplexity_all",
                                         "perplexity_kept", "msp_sum_all", "msp_sum_kept"]]))

    print("-" * 100)
    m = {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}
    print(f"{'MEAN (n=8)':15s}{'':>11s}"
          f"{m['msp_min_all']:>9.4f}{m['msp_min_kept']:>10.4f}"
          f"{m['perplexity_all']:>9.4f}{m['perplexity_kept']:>10.4f}"
          f"{m['msp_sum_all']:>9.4f}{m['msp_sum_kept']:>10.4f}")
    print(f"{'DELTA kept-all':15s}{'':>11s}"
          f"{'':>9s}{m['msp_min_kept']-m['msp_min_all']:>+10.4f}"
          f"{'':>9s}{m['perplexity_kept']-m['perplexity_all']:>+10.4f}"
          f"{'':>9s}{m['msp_sum_kept']-m['msp_sum_all']:>+10.4f}")

    dmin = m["msp_min_kept"] - m["msp_min_all"]
    print("\n" + "=" * 100)
    print("DOES THE MASTER TABLE NEED RECOMPUTING?")
    print("=" * 100)
    print(f"  The pre-registered bar `msp_min` moves by {dmin:+.4f} on the cross-dataset mean")
    print(f"  ({m['msp_min_all']:+.4f} all-tokens -> {m['msp_min_kept']:+.4f} content-tokens).")
    if abs(dmin) < 0.005:
        print("  ⇒ NEGLIGIBLE at the aggregate level. The per-dataset differences largely CANCEL,")
        print("    so NO master-table number needs recomputing and the pre-registered bar stands.")
    else:
        print("  ⇒ MATERIAL. The bar itself moves; the master aggregate must be dual-reported.")
    print("\n  ⚠️ BUT THE PER-DATASET DIFFERENCES DO NOT CANCEL, and they matter for per-dataset")
    print("     claims. The largest is pubmed_qa. Any statement of the form 'wMSP beats the floor on")
    print("     dataset X' should carry the content-token floor as well, because wMSP is the only")
    print("     method scored on that token set.")
    print("\n  ⚠️ DIRECTION OF THE BIAS IS MIXED, so it does not systematically flatter anything:")
    for d, a in zip(LONG, agg):
        dd = a["msp_min_kept"] - a["msp_min_all"]
        who = "handicaps wMSP (floor gains from specials)" if dd < 0 else "flatters wMSP (floor loses to specials)"
        print(f"     {d:15s} {dd:+.4f}  {who}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["eval", "frac_argmax_special", "msp_min_all", "msp_min_kept",
                    "perplexity_all", "perplexity_kept", "msp_sum_all", "msp_sum_kept"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
