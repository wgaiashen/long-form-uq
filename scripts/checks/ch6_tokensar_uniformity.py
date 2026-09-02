#!/usr/bin/env python
"""Is the external relevance score an independent comparator on long-form generation, or not.

The like-for-like comparison reports a relevance-weighted score alongside the fixed probability
aggregates. If the external relevance model barely differentiates tokens over a long response, then
its weights are close to uniform, the score collapses towards mean token NLL, and reporting it as a
separate baseline would overstate how many distinct comparators the table contains.

That was observed and stated in prose but never computed by a committed script, so it is computed
here. Two quantities per dataset, both on the ladder's own scored test rows:

  rho          Spearman rank correlation between the relevance-weighted score and mean token NLL.
               Rank correlation, not value correlation, because the evaluation metric is rank-based.
  entropy      the response's relevance weights normalised to sum to one, Shannon entropy divided by
               log(T). 1.0 is exactly uniform, 0.0 is all the weight on one token. Reported as the
               median over responses, with the interquartile range, because a mean would hide a
               bimodal population.

INDEX BASES. The relevance cache is a plain array in ORIGINAL record order while the ladder indexes
a FILTERED array with the unlabelled rows dropped, so the mapping is explicit and asserted. This
mirrors likeforlike_table.py exactly; getting it wrong shifts the datasets that drop rows by up to
several hundred places and still returns a plausible number.

    python scripts/checks/ch6_tokensar_uniformity.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, msp as msp_mod                          # noqa: E402
from luq.config import Config                                  # noqa: E402
from xl_rungs import eval_split, label_of                      # noqa: E402
from likeforlike_table import LONG_SRC, REGIME, SAR_SUFFIX, SLUG, MODEL  # noqa: E402

OUT = ROOT / "results" / "analysis" / f"ch6_cleanv2_tokensar_diagnostic__{SLUG}.csv"
EPS = 1e-12


def normalised_entropy(rel):
    """Shannon entropy of the normalised relevance weights, divided by log(T). NaN when undefined."""
    r = np.asarray(rel, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 2:
        return np.nan
    r = r - r.min() if r.min() < 0 else r          # weights must be non-negative to be a distribution
    total = r.sum()
    if total <= EPS:
        return np.nan                              # every token equally irrelevant: undefined, not 1.0
    w = r / total
    nz = w[w > EPS]
    return float(-(nz * np.log(nz)).sum() / np.log(T))


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("=" * 104)
    print(f"RELEVANCE-WEIGHTING UNIFORMITY  model={MODEL}  population=corrected span, scored test rows")
    print("rho: Spearman(relevance-weighted score, mean token NLL).  entropy: 1.0 is exactly uniform.")
    print("=" * 104)
    print(f"{'dataset':14s}{'n_test':>8s}{'rho':>9s}{'median H':>11s}{'IQR H':>18s}"
          f"{'min H':>9s}{'median T':>10s}")

    rows = []
    for d in LONG_SRC:
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID", prompt_regime=REGIME[d])
        recs_all = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        y_all = np.array([r.get(label_of(d), np.nan) for r in recs_all], dtype=float)
        keep = np.where(np.isfinite(y_all))[0]
        recs = [recs_all[i] for i in keep]
        split = np.array([r["split"] for r in recs])
        _, te = eval_split(split)
        te = np.asarray(te)

        sp = ROOT / "cache" / "sar" / f"{SLUG}__{d}__ID__sentence{SAR_SUFFIX.get(d, '')}.npz"
        if not sp.exists():
            print(f"{d:14s}  no relevance cache; reported as unavailable, never as uniform")
            rows.append({"dataset": d, "n_test": 0, "spearman_vs_mean_token_nll": "",
                         "median_normalised_weight_entropy": "", "iqr_lo": "", "iqr_hi": "",
                         "min_normalised_weight_entropy": "", "median_n_tokens": "",
                         "relevance_cache": "MISSING"})
            continue
        z = np.load(sp, allow_pickle=True)
        ts = np.asarray(z["tokensar"], dtype=float)
        rel = z["relevance"]
        if len(ts) != len(recs_all) or len(rel) != len(recs_all):
            sys.exit(f"FATAL {d}: relevance cache has {len(ts)} rows against {len(recs_all)} records "
                     "-- refusing to align two different bases by guess.")
        orig = keep[te]                                   # filtered test positions -> original order

        sar = ts[orig]
        mean_nll = np.array([msp_mod.msp_uncertainty(recs[i]["token_logprobs"], "perplexity")
                             for i in te])
        rho = float(stats.spearmanr(sar, mean_nll).statistic)

        H = np.array([normalised_entropy(rel[j]) for j in orig], dtype=float)
        Hf = H[np.isfinite(H)]
        T = np.array([len(recs[i]["token_logprobs"]) for i in te], dtype=float)
        lo, hi = np.percentile(Hf, [25, 75])

        print(f"{d:14s}{len(te):>8d}{rho:>9.4f}{np.median(Hf):>11.4f}"
              f"   [{lo:.4f}, {hi:.4f}]{Hf.min():>9.4f}{np.median(T):>10.0f}")
        rows.append({"dataset": d, "n_test": len(te),
                     "spearman_vs_mean_token_nll": round(rho, 4),
                     "median_normalised_weight_entropy": round(float(np.median(Hf)), 4),
                     "iqr_lo": round(float(lo), 4), "iqr_hi": round(float(hi), 4),
                     "min_normalised_weight_entropy": round(float(Hf.min()), 4),
                     "median_n_tokens": int(np.median(T)),
                     "relevance_cache": sp.name})

    scored = [r for r in rows if r["relevance_cache"] != "MISSING"]
    rhos = [r["spearman_vs_mean_token_nll"] for r in scored]
    Hs = [r["median_normalised_weight_entropy"] for r in scored]
    print(f"\nacross {len(scored)} datasets: Spearman {min(rhos):.3f} to {max(rhos):.3f}, "
          f"median normalised weight entropy {min(Hs):.3f} to {max(Hs):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
