#!/usr/bin/env python
"""A3/A4 — dataset-level regime table + EXPLORATORY diagnostics over the A2 rows table.

EVERY NUMBER HERE IS EXPLORATORY MECHANISM ANALYSIS on already-read test labels (the plan's
rule: analyses using Llama test labels are exploratory, never selection). The "best beta" column
is a test-label argmax and is printed ORACLE/DESCRIPTIVE ONLY. n = 8 datasets: the A4.1 feature
table is reported IN FULL, predeclared, with no significance hunting — Spearman rho is a
description of monotone association, not a test result.

Inputs: results/analysis/aggregation_regime_rows__<slug>.csv (built + gated by
aggregation_regime_rows.py; its floors reproduce the published master to 4 dp before any row is
written). Population: each dataset's canonical scored TEST rows, carve legacy.

A3   per-dataset regime table (family, n, length, cap share, quality, endpoint PRRs + delta,
     the full finite-beta Lehmer curve, oracle best beta, median concentration stats).
A4.1 dataset-level (n=8) Spearman: each predeclared feature vs D_endpoint = PRR(msp_min) −
     PRR(perplexity), and vs the oracle best finite beta. Full table, no winner-picking.
A4.2 within-dataset length terciles: PRR of perplexity / msp_min / Lehmer beta=1 / beta=2 per
     tercile + label mean per tercile (benchmark-validity diagnostic, not a selector).
A4.3 generation-cap stratification (both groups >= MIN_GROUP rows, predeclared 30).
A4.4 sentence-count terciles (weak proxy). Claim-count arm: UNAVAILABLE by construction —
     records persist aggregate ratios only, no per-claim states/counts exist (2026-08-10 audit);
     stated here rather than silently absent.

    python scripts/checks/aggregation_regime_audit.py                        # Llama
    python scripts/checks/aggregation_regime_audit.py --model Qwen/Qwen2.5-14B
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, results                            # noqa: E402
from luq.data import TASK_OF                              # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
BETAS = ["0.5", "1", "2", "4", "8", "16"]
SCORE_COLS = {"0": "perplexity_score", "inf": "msp_min_score",
              **{b: f"lehmer_beta_{b}" for b in BETAS}}
BETA_ORDER = ["0"] + BETAS + ["inf"]
MIN_GROUP = 30                                           # predeclared A4.3 minimum group size

# A4.1's predeclared dataset-level features (medians over the dataset's test rows).
A41_FEATURES = ["median_length", "iqr_length", "cap_rate", "median_max_z",
                "median_top1_mass_share", "median_top5_mass_share", "median_top10pct_mass_share",
                "median_nll_entropy_norm", "median_max_minus_second", "median_sentence_count"]


def prr_of(df, col):
    return results.prr(df["quality_label"].to_numpy(), df[col].to_numpy())


def curve(df):
    return {b: prr_of(df, SCORE_COLS[b]) for b in BETA_ORDER}


def terciles(df, col):
    """Rows split into 3 by the dataset's OWN col terciles. Ties go left (deterministic)."""
    q1, q2 = df[col].quantile([1 / 3, 2 / 3])
    return [df[df[col] <= q1], df[(df[col] > q1) & (df[col] <= q2)], df[df[col] > q2]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()
    slug = cache._slug(args.model)
    rows_csv = ROOT / "results" / "analysis" / f"aggregation_regime_rows__{slug}.csv"
    out_csv = ROOT / "results" / "analysis" / f"aggregation_regime_summary__{slug}.csv"
    df = pd.read_csv(rows_csv)
    missing = [d for d in LONG if d not in set(df["eval"])]
    if missing:
        raise SystemExit(f"rows table incomplete — missing {missing}; refusing a partial audit")

    print("=" * 110)
    print(f"A3/A4 AGGREGATION-REGIME AUDIT (EXPLORATORY)  model={args.model}")
    print(f"rows: {rows_csv.name}  ({len(df)} scored test responses, 8/8 datasets present)")
    print("=" * 110)

    # ---------------- A3 ----------------
    summary = []
    for d in LONG:
        g = df[df["eval"] == d]
        cv = curve(g)
        finite_best = max(BETAS + ["0"], key=lambda b: cv[b])          # best FINITE beta
        row = {
            "eval": d, "task": TASK_OF[d], "n": len(g),
            "med_len": g["n_tokens"].median(),
            "iqr_len": g["n_tokens"].quantile(0.75) - g["n_tokens"].quantile(0.25),
            "cap_share": g["hit_generation_cap"].mean(),
            "mean_quality": g["quality_label"].mean(),
            "prr_perplexity": cv["0"], "prr_msp_min": cv["inf"],
            "d_endpoint_min_minus_ppl": cv["inf"] - cv["0"],
            **{f"prr_beta_{b}": cv[b] for b in BETA_ORDER},
            "oracle_best_finite_beta": finite_best,
            "median_max_z": g["max_z"].median(),
            "median_top1_mass_share": g["top1_mass_share"].median(),
            "median_top5_mass_share": g["top5_mass_share"].median(),
            "median_top10pct_mass_share": g["top10pct_mass_share"].median(),
            "median_nll_entropy_norm": g["nll_entropy_norm"].median(),
            "median_max_minus_second": g["max_minus_second"].median(),
            "median_sentence_count": g["sentence_count"].median(),
        }
        summary.append(row)
    sm = pd.DataFrame(summary)

    print("\nA3 — DATASET-LEVEL REGIME TABLE (oracle-best-beta column is TEST-LABEL DESCRIPTIVE ONLY)")
    hdr = f"{'eval':15s}{'task':14s}{'n':>6s}{'medlen':>8s}{'cap%':>7s}{'qual':>7s}" \
          + "".join(f"b={b:>4s}" for b in BETA_ORDER) + f"{'best':>6s}{'d_end':>8s}"
    print(hdr)
    for _, r in sm.iterrows():
        print(f"{r['eval']:15s}{r['task']:14s}{r['n']:>6d}{r['med_len']:>8.1f}"
              f"{100 * r['cap_share']:>6.1f}%{r['mean_quality']:>7.3f}"
              + "".join(f"{r[f'prr_beta_{b}']:>+6.2f}" for b in BETA_ORDER)
              + f"{r['oracle_best_finite_beta']:>6s}{r['d_endpoint_min_minus_ppl']:>+8.3f}")
    print("  (honest LODO beta: W4 recorded AGGREGATE-level LODO only — the `w4_lodo` rows of "
          "sharpening_family__...__round2.csv; no per-dataset honest beta exists. Stated, not "
          "re-derived here.)")

    # ---------------- A4.1 ----------------
    feats = pd.DataFrame({
        "median_length": sm["med_len"], "iqr_length": sm["iqr_len"], "cap_rate": sm["cap_share"],
        "median_max_z": sm["median_max_z"],
        "median_top1_mass_share": sm["median_top1_mass_share"],
        "median_top5_mass_share": sm["median_top5_mass_share"],
        "median_top10pct_mass_share": sm["median_top10pct_mass_share"],
        "median_nll_entropy_norm": sm["median_nll_entropy_norm"],
        "median_max_minus_second": sm["median_max_minus_second"],
        "median_sentence_count": sm["median_sentence_count"],
    })
    d_end = sm["d_endpoint_min_minus_ppl"]
    best_b = sm["oracle_best_finite_beta"].astype(float)
    print("\nA4.1 — DATASET-LEVEL SPEARMAN (n = 8; EXPLORATORY; full predeclared table, no "
          "winner-picking; p-values are descriptive at n=8)")
    print(f"{'feature':30s}{'rho vs d_endpoint':>20s}{'p':>8s}{'rho vs best beta':>20s}{'p':>8s}")
    for f in A41_FEATURES:
        r1, p1 = spearmanr(feats[f], d_end)
        r2, p2 = spearmanr(feats[f], best_b)
        print(f"{f:30s}{r1:>+20.3f}{p1:>8.3f}{r2:>+20.3f}{p2:>8.3f}")
    print("  claim-count feature: UNAVAILABLE (no per-claim metadata exists in any record).")

    # ---------------- A4.2 length terciles ----------------
    print("\nA4.2 — LENGTH TERCILES per dataset (benchmark-validity diagnostic; EXPLORATORY)")
    print(f"{'eval':15s}{'tercile':>9s}{'n':>6s}{'label_mean':>11s}{'ppl':>8s}{'min':>8s}"
          f"{'b=1':>8s}{'b=2':>8s}")
    for d in LONG:
        g = df[df["eval"] == d]
        for t, gg in enumerate(terciles(g, "n_tokens")):
            if len(gg) < MIN_GROUP or gg["quality_label"].nunique() < 2:
                print(f"{d:15s}{t + 1:>9d}{len(gg):>6d}   (too small / degenerate — skipped, "
                      f"stated not silent)")
                continue
            print(f"{d:15s}{t + 1:>9d}{len(gg):>6d}{gg['quality_label'].mean():>11.3f}"
                  f"{prr_of(gg, 'perplexity_score'):>+8.3f}{prr_of(gg, 'msp_min_score'):>+8.3f}"
                  f"{prr_of(gg, 'lehmer_beta_1'):>+8.3f}{prr_of(gg, 'lehmer_beta_2'):>+8.3f}")

    # ---------------- A4.3 cap stratification ----------------
    print(f"\nA4.3 — GENERATION-CAP STRATA (groups >= {MIN_GROUP} rows; EXPLORATORY)")
    print(f"{'eval':15s}{'group':>10s}{'n':>6s}{'label_mean':>11s}{'ppl':>8s}{'min':>8s}"
          f"{'b=1':>8s}{'b=2':>8s}")
    for d in LONG:
        g = df[df["eval"] == d]
        for name, gg in (("finished", g[g["hit_generation_cap"] == 0]),
                         ("capped", g[g["hit_generation_cap"] == 1])):
            if len(gg) < MIN_GROUP or gg["quality_label"].nunique() < 2:
                print(f"{d:15s}{name:>10s}{len(gg):>6d}   (group below minimum — reported, "
                      f"not scored)")
                continue
            print(f"{d:15s}{name:>10s}{len(gg):>6d}{gg['quality_label'].mean():>11.3f}"
                  f"{prr_of(gg, 'perplexity_score'):>+8.3f}{prr_of(gg, 'msp_min_score'):>+8.3f}"
                  f"{prr_of(gg, 'lehmer_beta_1'):>+8.3f}{prr_of(gg, 'lehmer_beta_2'):>+8.3f}")

    # ---------------- A4.4 sentence-count terciles ----------------
    print("\nA4.4 — SENTENCE-COUNT TERCILES (weak proxy; claim-count arm UNAVAILABLE by "
          "construction; EXPLORATORY)")
    print(f"{'eval':15s}{'tercile':>9s}{'n':>6s}{'label_mean':>11s}{'ppl':>8s}{'min':>8s}"
          f"{'b=1':>8s}{'b=2':>8s}")
    for d in LONG:
        g = df[df["eval"] == d]
        for t, gg in enumerate(terciles(g, "sentence_count")):
            if len(gg) < MIN_GROUP or gg["quality_label"].nunique() < 2:
                print(f"{d:15s}{t + 1:>9d}{len(gg):>6d}   (too small / degenerate — skipped)")
                continue
            print(f"{d:15s}{t + 1:>9d}{len(gg):>6d}{gg['quality_label'].mean():>11.3f}"
                  f"{prr_of(gg, 'perplexity_score'):>+8.3f}{prr_of(gg, 'msp_min_score'):>+8.3f}"
                  f"{prr_of(gg, 'lehmer_beta_1'):>+8.3f}{prr_of(gg, 'lehmer_beta_2'):>+8.3f}")

    sm.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
