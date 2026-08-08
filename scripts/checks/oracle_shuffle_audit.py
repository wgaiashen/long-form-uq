#!/usr/bin/env python
"""F-A -- RETROSPECTIVE SHUFFLED-CANDIDATE AUDIT of every oracle ceiling this project quotes.

Plan: ../PLAN_sharpening_axis.md.   Results: ../STOCKTAKE_sharpening_axis.md §9.

WHY THIS EXISTS
---------------
`router_pdl.py` established that per-example oracle headroom is largely a MONOTONICITY ARTEFACT: a
SHUFFLED candidate bought +0.091 and pure Gaussian noise +0.087, against multimax's +0.059. A max
over K candidates is upward-biased even when no candidate is genuinely better on any particular unit,
because the max picks up noise.

⚠️ That does not only bound future work -- it invalidates ORACLE CEILINGS ALREADY BEING QUOTED, in the
stocktake, in the deck, and in this workstream's own summaries. The most-quoted one is the
best-of-two-endpoints oracle at +0.2472 (i.e. +0.062 over msp_min). This script computes the shuffled
baseline for each and reports the NET figure. **Anything that does not clear its shuffled baseline is
retracted.**

THE NULL, AND WHY IT IS THE RIGHT ONE
--------------------------------------
For each candidate, keep its OVERALL LEVEL but destroy its UNIT-SPECIFIC pattern: permute that
candidate's per-unit deviations across units. Then take the max as before. The result is what
"selecting the best candidate per unit" scores when there is nothing real to select on -- the pure
noise-max bias, at the same K, the same n, and the same marginal spread.

    col_mean[j]          preserved  -> a candidate that is better overall stays better overall
    resid[:, j] permuted            -> but WHERE it is better is destroyed

So `net = real_oracle - shuffled_oracle` is the headroom attributable to genuine per-unit signal.
Reporting `real_oracle - always_best_single` (what the project has been doing) conflates the two.

⚠️ PREDICTION, STATED IN THE PLAN BEFORE THIS WAS RUN, so it can be wrong: the 2-ENDPOINT oracle
should largely survive (its candidates differ hugely per dataset -- 0.545 on pubmed -- and max-of-K
bias is worst when candidates are equal in expectation), while the per-cell LAMBDA oracle over 3
near-identical candidates should mostly evaporate. If that ordering is violated, the intuition behind
the whole retraction set is wrong and it all needs re-reading.

Reads the master table only. No training, no GPU.

    python scripts/checks/oracle_shuffle_audit.py
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MASTER = ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
OUT = ROOT / "results" / "oracle_shuffle_audit__meta-llama_Meta-Llama-3.1-8B.csv"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
N_PERM = 4000
SEED = 0


def load():
    g = defaultdict(dict)
    for r in _csv.DictReader(open(MASTER)):
        if r.get("seed_regime") != "3seed":
            continue
        try:
            g[r["method"]][(r["rung"], r["eval"])] = float(r["prr"])
        except (ValueError, TypeError):
            continue
    return g


def shuffled_oracle_INDEP(M, rng, n_perm=N_PERM):
    """⚠️ THE FIRST NULL I WROTE, AND IT IS WRONG. Kept, and reported, because it nearly caused a
    blanket retraction of every oracle in the project on a broken control.

    It permutes each candidate's residuals INDEPENDENTLY down the units. That destroys the
    POSITIVE CORRELATION between candidates (a dataset that is easy for msp_min tends to be easy for
    perplexity too). The max of near-independent variables is larger than the max of correlated ones,
    so this null is INFLATED -- it returned a shuffled oracle ABOVE the real one in all five cases,
    i.e. a negative "net headroom", which is not a finding about the data but an artefact of the null.
    """
    col_mean = M.mean(axis=0)
    resid = M - col_mean
    vals = np.empty(n_perm)
    for i in range(n_perm):
        R = np.column_stack([rng.permutation(resid[:, j]) for j in range(M.shape[1])])
        vals[i] = (col_mean + R).max(axis=1).mean()
    return float(vals.mean()), float(vals.std())


def shuffled_oracle(M, rng, n_perm=N_PERM):
    """THE CORRECT NULL: destroy ONLY the candidate x unit interaction, keeping everything else.

    Model the table as  M[i,j] = mu_j (candidate level) + a_i (unit effect) + e_ij (interaction).
    The null being tested is "no candidate is differentially better on any particular unit", i.e.
    e_ij is EXCHANGEABLE ACROSS CANDIDATES WITHIN A UNIT. So the permutation is WITHIN EACH ROW,
    across candidates -- which preserves:
        * each candidate's overall level  (mu_j)
        * each unit's difficulty          (a_i)
        * the correlation between candidates
    and destroys only WHICH candidate is best WHERE, which is precisely what an oracle exploits.
    """
    col_mean = M.mean(axis=0)
    resid = M - col_mean                       # a_i + e_ij
    unit = resid.mean(axis=1, keepdims=True)   # a_i
    dev = resid - unit                         # e_ij, exchangeable across j under H0
    vals = np.empty(n_perm)
    for i in range(n_perm):
        P = np.apply_along_axis(rng.permutation, 1, dev)
        vals[i] = (col_mean + unit + P).max(axis=1).mean()
    return float(vals.mean()), float(vals.std()), vals


def audit(name, M, labels, rng, rows):
    """M: (n_units, n_cand). Reports real oracle, shuffled oracle, and the NET headroom."""
    best_single = float(M.mean(axis=0).max())
    best_name = labels[int(np.argmax(M.mean(axis=0)))]
    real = float(M.max(axis=1).mean())
    sh, sd, dist = shuffled_oracle(M, rng)
    bad, _ = shuffled_oracle_INDEP(M, rng)
    gross = real - best_single
    net = real - sh
    # ⚠️ The verdict is a PERMUTATION p, not a fixed threshold on the net. A fixed cutoff (my first
    # draft used +0.005) is arbitrary and, at n = 8 where the null's own spread is ~0.011-0.015, it
    # would call differences of one standard deviation "survivors". p = fraction of null oracles that
    # reach the real one.
    p = float((dist >= real).mean())
    verdict = "SURVIVES" if p < 0.05 else "RETRACT"
    print(f"\n{name}")
    print(f"  units={M.shape[0]}  candidates={M.shape[1]}  {labels}")
    print(f"  always-best-single ({best_name}) {best_single:+.4f}")
    print(f"  REAL oracle                      {real:+.4f}   gross headroom {gross:+.4f}"
          f"   <- what has been quoted")
    print(f"  SHUFFLED oracle (correct null)   {sh:+.4f} +/- {sd:.4f}"
          f"   artefact = {sh - best_single:+.4f}")
    print(f"  ** NET headroom                  {net:+.4f} **   perm p={p:.4f}  -> {verdict}")
    print(f"     [broken independent-permutation null would have said {bad:+.4f}, "
          f"net {real - bad:+.4f} -- see the docstring]")
    rows.append((name, M.shape[0], M.shape[1], f"{best_single:.4f}", f"{real:.4f}",
                 f"{sh:.4f}", f"{gross:.4f}", f"{net:.4f}", f"{p:.4f}", verdict, f"{bad:.4f}"))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    g = load()
    rng = np.random.RandomState(SEED)
    rows = []

    print("=" * 100)
    print("F-A -- SHUFFLED-CANDIDATE AUDIT OF QUOTED ORACLE CEILINGS")
    print("Population: widened cells_long, meta-llama/Llama-3.1-8B, 3-seed. OOD rungs unless stated.")
    print(f"Null: per candidate, keep its level, permute its per-unit deviations. {N_PERM} permutations.")
    print("=" * 100)

    def dmean(m):
        """per-dataset OOD mean, n=8"""
        return np.array([np.mean([g[m][(r, d)] for r in OOD]) for d in LONG])

    def cells(m):
        """per-cell OOD, n=32"""
        return np.array([g[m][(r, d)] for d in LONG for r in OOD])

    # 1. THE MOST-QUOTED ONE: best-of-two-endpoints per dataset (+0.2472)
    audit("1. best-of-two-endpoints, per DATASET (the +0.2472 figure)",
          np.column_stack([dmean("msp_min"), dmean("perplexity")]),
          ["msp_min", "perplexity"], rng, rows)

    # 2. free-only selection: the three floors
    audit("2. free-only selection (3 floors), per DATASET",
          np.column_stack([dmean("msp_min"), dmean("perplexity"), dmean("msp_sum")]),
          ["msp_min", "perplexity", "msp_sum"], rng, rows)

    # 3. per-dataset over 4 methods
    four = ["msp_min", "perplexity", "SAPLMA", "armA(attention)"]
    audit("3. per-DATASET over 4 methods", np.column_stack([dmean(m) for m in four]), four, rng, rows)

    # 4. per-CELL over the same 4 (n=32) -- the unit the project usually quotes
    audit("4. per-CELL over 4 methods (n=32)", np.column_stack([cells(m) for m in four]), four, rng, rows)

    # 5. G3: the per-cell LAMBDA oracle, the one gating F4
    lam = ["wMSP-norm", "wMSP-shrink@2", "wMSP-shrink@10"]
    net_lam = audit("5. per-CELL LAMBDA oracle (G3 -- gates F4)",
                    np.column_stack([cells(m) for m in lam]), lam, rng, rows)

    print("\n" + "=" * 100)
    print("READING")
    print("=" * 100)
    print("  'gross headroom' is real_oracle - always_best_single: what has been quoted.")
    print("  'NET headroom'   is real_oracle - shuffled_oracle:    what is actually attributable")
    print("                   to genuine per-unit signal rather than to taking a max over K.")
    print("  Anything with NET <= +0.005 is RETRACTED from the stocktake and the deck.")
    if net_lam <= 0.005:
        print("\n  ⚠️ G3 VERDICT: the per-cell LAMBDA headroom does NOT clear its shuffled baseline.")
        print("     F4 (pool-aware shrinkage) is DEAD BEFORE W5 LANDS -- there is no real per-cell")
        print("     lambda signal to map onto, so no mapping from pool features can recover one.")
    else:
        print(f"\n  G3 VERDICT: lambda headroom net {net_lam:+.4f} survives -> F4 may proceed to G1/G2.")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["case", "n_units", "n_cand", "always_best_single", "real_oracle",
                    "shuffled_oracle", "gross_headroom", "net_headroom", "perm_p", "verdict", "broken_indep_null"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
