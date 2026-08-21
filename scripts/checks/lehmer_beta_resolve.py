"""Resolve the Llama FActScore best-Lehmer-beta disagreement (16 vs inf) and print the
full beta curve for every dataset on both models.

Two project documents disagree about the Llama FActScore oracle-best beta.  The grid on disk
settles it, and the answer is that BOTH are right about different quantities: the summary
carries an `oracle_best_finite_beta` column, which by construction cannot return `inf`, while
the argmax over the whole evaluated grid (which includes the `inf` endpoint, i.e. msp_min)
can.  This script prints both, plus the gap between them, so the report can state whether
they are effectively tied.

Population note: this is the aggregation-regime grid — per-example unsupervised scores on the
evaluation split, NOT the supervised OOD ladder.  n and label_field are printed per dataset.
The two models are reported separately and never pooled.

Usage:
    python scripts/checks/lehmer_beta_resolve.py
"""

import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = ["meta-llama_Meta-Llama-3.1-8B", "Qwen_Qwen2.5-14B"]
# The evaluated grid.  beta=0 is the perplexity endpoint and beta=inf the msp_min endpoint;
# both are genuine members of the family, not extrapolations.
BETAS = ["0", "0.5", "1", "2", "4", "8", "16", "inf"]
COLS = [f"prr_beta_{b}" for b in BETAS]


def main():
    for m in MODELS:
        path = f"{REPO}/results/analysis/aggregation_regime_summary__{m}.csv"
        d = pd.read_csv(path)
        print(f"\n{'=' * 100}\nMODEL {m}   source {os.path.basename(path)}\n{'=' * 100}")

        # full curve, one row per dataset
        show = d[["eval", "n"] + COLS + ["oracle_best_finite_beta"]].copy()
        show.columns = ["eval", "n"] + [f"b={b}" for b in BETAS] + ["best_finite(file)"]
        print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

        # argmax over the WHOLE grid (inf included) vs over finite betas only
        print(f"\n{'eval':15s} {'argmax(all incl inf)':>22s} {'PRR':>9s} "
              f"{'argmax(finite)':>15s} {'PRR':>9s} {'gap':>9s}")
        rows = []
        for _, r in d.iterrows():
            vals = {b: r[f"prr_beta_{b}"] for b in BETAS}
            b_all = max(vals, key=vals.get)
            fin = {b: v for b, v in vals.items() if b != "inf"}
            b_fin = max(fin, key=fin.get)
            gap = vals[b_all] - fin[b_fin]
            rows.append(dict(eval=r["eval"], argmax_all=b_all, prr_all=vals[b_all],
                             argmax_finite=b_fin, prr_finite=fin[b_fin], gap=gap,
                             file_best_finite=r["oracle_best_finite_beta"]))
            print(f"{r['eval']:15s} {b_all:>22s} {vals[b_all]:+9.4f} "
                  f"{b_fin:>15s} {fin[b_fin]:+9.4f} {gap:+9.4f}")

        # the file's own column must agree with our recomputed finite argmax
        for r in rows:
            got, want = str(r["file_best_finite"]), r["argmax_finite"]
            if float(got) != float(want):
                print(f"  {r['eval']}: file oracle_best_finite_beta={got} but recomputed "
                      f"finite argmax={want}")

        fs = [r for r in rows if r["eval"] == "factscore"][0]
        print(f"\nFActScore focus: beta=16 -> {d[d['eval'] == 'factscore']['prr_beta_16'].iloc[0]:+.6f}   "
              f"beta=inf -> {d[d['eval'] == 'factscore']['prr_beta_inf'].iloc[0]:+.6f}   "
              f"difference = {abs(d[d['eval'] == 'factscore']['prr_beta_inf'].iloc[0] - d[d['eval'] == 'factscore']['prr_beta_16'].iloc[0]):.6f}")
        print(f"  argmax over the whole grid = beta {fs['argmax_all']}; "
              f"best FINITE beta = {fs['argmax_finite']}")


if __name__ == "__main__":
    main()
