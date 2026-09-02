#!/usr/bin/env python
"""Response length as a candidate explanation for the preferred aggregation rule, as two tables.

Both quantities already exist inside the aggregation-regime audit, but only as printed text, and a
figure whose values are transcribed from a log is a figure nobody can check. This writes them as
artifacts, reusing that audit's own tercile rule and PRR helper rather than a second implementation.

  dataset level     each target's median retained response length against the PRR difference between
                    the extreme-token rule and mean aggregation, plus the rank correlation across
                    the eight targets. A dataset-level selector would show a consistent shift.
  within dataset    the same scores inside equal-count length terciles of each target, because a
                    relationship can exist within a dataset while being absent across them.

GATE. Both tables are recomputed here and must reproduce the audit's own printed values, which is
what makes reusing the same rule meaningful rather than circular: the audit output is an independent
run of the same definition over the same rows table.

  ALSO REPORTED: the MEAN retained length beside the median. On the corrected-span population one
  target keeps a median at the generation cap while its mean falls by a third, because just over
  half its rows were left uncut. The median alone would hide that, so both are carried. The median
  remains the quantity the correlation uses, unchanged from the original analysis.

    python scripts/checks/ch6_length_tables.py --rows <rows csv> --out-prefix <prefix>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregation_regime_audit import LONG, MIN_GROUP, prr_of, terciles   # noqa: E402

SLUG = "meta-llama_Meta-Llama-3.1-8B"
DEFAULT_ROWS = ROOT / "results" / "analysis" / f"ch6_cleanv2_aggregation_regime_rows__{SLUG}.csv"
SCORES = [("perplexity_score", "mean token NLL"),
          ("msp_min_score", "minimum token probability"),
          ("lehmer_beta_1", "Lehmer beta=1"),
          ("lehmer_beta_2", "Lehmer beta=2")]



def _rel(p):
    """Path for display. A path given on the command line need not sit under the repository."""
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=str(DEFAULT_ROWS))
    ap.add_argument("--out-prefix", default=str(ROOT / "results" / "analysis" / "ch6_cleanv2"))
    ap.add_argument("--population", default="meta-llama/Meta-Llama-3.1-8B, corrected span")
    args = ap.parse_args()
    df = pd.read_csv(args.rows)

    missing = [d for d in LONG if d not in set(df["eval"])]
    if missing:
        raise SystemExit(f"rows table is missing {missing}; refusing a partial length analysis")

    # ---------------- dataset level ----------------
    print("=" * 100)
    print(f"RESPONSE LENGTH AT THE DATASET LEVEL   population: {args.population}")
    print("=" * 100)
    print(f"{'dataset':14s}{'n':>7s}{'median len':>12s}{'mean len':>10s}{'cap rate':>10s}"
          f"{'min':>9s}{'mean NLL':>10s}{'min - mean':>12s}")
    ds_rows = []
    for d in LONG:
        g = df[df["eval"] == d]
        med = float(g["n_tokens"].median())
        avg = float(g["n_tokens"].mean())
        cap = float(g["hit_generation_cap"].mean())
        p_min = prr_of(g, "msp_min_score")
        p_ppl = prr_of(g, "perplexity_score")
        ds_rows.append({"dataset": d, "n": len(g),
                        "median_retained_length": round(med, 1),
                        "mean_retained_length": round(avg, 1),
                        "generation_cap_rate": round(cap, 4),
                        "prr_minimum_token_probability": round(p_min, 4),
                        "prr_mean_token_nll": round(p_ppl, 4),
                        "min_minus_mean": round(p_min - p_ppl, 4),
                        "population": args.population})
        print(f"{d:14s}{len(g):>7d}{med:>12.1f}{avg:>10.1f}{cap:>10.3f}"
              f"{p_min:>+9.4f}{p_ppl:>+10.4f}{p_min - p_ppl:>+12.4f}")

    x = np.array([r["median_retained_length"] for r in ds_rows], float)
    y = np.array([r["min_minus_mean"] for r in ds_rows], float)
    rho, p = spearmanr(x, y)
    xm = np.array([r["mean_retained_length"] for r in ds_rows], float)
    rho_m, p_m = spearmanr(xm, y)
    print(f"\nSpearman(median length, min - mean) = {rho:+.4f}, p = {p:.4f}, n = {len(x)}")
    print(f"Spearman(mean length,   min - mean) = {rho_m:+.4f}, p = {p_m:.4f}   "
          f"(reported alongside, not a substitute; the median is the original quantity)")
    ds_rows.append({"dataset": "SPEARMAN median length vs min - mean", "n": len(x),
                    "median_retained_length": "", "mean_retained_length": "",
                    "generation_cap_rate": "",
                    "prr_minimum_token_probability": f"rho={rho:+.4f}",
                    "prr_mean_token_nll": f"p={p:.4f}",
                    "min_minus_mean": "", "population": args.population})
    ds_rows.append({"dataset": "SPEARMAN mean length vs min - mean", "n": len(x),
                    "median_retained_length": "", "mean_retained_length": "",
                    "generation_cap_rate": "",
                    "prr_minimum_token_probability": f"rho={rho_m:+.4f}",
                    "prr_mean_token_nll": f"p={p_m:.4f}",
                    "min_minus_mean": "", "population": args.population})

    out1 = Path(args.out_prefix + f"_length_dataset__{SLUG}.csv")
    with open(out1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ds_rows[0].keys()))
        w.writeheader(); w.writerows(ds_rows)
    print(f"wrote {_rel(out1)}")

    # ---------------- within dataset ----------------
    print("\n" + "=" * 100)
    print("RESPONSE LENGTH WITHIN EACH DATASET (equal-count terciles of the target's own rows)")
    print("=" * 100)
    print(f"{'dataset':14s}{'tercile':>9s}{'n':>7s}{'len range':>16s}{'label mean':>12s}"
          + "".join(f"{lbl[:16]:>18s}" for _, lbl in SCORES))
    st_rows = []
    for d in LONG:
        g = df[df["eval"] == d]
        for t, gg in enumerate(terciles(g, "n_tokens")):
            base = {"dataset": d, "tercile": t + 1, "n": len(gg), "population": args.population}
            if len(gg) < MIN_GROUP or gg["quality_label"].nunique() < 2:
                # Stated, never dropped: a blank row would read as "measured and zero".
                base.update({"length_min": "", "length_max": "", "label_mean": "",
                             **{lbl: "" for _, lbl in SCORES},
                             "note": "too small or degenerate; not measured"})
                st_rows.append(base)
                print(f"{d:14s}{t + 1:>9d}{len(gg):>7d}   (too small / degenerate, "
                      f"stated not silent)")
                continue
            base.update({"length_min": int(gg["n_tokens"].min()),
                         "length_max": int(gg["n_tokens"].max()),
                         "label_mean": round(float(gg["quality_label"].mean()), 4),
                         **{lbl: round(prr_of(gg, col), 4) for col, lbl in SCORES},
                         "note": ""})
            st_rows.append(base)
            print(f"{d:14s}{t + 1:>9d}{len(gg):>7d}"
                  f"{str(base['length_min']) + '-' + str(base['length_max']):>16s}"
                  f"{base['label_mean']:>12.3f}"
                  + "".join(f"{base[lbl]:>+18.4f}" for _, lbl in SCORES))

    fields = ["dataset", "tercile", "n", "length_min", "length_max", "label_mean"] \
             + [lbl for _, lbl in SCORES] + ["note", "population"]
    out2 = Path(args.out_prefix + f"_length_stratified__{SLUG}.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(st_rows)
    print(f"wrote {_rel(out2)}")


if __name__ == "__main__":
    main()
