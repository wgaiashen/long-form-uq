#!/usr/bin/env python
"""The fixed Lehmer sweep as a report table: per target, per coefficient, plus the macro.

The Lehmer mean interpolates between averaging token surprisal and taking its single largest value,
so sweeping its coefficient traces the whole mean-to-extreme continuum with one estimator. The two
endpoints are the existing fixed baselines, which is what makes the curve interpretable: coefficient
zero IS mean token NLL and the limit IS the minimum-token-probability ranking.

THE PER-TARGET BEST COEFFICIENT IS DESCRIPTIVE ONLY. It is an argmax over the grid computed against
the target's own labels, so it is an upper bound on what any selection rule could achieve, never a
selection rule itself. It is labelled as such in the output and must be labelled as such wherever it
is quoted.

Reads the sweep the registered scorer already wrote; adds no new grid and no new coefficient.

    python scripts/checks/ch6_lehmer_sweep_table.py --sweep <csv> --out <csv>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
DEFAULT_SWEEP = ROOT / "results" / "analysis" / f"ch6_cleanv2_sharpening_family__{SLUG}.csv"
DEFAULT_OUT = ROOT / "results" / "analysis" / f"ch6_cleanv2_lehmer_sweep__{SLUG}.csv"
ORDER = ["0.0", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "inf"]
LABEL = {"0.0": "0 (mean token NLL)", "inf": "infinity (minimum token probability)"}



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
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--population", default="meta-llama/Meta-Llama-3.1-8B, corrected span")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.sweep)) if r["family"] == "lehmer_beta"]
    if not rows:
        raise SystemExit(f"{args.sweep} carries no Lehmer family rows")
    prr = {(r["dataset"], r["param"]): float(r["prr"]) for r in rows}

    params = sorted({r["param"] for r in rows},
                    key=lambda p: ORDER.index(p) if p in ORDER else 99)
    if params != ORDER:
        raise SystemExit(f"grid changed: found {params}, expected the registered {ORDER}. "
                         "This table reports the registered grid and never a substitute.")
    missing = [(d, p) for d in LONG for p in params if (d, p) not in prr]
    if missing:
        raise SystemExit(f"sweep is incomplete: {missing[:6]}; refusing a partial curve")

    print("=" * 116)
    print(f"FIXED LEHMER SWEEP   population: {args.population}   grid: {ORDER}")
    print("The best coefficient per target is a target-label argmax: descriptive, not selectable.")
    print("=" * 116)
    print(f"{'dataset':14s}" + "".join(f"{('b=' + p):>10s}" for p in params)
          + f"{'best b':>9s}{'best PRR':>10s}{'gain over mean':>16s}{'gain over extreme':>19s}")

    out, curves = [], {p: [] for p in params}
    for d in LONG:
        vals = [prr[(d, p)] for p in params]
        for p, v in zip(params, vals):
            curves[p].append(v)
        finite = [(p, v) for p, v in zip(params, vals) if p != "inf"]
        best_p, best_v = max(finite, key=lambda t: t[1])
        mean_end, extreme_end = prr[(d, "0.0")], prr[(d, "inf")]
        out.append({"scope": "per target", "dataset": d,
                    **{f"beta={p}": round(prr[(d, p)], 4) for p in params},
                    "best_finite_beta_descriptive": best_p,
                    "best_finite_prr_descriptive": round(best_v, 4),
                    "mean_endpoint": round(mean_end, 4),
                    "extreme_endpoint": round(extreme_end, 4),
                    "best_minus_mean_endpoint": round(best_v - mean_end, 4),
                    "best_minus_extreme_endpoint": round(best_v - extreme_end, 4),
                    "population": args.population})
        print(f"{d:14s}" + "".join(f"{v:>+10.4f}" for v in vals)
              + f"{best_p:>9s}{best_v:>+10.4f}{best_v - mean_end:>+16.4f}"
              f"{best_v - extreme_end:>+19.4f}")

    macro = {p: float(np.mean(curves[p])) for p in params}
    best_fixed = max((p for p in params if p != "inf"), key=lambda p: macro[p])
    row = {"scope": "MACRO (eight targets)", "dataset": "",
           **{f"beta={p}": round(macro[p], 4) for p in params},
           "best_finite_beta_descriptive": best_fixed,
           "best_finite_prr_descriptive": round(macro[best_fixed], 4),
           "mean_endpoint": round(macro["0.0"], 4),
           "extreme_endpoint": round(macro["inf"], 4),
           "best_minus_mean_endpoint": round(macro[best_fixed] - macro["0.0"], 4),
           "best_minus_extreme_endpoint": round(macro[best_fixed] - macro["inf"], 4),
           "population": args.population}
    out.append(row)
    print(f"{'MACRO':14s}" + "".join(f"{macro[p]:>+10.4f}" for p in params)
          + f"{best_fixed:>9s}{macro[best_fixed]:>+10.4f}")
    print(f"\nBest fixed interior coefficient on the macro is beta={best_fixed} at "
          f"{macro[best_fixed]:+.4f}, against {macro['0.0']:+.4f} at the mean endpoint and "
          f"{macro['inf']:+.4f} at the extreme endpoint. Chosen after inspecting target labels, so "
          f"it is a descriptive upper bound, not a model-selection result.")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"wrote {_rel(Path(args.out))}")


if __name__ == "__main__":
    main()
