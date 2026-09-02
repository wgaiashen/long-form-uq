#!/usr/bin/env python
"""When does shrinking the learned correction towards uniform weighting help.

The constrained method decomposes exactly into a fixed content-token mean-surprisal reference plus a
learned activation-derived correction, and shrinkage pulls that correction towards zero. That gives a
concrete hypothesis: shrinkage should help where the fixed reference already ranks responses well,
and need not help where the reference is weak. The competing explanation is that shrinkage helps
merely by reducing flexibility, which predicts instead that datasets with a larger unconstrained
correction gain more.

Four dataset-level rank correlations, all pre-specified, all descriptive at eight datasets:

  1  mean departure from uniform weighting   vs shrinkage gain
  2  mean absolute learned correction        vs shrinkage gain
  3  reference quality                       vs shrinkage gain
  4  the same relation against the published one, as a reproduction

Reference quality is the reference's OWN ranking performance, scored against the same labels: the
content-token mean surprisal the method actually falls back to, not the all-token score.

POPULATION CAVEAT, PRESERVED RATHER THAN FIXED. The stored weights cover an evaluation split taken
over the full record list, which on two targets is larger than the ladder's scored population. The
mechanism statistics for those two therefore describe a marginally different population from their
own PRR. That is a property of the original analysis and is reported, not silently corrected.

    python scripts/checks/ch6_shrinkage_mechanism_tests.py --decomp <json> --master <csv> --out <csv>
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache, results                                 # noqa: E402
from luq.config import Config                                  # noqa: E402
from attn_pool import PROMPT_REGIME                            # noqa: E402
from xl_rungs import label_of                                  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
# The decomposition file labels its arms by the shrinkage coefficient itself, not by the internal
# method name. The unregularised arm is the one with no penalty on departures from uniform weighting.
UNREGULARISED = "lambda=0"
OOD = ["SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"]


def _rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def labels_by_position(dataset):
    """Record-position -> quality label, on the population the weight dump indexed."""
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))
    lf = label_of(dataset)
    # An unlabelled row stores a null, not a missing key, so a plain get-with-default still yields
    # None. Those rows are dropped by the finite filter downstream; they must never become a number.
    def _val(r):
        v = r.get(lf)
        return float(v) if isinstance(v, (int, float)) else np.nan
    return {i: _val(r) for i, r in enumerate(recs)}, lf


def shrink_gain(master_path):
    """Per dataset: constrained minus unconstrained, averaged over the four shifted settings."""
    col = None
    acc = defaultdict(dict)
    for r in csv.DictReader(open(master_path)):
        col = col or ("prr_mean" if "prr_mean" in r else "prr")
        if r["rung"] not in OOD:
            continue
        # The two assembled ladders label the same two estimators differently, because their method
        # identifiers are baked into stored result files and cannot be renamed without breaking the
        # provenance of every row that already carries them. Both spellings are accepted here:
        # "shrink2"/"shrink@2" is the constrained estimator at coefficient 2, and "norm" is the same
        # estimator with no penalty on departures from uniform weighting.
        name = {"wmsp_shrink2": "constrained", "wMSP-shrink@2": "constrained",
                "wmsp_norm": "unconstrained", "wMSP-norm": "unconstrained"}.get(r["method"])
        if name:
            acc[(r["eval"], name)].setdefault("v", []).append(float(r[col]))
    out = {}
    for d in LONG:
        a, b = acc.get((d, "constrained")), acc.get((d, "unconstrained"))
        if not a or not b:
            raise SystemExit(f"{master_path}: no constrained/unconstrained rows for {d}")
        out[d] = float(np.mean(a["v"]) - np.mean(b["v"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decomp", required=True, help="per-response decomposition json for one arm")
    ap.add_argument("--master", required=True, help="the assembled ladder for the SAME population")
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", default="meta-llama/Meta-Llama-3.1-8B")
    args = ap.parse_args()

    rows_json = [r for r in json.load(open(args.decomp)) if r["arm"] == UNREGULARISED]
    if not rows_json:
        raise SystemExit(f"{args.decomp} has no rows for the unregularised arm {UNREGULARISED!r}")
    by_ds = defaultdict(list)
    for r in rows_json:
        by_ds[r["dataset"]].append(r)

    gain = shrink_gain(args.master)

    print("=" * 104)
    print(f"WHEN DOES SHRINKAGE HELP   population: {args.population}")
    print(f"  decomposition {_rel(args.decomp)}")
    print(f"  ladder        {_rel(args.master)}")
    print("=" * 104)
    print(f"{'dataset':14s}{'n':>7s}{'reference PRR':>15s}{'mean departure':>16s}"
          f"{'mean |correction|':>19s}{'shrinkage gain':>16s}")

    rows = []
    for d in LONG:
        rs = by_ds.get(d)
        if not rs:
            print(f"{d:14s}  no stored weights; reported as unavailable, never as zero")
            rows.append({"dataset": d, "n": 0, "reference_prr": "", "mean_omega": "",
                         "mean_abs_correction": "", "shrinkage_gain": round(gain[d], 4),
                         "population": args.population})
            continue
        lab, lf = labels_by_position(d)
        pos = [r["pos"] for r in rs]
        y = np.array([lab.get(p, np.nan) for p in pos], float)
        mu = np.array([r["mu_C"] for r in rs], float)
        keep = np.isfinite(y) & np.isfinite(mu)
        # The reference is a CONFIDENCE-like quantity already: higher mean surprisal means more
        # uncertain, which is the direction the ranking metric expects.
        ref = results.prr(y[keep], mu[keep])
        om = np.array([r["Omega_C"] for r in rs], float)
        ac = np.array([r["absC"] for r in rs], float)
        om_m = float(np.nanmean(om)) if np.isfinite(om).any() else float("nan")
        ac_m = float(np.nanmean(ac)) if np.isfinite(ac).any() else float("nan")
        print(f"{d:14s}{int(keep.sum()):>7d}{ref:>+15.4f}{om_m:>16.4f}{ac_m:>19.4f}"
              f"{gain[d]:>+16.4f}")
        rows.append({"dataset": d, "n": int(keep.sum()), "label_field": lf,
                     "reference_prr": round(float(ref), 4),
                     "mean_omega": round(om_m, 4), "mean_abs_correction": round(ac_m, 4),
                     "shrinkage_gain": round(gain[d], 4), "population": args.population})

    data_rows = list(rows)          # snapshot: the summary rows appended below are not data

    def test(key, label):
        pairs = [(r[key], r["shrinkage_gain"]) for r in data_rows
                 if r[key] != "" and np.isfinite(float(r[key]))]
        x = np.array([p[0] for p in pairs], float)
        g = np.array([p[1] for p in pairs], float)
        rho, p = spearmanr(x, g)
        print(f"  {label:52s} rho {rho:+.3f}  p {p:.4f}  n {len(x)}")
        return {"dataset": f"SPEARMAN {label}", "n": len(x), "label_field": "",
                "reference_prr": f"rho={rho:+.3f}", "mean_omega": f"p={p:.4f}",
                "mean_abs_correction": "", "shrinkage_gain": "", "population": args.population}

    print("\nDataset-level rank correlations against the shrinkage gain")
    rows.append(test("mean_omega", "departure from uniform weighting vs gain"))
    rows.append(test("mean_abs_correction", "absolute learned correction vs gain"))
    rows.append(test("reference_prr", "reference quality vs gain"))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {_rel(args.out)}")


if __name__ == "__main__":
    main()
