#!/usr/bin/env python
"""Recompute the reported numbers from the published result tables.

Reads the per-population master tables and prints the quantities the write-up quotes, so that a
reader can check them against the data rather than taking them on trust. Nothing is retyped: every
figure below is computed from the files.

Two conventions govern every number here, and both matter for reading the output.

    A blank value means not measured. It never means zero. A cell missing from a master makes this
    script raise rather than quietly average over what is present.

    The unit of analysis is the dataset, n = 8, with training seeds averaged inside a cell first.
    The training-free scores do not depend on the training pool, so their value repeats across the
    four shifted settings; treating those as 32 independent observations would inflate them
    fourfold.

Comparisons should be made at four decimal places. The write-up prints three, and averaging values
that have already been rounded to three produces differences of 0.001 that are artefacts of the
rounding rather than of the data.

    python scripts/report_numbers.py
    python scripts/report_numbers.py --dir published_results
    python scripts/report_numbers.py --dir results/hybrids --glob 'pdl_consolidated_master__*.csv'
"""
import argparse
import csv
import glob as globmod
import os
import sys

import numpy as np
from scipy import stats

EVALS = ["asqa", "cnn_dailymail", "expertqa", "factscore",
         "med_quad", "pubmed_qa", "samsum", "xsum"]
ID = "ID"
SHIFTED = ["SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"]
RUNGS = [ID] + ["SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"]

# Display order and labels. The keys are the implementation names in the tables; the labels are the
# names used in the write-up. src/luq/method_names.py holds the full mapping.
COMMON = [("saplma", "SAPLMA"),
          ("wmsp_shrink2", "CAWSA lambda = 2"),
          ("attention", "Attention pooling"),
          ("uniform", "Mean pooling"),
          ("floor_min", "Minimum token probability")]
POPULATIONS = ["meta-llama/Meta-Llama-3.1-8B", "Qwen/Qwen2.5-14B", "google/gemma-2-9b"]
SHORT = {"meta-llama/Meta-Llama-3.1-8B": "Llama-3.1-8B",
         "Qwen/Qwen2.5-14B": "Qwen2.5-14B",
         "google/gemma-2-9b": "Gemma-2-9B"}


def load(directory, pattern):
    """Return {population: {(method, rung, eval): prr}} from every table in `directory`.

    A population is identified by the `model` column rather than by the filename, so the same code
    reads the published directory and the working one.
    """
    paths = sorted(globmod.glob(os.path.join(directory, pattern)))
    if not paths:
        raise SystemExit(f"no tables matched {pattern!r} in {directory!r}")
    out = {}
    for path in paths:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            model = row.get("model", "").strip()
            value = row.get("prr_mean", "")
            if not model or value in ("", "nan"):
                continue
            out.setdefault(model, {})[(row["method"], row["rung"], row["eval"])] = float(value)
    missing = [p for p in POPULATIONS if p not in out]
    if missing:
        raise SystemExit(f"tables in {directory!r} do not cover {missing}")
    return out


def cell(table, method, rung):
    """The eight-dataset mean for one method and rung. Raises on a gap rather than averaging over
    whatever happens to be present, because a partial mean and a complete one must not look alike."""
    values = []
    for target in EVALS:
        key = (method, rung, target)
        if key not in table:
            raise SystemExit(f"missing cell: {method} / {rung} / {target}")
        values.append(table[key])
    return float(np.mean(values))


def per_dataset_shifted(table, method):
    """One value per dataset: the mean over the four shifted settings."""
    return np.array([np.mean([table[(method, r, t)] for r in SHIFTED]) for t in EVALS])


def paired(delta):
    """Exact two-sided Wilcoxon signed-rank test on eight paired differences, with the descriptive
    summary that must be read alongside it. At n = 8 the test has little power, so the mean
    difference and the count of datasets favouring each method carry most of the information."""
    delta = np.asarray(delta, float)
    if np.allclose(delta, 0):
        return delta.mean(), int((delta > 0).sum()), 1.0
    p = float(stats.wilcoxon(delta, alternative="two-sided",
                             zero_method="wilcox", mode="exact").pvalue)
    return float(delta.mean()), int((delta > 0).sum()), p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="published_results",
                    help="directory holding the per-population tables")
    ap.add_argument("--glob", default="*.csv", help="filename pattern within that directory")
    ap.add_argument("--ensemble", action="append", default=[],
                    help="a combination result file, repeatable, one per population. Section 5.7 "
                         "cannot be computed from the masters, which carry no combination rows.")
    args = ap.parse_args()

    data = load(args.dir, args.glob)
    print(f"tables read from {args.dir!r}\n")

    # ---- Table 5.1 -----------------------------------------------------------------------
    print("Table 5.1, the main comparison")
    head = f"{'model':14s} {'method':28s}" + "".join(f"{r.replace('-long',''):>12s}" for r in RUNGS) \
           + f"{'mean shifted':>14s}"
    print(head)
    print("  values at four decimals: the write-up prints three, and a value sitting on a half-way\n"
          "  point can round either way, which is a typographic difference and not a disagreement")
    for pop in POPULATIONS:
        for key, label in COMMON:
            row = [cell(data[pop], key, r) for r in RUNGS]
            shifted = float(np.mean([cell(data[pop], key, r) for r in SHIFTED]))
            print(f"{SHORT[pop]:14s} {label:28s}"
                  + "".join(f"{v:12.4f}" for v in row) + f"{shifted:14.4f}")
    print()

    # ---- Table 5.2 -----------------------------------------------------------------------
    print("Table 5.2, the two cross-task settings")
    print(f"{'model':14s} {'setting':16s}{'CAWSA':>10s}{'SAPLMA':>10s}{'difference':>13s}")
    for pop in POPULATIONS:
        for rung in ["DiffTask-long", "1ds-Diff-long"]:
            c = cell(data[pop], "wmsp_shrink2", rung)
            s = cell(data[pop], "saplma", rung)
            print(f"{SHORT[pop]:14s} {rung.replace('-long',''):16s}"
                  f"{c:10.4f}{s:10.4f}{c - s:+13.4f}")
    print()

    # ---- Section 5.1, the equal-model averages -------------------------------------------
    print("Section 5.1, equal-model averages across the three populations")
    for key, label in [("wmsp_shrink2", "CAWSA lambda = 2"),
                       ("floor_min", "Minimum token probability"),
                       ("floor_ppl", "Mean token NLL"),
                       ("floor_sum", "Sum NLL")]:
        row = [float(np.mean([cell(data[p], key, r) for p in POPULATIONS])) for r in RUNGS]
        print(f"  {label:28s}" + "".join(f"{v:12.4f}" for v in row))

    deltas = np.array([
        np.mean([np.mean([data[p][("wmsp_shrink2", r, t)] for r in SHIFTED]) for p in POPULATIONS])
        - np.mean([np.mean([data[p][("floor_min", r, t)] for r in SHIFTED]) for p in POPULATIONS])
        for t in EVALS])
    mean, wins, p = paired(deltas)
    print(f"\n  CAWSA over minimum token probability, shifted settings, dataset as the unit:")
    print(f"    mean {mean:+.4f}   higher on {wins} of 8   exact Wilcoxon p = {p:.4f}\n")

    # ---- Section 5.2, the dataset-level cross-task comparison ----------------------------
    deltas = np.array([
        np.mean([data[p][("wmsp_shrink2", "DiffTask-long", t)] for p in POPULATIONS])
        - np.mean([data[p][("saplma", "DiffTask-long", t)] for p in POPULATIONS])
        for t in EVALS])
    mean, wins, p = paired(deltas)
    print("Section 5.2, CAWSA against SAPLMA at DiffTask, dataset as the unit")
    print(f"    mean {mean:+.4f}   CAWSA higher on {wins} of 8   exact Wilcoxon p = {p:.4f}\n")

    # ---- Figure 5.4, the shrinkage gains --------------------------------------------------
    print("Figure 5.4, shrinkage gain at DiffTask, lambda = 2 against the lambda = 0 control")
    for pop in POPULATIONS:
        gain = cell(data[pop], "wmsp_shrink2", "DiffTask-long") \
               - cell(data[pop], "wmsp_norm", "DiffTask-long")
        print(f"  {SHORT[pop]:14s} {gain:+.4f}")
    print()

    # ---- Section 7.1, the equal-model advantage -------------------------------------------
    print("Section 7.1, SAPLMA advantage over CAWSA, equal-model average")
    for rung in [ID, "DiffTask-long", "1ds-Diff-long"]:
        adv = float(np.mean([cell(data[p], "saplma", rung) for p in POPULATIONS])) \
              - float(np.mean([cell(data[p], "wmsp_shrink2", rung) for p in POPULATIONS]))
        print(f"  {rung.replace('-long',''):16s} {adv:+.4f}")
    print()

    # ---- Section 5.7, the combination -----------------------------------------------------
    # Reported on the two populations with complete clean per-response inputs, so this reads the
    # combination result files rather than the masters, which carry no combination rows.
    print("Section 5.7 and Table 5.3, the combination, equal-model over the two clean populations")
    if not args.ensemble:
        print("  not computed. Pass --ensemble once per population. The field is left blank rather\n"
              "  than filled from another source.")
    else:
        prof, stat = {}, {}
        for path in args.ensemble:
            for r in csv.DictReader(open(path, newline="")):
                if r["section"] == "rung_profile" and r["value"]:
                    prof.setdefault((r["method"], r["rung"]), []).append(float(r["value"]))
                if r["section"] == "estimand_A" and r.get("macro"):
                    stat.setdefault(r["method"], []).append(r)
        order = ["CAWSA \u03bb=2", "SAPLMA", "CONTROL  SAPLMA + attention-pool",
                 "PRIMARY  CAWSA \u03bb=2 + SAPLMA", "REF      msp_min + SAPLMA"]
        print(f"  {'method':34s}" + "".join(f"{r.replace('-long',''):>11s}" for r in RUNGS)
              + f"{'mean OOD':>11s}")
        for meth in order:
            vals = [prof.get((meth, r)) for r in RUNGS]
            if any(v is None for v in vals):
                print(f"  {meth:34s}  not present in the files given"); continue
            means = [sum(v) / len(v) for v in vals]
            print(f"  {meth:34s}" + "".join(f"{v:11.4f}" for v in means)
                  + f"{sum(means[1:]) / 4:11.4f}")
        print()
        print("  against SAPLMA alone, out of distribution, dataset as the unit:")
        for meth in ["PRIMARY  CAWSA \u03bb=2 + SAPLMA", "REF      msp_min + SAPLMA"]:
            for r in stat.get(meth, []):
                print(f"    {meth:34s} macro {float(r['macro']):+.4f}  "
                      f"{r['signs_positive']} of {r['n']}  p = {float(r['wilcoxon_p']):.4f}")
        print()
        print("  The reference combination of minimum token probability with the probe is printed\n"
              "  deliberately. Its mean out-of-distribution score is close to, and can exceed, the\n"
              "  reported combination, so it should be read next to its in-distribution behaviour\n"
              "  rather than on the shifted settings alone.")
    print()


if __name__ == "__main__":
    sys.exit(main())
