#!/usr/bin/env python
"""Summarise the like-for-like probability-aggregation comparison across whichever models have a
`results/hybrids/likeforlike__<slug>.csv` on disk (built by `likeforlike_table.py`).

Reads only what is already written; does not compute anything itself. Prints, per model:
  - macro PRR for each fixed weighting rule and for CAWSA at every rung, mean OOD
  - per-dataset PRR for every method (the "per-target" breakdown)
and, if more than one model is present, an equal-weight average across models.

    python scripts/checks/likeforlike_multimodel_summary.py
    python scripts/checks/likeforlike_multimodel_summary.py --models meta-llama/Meta-Llama-3.1-8B Qwen/Qwen2.5-14B
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache  # noqa: E402

FIXED = ["sequence NLL (published MSP)", "mean token NLL", "minimum token probability",
         "TokenSAR", "answer-span mean NLL", "answer-span sequence NLL"]
OOD_RUNGS = ["LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
CAWSA_RUNGS = ["ID"] + OOD_RUNGS


def load(model):
    slug = cache._slug(model)
    path = ROOT / "results" / "hybrids" / f"likeforlike__{slug}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    datasets = sorted(set(r["dataset"] for r in rows))
    return {"rows": rows, "datasets": datasets, "path": path}


def macro(rows, method):
    vals = [float(r["prr"]) for r in rows if r["method"] == method]
    return sum(vals) / len(vals) if vals else None, len(vals)


def per_dataset(rows, method):
    return {r["dataset"]: float(r["prr"]) for r in rows if r["method"] == method}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                     default=["meta-llama/Meta-Llama-3.1-8B", "Qwen/Qwen2.5-14B", "google/gemma-2-9b"])
    args = ap.parse_args()

    loaded = {}
    for m in args.models:
        d = load(m)
        if d is None:
            print(f"-- {m}: no results/hybrids/likeforlike__{cache._slug(m)}.csv on disk, skipping --")
            continue
        loaded[m] = d

    if not loaded:
        sys.exit("no like-for-like result files found for any requested model.")

    macro_by_model = {}
    for m, d in loaded.items():
        print(f"\n=== {m}  (datasets: {', '.join(d['datasets'])}, n={len(d['datasets'])}) ===")
        macros = {}
        for meth in FIXED:
            v, n = macro(d["rows"], meth)
            macros[meth] = v
            print(f"  {meth:35s} macro={v:+.4f}  (n_datasets={n})" if v is not None else f"  {meth}: MISSING")
        for rg in CAWSA_RUNGS:
            key = f"CAWSA lambda=2 [{rg}]"
            v, n = macro(d["rows"], key)
            macros[key] = v
            print(f"  {key:35s} macro={v:+.4f}  (n_datasets={n})" if v is not None else f"  {key}: MISSING")
        ood = [macros[f"CAWSA lambda=2 [{rg}]"] for rg in OOD_RUNGS if macros.get(f"CAWSA lambda=2 [{rg}]") is not None]
        if len(ood) == len(OOD_RUNGS):
            mean_ood = sum(ood) / len(ood)
            macros["CAWSA mean OOD"] = mean_ood
            print(f"  {'CAWSA mean OOD':35s} macro={mean_ood:+.4f}")
        macro_by_model[m] = macros

        print("\n  per-target PRR:")
        header = "    " + "dataset".ljust(14) + "".join(f"{meth[:16]:>18s}" for meth in FIXED)
        print(header)
        for ds in d["datasets"]:
            line = "    " + ds.ljust(14)
            for meth in FIXED:
                pv = per_dataset(d["rows"], meth).get(ds)
                line += f"{pv:>+18.4f}" if pv is not None else f"{'--':>18s}"
            print(line)

    if len(macro_by_model) > 1:
        print(f"\n=== EQUAL-MODEL AVERAGE across {len(macro_by_model)} models "
              f"({', '.join(macro_by_model)}) ===")
        all_keys = FIXED + [f"CAWSA lambda=2 [{rg}]" for rg in CAWSA_RUNGS] + ["CAWSA mean OOD"]
        for k in all_keys:
            vs = [macro_by_model[m][k] for m in macro_by_model if macro_by_model[m].get(k) is not None]
            if len(vs) == len(macro_by_model):
                per_model_str = ", ".join(
                    f"{m.split('/')[-1]}={macro_by_model[m][k]:+.4f}" for m in macro_by_model)
                print(f"  {k:35s} avg={sum(vs)/len(vs):+.4f}   (per-model: {per_model_str})")
            else:
                print(f"  {k:35s} incomplete across models ({len(vs)}/{len(macro_by_model)}), no average taken")


if __name__ == "__main__":
    main()
