#!/usr/bin/env python
"""Result tables read directly from the corrected-span master: transfer drop, pooling, floors.

Three of the mechanism-chapter analyses need no recomputation, because every quantity they report
is already a cell of the assembled corrected-span ladder. Reading them here rather than rerunning
a driver keeps them provably on the same population as the main results, and it keeps the
statistical conventions in one place (report_pack_stats: the dataset is the unit, n = 8, seeds
averaged inside a cell, exact two-sided Wilcoxon, bootstrap over datasets).

Outputs, all under results/analysis/:

  ch6_cleanv2_probe_drift.csv        macro PRR by rung for the supervised core estimators
  ch6_cleanv2_probe_drift_drop.csv   the same, expressed as the drop from the matched setting
  ch6_cleanv2_pooling_comparison.csv learned attention pooling against uniform hidden-state pooling
  ch6_cleanv2_probability_aggregates.csv  the three fixed probability aggregators, per dataset

TRAINING-FREE SCORES ARE NOT GIVEN A DROP ROW. Their score for a fixed response does not depend on
the labelled training pool, so their PRR is identical at all five rungs by construction and a
"drop" column would invite reading a transfer effect into an arithmetic identity. They appear in
the absolute table and in their own per-dataset table instead.

    python scripts/checks/ch6_cleanv2_pack.py
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from report_pack_stats import EVALS, OOD_RUNGS, mean_ood, paired, fmt_paired  # noqa: E402

SLUG = "meta-llama_Meta-Llama-3.1-8B"
MASTER = ROOT / "results" / "cleanv2" / f"pdl_cleanv2_master__{SLUG}.csv"
CONSOLIDATED = ROOT / "results" / "hybrids" / f"pdl_consolidated_master__{SLUG}.csv"
LIKEFORLIKE = ROOT / "results" / "hybrids" / f"likeforlike__{SLUG}.csv"
OUTDIR = ROOT / "results" / "analysis"
POPULATION = "meta-llama/Meta-Llama-3.1-8B, corrected span (med_quad), eight long-form targets"

RUNGS = ["ID"] + ["SameTask-long", "LOO-long", "DiffTask-long", "1ds-Diff-long"]

# Report-facing names for the estimators this pack reads. The keys are the identifiers the ladder
# writes; renaming those retroactively would break results provenance, so the mapping lives here.
SUPERVISED = [
    ("saplma", "SAPLMA"),
    ("uniform", "Mean hidden-state pooling"),
    ("attention", "Learned attention pooling"),
    ("wmsp_norm", "Activation-weighted surprisal at shrinkage 0"),
    ("wmsp_shrink1_5", "Activation-weighted surprisal at shrinkage 1.5"),
    ("wmsp_shrink2", "Activation-weighted surprisal at shrinkage 2"),
]
FLOORS = [
    ("floor_sum", "Sequence NLL"),
    ("floor_ppl", "Mean token NLL"),
    ("floor_min", "Minimum token probability"),
]
# Cross-check tolerances. The two masters publish the same cells and must agree; the like-for-like
# table recomputes its floors from the records, so it agrees to its own rounding, not to the bit.
TOL_MASTER = 5e-4
TOL_LIKEFORLIKE = 1e-3


def load():
    df = pd.read_csv(MASTER)
    df = df[~df.method.astype(str).str.startswith("VERDICT")].copy()
    df["prr_mean"] = df["prr_mean"].astype(float)
    return df


def macro_by_rung(df, method):
    """Mean PRR over the eight targets, one value per rung. Fails loudly on a gap."""
    out = {}
    for rung in RUNGS:
        s = df[(df.method == method) & (df.rung == rung)].set_index("eval")["prr_mean"]
        missing = [e for e in EVALS if e not in s.index]
        if missing:
            raise SystemExit(f"{method} at {rung}: missing {missing}; refusing to average a partial grid")
        out[rung] = float(s.loc[EVALS].mean())
    out["mean OOD"] = float(sum(out[r] for r in OOD_RUNGS) / len(OOD_RUNGS))
    return out


def gate_against_consolidated(df):
    """The two report-facing masters publish the same cells; disagreement means one is stale."""
    if not CONSOLIDATED.exists():
        print("  consolidated master absent; cross-check skipped")
        return
    other = pd.read_csv(CONSOLIDATED)
    other = other[other.model == "meta-llama/Meta-Llama-3.1-8B"]
    merged = df.merge(other, on=["eval", "rung", "method"], suffixes=("_cv2", "_con"))
    if merged.empty:
        raise SystemExit("cross-check found no shared cells; the two masters do not line up")
    d = (merged.prr_mean_cv2 - merged.prr_mean_con.astype(float)).abs()
    worst = merged.loc[d.idxmax()]
    ok = d.max() <= TOL_MASTER
    print(f"  [{'PASS' if ok else 'FAIL'}] agrees with the consolidated master on "
          f"{len(merged)} shared cells, max abs diff {d.max():.2e} "
          f"(worst {worst.method} / {worst.eval} / {worst.rung})")
    if not ok:
        raise SystemExit("the two masters disagree beyond tolerance; stop and reconcile them")


def gate_floors_against_likeforlike(per_dataset):
    """The like-for-like table recomputes the same three floors from the records."""
    if not LIKEFORLIKE.exists():
        print("  like-for-like table absent; floor cross-check skipped")
        return
    lfl = pd.read_csv(LIKEFORLIKE)
    names = {"floor_sum": "sequence NLL (published MSP)",
             "floor_ppl": "mean token NLL",
             "floor_min": "minimum token probability"}
    worst, where = 0.0, None
    for key, lfl_name in names.items():
        sub = lfl[lfl.method == lfl_name].set_index("dataset")["prr"].astype(float)
        for ev in EVALS:
            diff = abs(per_dataset[key][ev] - float(sub.loc[ev]))
            if diff > worst:
                worst, where = diff, f"{key} / {ev}"
    ok = worst <= TOL_LIKEFORLIKE
    print(f"  [{'PASS' if ok else 'FAIL'}] the three floors agree with the like-for-like table, "
          f"max abs diff {worst:.2e} ({where})")
    if not ok:
        raise SystemExit("floor values disagree with the independently computed table")


def write(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load()

    print("=" * 92)
    print("TABLES READ FROM THE CORRECTED-SPAN MASTER")
    print(f"population: {POPULATION}")
    print("=" * 92)
    print("\nGates")
    gate_against_consolidated(df)

    # ---- absolute macro by rung, and the drop from the matched setting -----------------------
    print("\nTransfer behaviour of the supervised estimators")
    abs_rows, drop_rows = [], []
    for key, name in SUPERVISED + FLOORS:
        m = macro_by_rung(df, key)
        abs_rows.append({"method_key": key, "method": name,
                         "fitted_on_source_pool": key not in dict(FLOORS),
                         **{r: round(m[r], 4) for r in RUNGS}, "mean OOD": round(m["mean OOD"], 4),
                         "n_datasets": len(EVALS), "population": POPULATION})
        if key in dict(FLOORS):
            continue
        drop_rows.append({"method_key": key, "method": name,
                          "ID": round(m["ID"], 4),
                          **{f"drop to {r}": round(m["ID"] - m[r], 4) for r in RUNGS[1:]},
                          "drop to mean OOD": round(m["ID"] - m["mean OOD"], 4),
                          "n_datasets": len(EVALS), "population": POPULATION})
        print(f"  {name:52s} ID {m['ID']:+.4f}  mean OOD {m['mean OOD']:+.4f}  "
              f"drop {m['ID'] - m['mean OOD']:+.4f}")

    write(OUTDIR / "ch6_cleanv2_probe_drift.csv",
          ["method_key", "method", "fitted_on_source_pool"] + RUNGS
          + ["mean OOD", "n_datasets", "population"], abs_rows)
    write(OUTDIR / "ch6_cleanv2_probe_drift_drop.csv",
          ["method_key", "method", "ID"] + [f"drop to {r}" for r in RUNGS[1:]]
          + ["drop to mean OOD", "n_datasets", "population"], drop_rows)

    # ---- the three fixed probability aggregators, per dataset --------------------------------
    print("\nFixed probability aggregation, per target")
    per_dataset = {}
    agg_rows = []
    for key, name in FLOORS:
        s = df[(df.method == key) & (df.rung == "ID")].set_index("eval")["prr_mean"]
        per_dataset[key] = {e: float(s.loc[e]) for e in EVALS}
    for ev in EVALS:
        row = {"dataset": ev, "population": POPULATION}
        for key, name in FLOORS:
            row[name] = round(per_dataset[key][ev], 4)
        row["min minus mean"] = round(per_dataset["floor_min"][ev] - per_dataset["floor_ppl"][ev], 4)
        agg_rows.append(row)
        print(f"  {ev:14s} sum {row['Sequence NLL']:+.4f}  mean {row['Mean token NLL']:+.4f}  "
              f"min {row['Minimum token probability']:+.4f}  "
              f"min-mean {row['min minus mean']:+.4f}")
    macro = {"dataset": "MACRO (eight targets)", "population": POPULATION}
    for key, name in FLOORS:
        macro[name] = round(sum(per_dataset[key].values()) / len(EVALS), 4)
    macro["min minus mean"] = round(macro["Minimum token probability"] - macro["Mean token NLL"], 4)
    agg_rows.append(macro)
    print(f"  {'MACRO':14s} sum {macro['Sequence NLL']:+.4f}  mean {macro['Mean token NLL']:+.4f}  "
          f"min {macro['Minimum token probability']:+.4f}")

    print("\nGate")
    gate_floors_against_likeforlike(per_dataset)
    write(OUTDIR / "ch6_cleanv2_probability_aggregates.csv",
          ["dataset"] + [n for _, n in FLOORS] + ["min minus mean", "population"], agg_rows)

    # ---- learned attention pooling against uniform hidden-state pooling ----------------------
    print("\nLearned attention pooling against uniform hidden-state pooling")
    id_a = df[(df.method == "attention") & (df.rung == "ID")].set_index("eval")["prr_mean"]
    id_u = df[(df.method == "uniform") & (df.rung == "ID")].set_index("eval")["prr_mean"]
    ood_a, ood_u = mean_ood(df, "attention"), mean_ood(df, "uniform")
    d_id = [float(id_a.loc[e] - id_u.loc[e]) for e in EVALS]
    d_ood = [float(ood_a.loc[e] - ood_u.loc[e]) for e in EVALS]
    r_id, r_ood = paired(d_id), paired(d_ood)
    print("  " + fmt_paired("ID delta", r_id).replace("\n", "\n  "))
    print("  " + fmt_paired("mean OOD delta", r_ood).replace("\n", "\n  "))

    pool_rows = []
    for ev in EVALS:
        pool_rows.append({"scope": "per dataset", "dataset": ev,
                          "attention ID": round(float(id_a.loc[ev]), 4),
                          "uniform ID": round(float(id_u.loc[ev]), 4),
                          "ID delta": round(float(id_a.loc[ev] - id_u.loc[ev]), 4),
                          "attention mean OOD": round(float(ood_a.loc[ev]), 4),
                          "uniform mean OOD": round(float(ood_u.loc[ev]), 4),
                          "mean OOD delta": round(float(ood_a.loc[ev] - ood_u.loc[ev]), 4),
                          "population": POPULATION})
    ma, mu = macro_by_rung(df, "attention"), macro_by_rung(df, "uniform")
    for rung in RUNGS:
        pool_rows.append({"scope": "macro trajectory", "dataset": rung,
                          "attention ID": "", "uniform ID": "", "ID delta": "",
                          "attention mean OOD": round(ma[rung], 4),
                          "uniform mean OOD": round(mu[rung], 4),
                          "mean OOD delta": round(ma[rung] - mu[rung], 4),
                          "population": POPULATION})
    for label, r in (("ID", r_id), ("mean OOD", r_ood)):
        pool_rows.append({"scope": f"paired test, {label}", "dataset": f"n = {r['n']} datasets",
                          "attention ID": "", "uniform ID": "", "ID delta": "",
                          "attention mean OOD": f"macro {r['macro_mean']:+.4f}",
                          "uniform mean OOD": f"{r['n_positive']}/{r['n']} positive",
                          "mean OOD delta": f"exact Wilcoxon p={r['wilcoxon_p_exact']:.4f}, "
                                            f"boot95% [{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}]",
                          "population": POPULATION})
    write(OUTDIR / "ch6_cleanv2_pooling_comparison.csv",
          ["scope", "dataset", "attention ID", "uniform ID", "ID delta",
           "attention mean OOD", "uniform mean OOD", "mean OOD delta", "population"], pool_rows)

    print("\nAll gates passed.")


if __name__ == "__main__":
    main()
