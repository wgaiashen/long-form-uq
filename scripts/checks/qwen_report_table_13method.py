#!/usr/bin/env python
"""Assemble the 13-method Qwen report table.

One-off assembly script for the final-report table, NOT a modification of
qwen_pdl_master_clean.py (whose 9-method list stays exactly what the invariance sweep and M2
rescoring were verified against). Pulls together four separately-computed sources:

  1-9   the clean-population ladder master  (qwen_pdl_master_clean.py's output)
  10    Lehmer beta=1, clean population     (lehmer_qwen.py --out ..., reshaped: it is one row per
                                             dataset with no rung column because the score is
                                             training-free and therefore rung-invariant -- same
                                             reasoning as the floors, confirmed from that script's
                                             own docstring; broadcasting it to all 5 rungs here
                                             matches how those floors already appear in the master)
  11    P(True), training-free              (score_ptrue_unsup.py --model Qwen/Qwen2.5-14B output;
                                             also rung-invariant by the same construction, and it
                                             already emits all 5 rungs itself)
  12-13 P(True) probe + Lookback            (a --baselines ptrue,lookback ladder pass, once it
                                             exists -- ABSENT stays absent, never zero, if the file
                                             is not there yet)

Primary numeric column is the cross-dataset OOD PRR mean: average the 4 OOD rungs within an eval
FIRST, then average over the 8 evals (n=8, not 32 cells) -- the same convention
qwen_pdl_master.py's own "Cross-dataset OOD aggregate" section uses, for the same reason (floors
would otherwise be counted 4x, since their PRR does not depend on the training pool).

    python scripts/checks/qwen_report_table_13method.py
"""
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
SLUG = "Qwen_Qwen2.5-14B"

EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
ALL_RUNGS = ["ID"] + OOD_RUNGS

# (row label, code method name(s) to look up, primary signal, supervision, response aggregation)
SPEC = [
    ("Sequence negative log-likelihood", "floor_sum", "Token probability", "None", "Sum"),
    ("Mean token NLL", "floor_ppl", "Token probability", "None", "Mean"),
    ("Minimum token probability", "floor_min", "Token probability", "None", "Extreme token"),
    ("SAPLMA", "saplma", "Hidden state", "Graded labels", "Mean pooling and MLP"),
    ("Mean-pool probe", "uniform", "Hidden state", "Graded labels", "Mean pooling and linear probe"),
    ("Learned attention-pool probe", "attention", "Hidden state", "Graded labels", "Learned position weights"),
    ("Normalised wMSP", "wmsp_norm", "Hidden state and token NLL", "Graded labels", "Learned NLL weighting"),
    ("wMSP shrinkage 2", "wmsp_shrink2", "Hidden state and token NLL", "Graded labels", "Regularised learned NLL weighting"),
    ("wMSP shrinkage 10", "wmsp_shrink10", "Hidden state and token NLL", "Graded labels", "Stronger uniform regularisation"),
    ("Fixed Lehmer β = 1", "lehmer_b1", "Token probability", "None", "Fixed intermediate aggregation"),
    ("P(True), training-free", "ptrue_unsup", "Self-evaluation probability", "None", "Single verification position"),
    ("P(True) verdict-state probe", "ptrue", "Verification hidden state", "Graded labels", "Single verification position"),
    ("Lookback Lens", "lookback", "Attention", "Graded labels", "Response-averaged attention features"),
]


def load_master_rows():
    """(method, eval, rung) -> prr_mean, from the 9-method clean master."""
    out = {}
    p = ANALYSIS / f"pdl_master_qwenclean__{SLUG}.csv"
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        if r["prr_mean"] in ("", None):
            continue
        out[(r["method"], r["eval"], r["rung"])] = float(r["prr_mean"])
    return out


def load_lehmer_b1():
    """(eval,) -> prr, beta=1 only, from the clean-population Lehmer recompute."""
    out = {}
    p = ANALYSIS / f"lehmer_qwen_clean__{SLUG}.csv"
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        if r.get("dataset") in EVALS and r.get("beta") == "1.0":
            out[r["dataset"]] = float(r["prr"])
    return out


def load_ptrue_unsup():
    """(eval, rung) -> prr, from score_ptrue_unsup.py's Qwen output (already 5-rung broadcast)."""
    out = {}
    p = RESULTS / f"pdl_fam_ptrueunsup__{SLUG}.csv"
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        if r["method"] == "ptrue_unsup" and r["eval"] in EVALS:
            out[(r["eval"], r["rung"])] = float(r["prr_mean"])
    return out


def load_baselines():
    """(method, eval, rung) -> prr, from the --baselines ptrue,lookback ladder pass, if it exists."""
    out = {}
    found_any = False
    for ev in EVALS:
        p = ANALYSIS / f"pdl_qwenclean_baselines_{ev}__{SLUG}.csv"
        if not p.exists():
            continue
        found_any = True
        for r in csv.DictReader(open(p)):
            if r["method"] in ("ptrue", "lookback") and r["prr_mean"] not in ("", None):
                out[(r["method"], r["eval"], r["rung"])] = float(r["prr_mean"])
    return out, found_any


def ood_mean_for(get_value, method_key):
    """Cross-dataset OOD PRR mean (n=8): average the 4 OOD rungs per eval first, then the 8 evals."""
    per_ds = []
    for ev in EVALS:
        vals = [get_value(method_key, ev, rg) for rg in OOD_RUNGS]
        vals = [v for v in vals if v is not None]
        if vals:
            per_ds.append(mean(vals))
    if len(per_ds) < len(EVALS):
        return None, len(per_ds)
    return mean(per_ds), len(per_ds)


def id_value_for(get_value, method_key):
    per_ds = [get_value(method_key, ev, "ID") for ev in EVALS]
    per_ds = [v for v in per_ds if v is not None]
    if len(per_ds) < len(EVALS):
        return None, len(per_ds)
    return mean(per_ds), len(per_ds)


def main():
    master = load_master_rows()
    lehmer = load_lehmer_b1()
    ptrue_unsup = load_ptrue_unsup()
    baselines, baselines_exist = load_baselines()

    def get(method_key, ev, rung):
        if method_key == "lehmer_b1":
            return lehmer.get(ev)                       # rung-invariant, no rung lookup needed
        if method_key == "ptrue_unsup":
            return ptrue_unsup.get((ev, rung))
        if method_key in ("ptrue", "lookback"):
            return baselines.get((method_key, ev, rung))
        return master.get((method_key, ev, rung))

    rows = []
    for label, key, signal, supervision, aggregation in SPEC:
        ood, ood_n = ood_mean_for(get, key)
        idv, id_n = id_value_for(get, key)
        rows.append({
            "Method": label, "Primary signal": signal, "Supervision": supervision,
            "Response aggregation": aggregation,
            "Qwen ID PRR": f"{idv:+.4f}" if idv is not None else "",
            "Qwen OOD PRR (n=8)": f"{ood:+.4f}" if ood is not None else "",
            "coverage": f"{ood_n}/8 evals" if ood is not None else f"PENDING ({ood_n}/8 evals)",
        })

    print("=" * 120)
    print("QWEN 13-METHOD REPORT TABLE")
    print("=" * 120)
    if not baselines_exist:
        print("No --baselines ptrue,lookback ladder pass found yet -- methods 12-13 will read PENDING.")
    header = f"{'Method':32s}{'Signal':26s}{'Supervision':14s}{'Aggregation':38s}{'ID PRR':>9s}{'OOD PRR':>10s}  coverage"
    print(header)
    for r in rows:
        print(f"{r['Method']:32s}{r['Primary signal']:26s}{r['Supervision']:14s}"
              f"{r['Response aggregation']:38s}{r['Qwen ID PRR']:>9s}{r['Qwen OOD PRR (n=8)']:>10s}  {r['coverage']}")

    out_csv = ANALYSIS / f"qwen_report_table_13method__{SLUG}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_csv}")

    out_md = ANALYSIS / f"qwen_report_table_13method__{SLUG}.md"
    lines = ["| Method | Primary signal | Supervision | Response aggregation | Qwen ID PRR | Qwen OOD PRR (n=8) | Coverage |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['Method']} | {r['Primary signal']} | {r['Supervision']} | "
                     f"{r['Response aggregation']} | {r['Qwen ID PRR']} | {r['Qwen OOD PRR (n=8)']} | {r['coverage']} |")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
