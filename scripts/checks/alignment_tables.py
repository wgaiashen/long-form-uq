#!/usr/bin/env python
"""W-Align -- build the comparison tables, paired statistics and rung profile from the ablation output.

Results: ../STOCKTAKE_alignment.md.  Driver that produced the inputs: token_state_alignment.py.

⚠️ POST-HOC SENSITIVITY ANALYSIS. These statistics are descriptive; they are NOT pre-registered
hypothesis tests, and no alignment is selected on them.

WHAT THIS READS
---------------
results/sensitivity/token_state_alignment/alignment_prr_<eval>__<slug>.csv, one per eval, each holding
  * per-seed rows           (seed = 1/2/3)          <- PRIMARY, everything here is recomputed from these
  * seed-mean rows          (seed = "mean")         <- convenience; cross-checked against the above
  * CONTROL2_vs_master rows (alignment column)      <- the gate, re-asserted here
  * CONTROL3_distinctness rows                      <- the gate, re-asserted here

⚠️ MEAN OVER SEEDS OF THE PER-SEED PRR -- never the PRR of a seed-averaged score. Those differ, and the
second is systematically higher because averaging cancels seed noise (measured elsewhere in this project
at +0.050 for SAPLMA at LOO-long). The driver stores per-seed PRRs, so this file just averages them.

UNIT OF ANALYSIS = THE DATASET, n = 8. The 32 OOD cells are not 32 independent observations.

    python scripts/checks/alignment_tables.py
"""
import argparse
import csv as _csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from complementary_ensemble import wilcoxon_exact, boot_ci_mean   # noqa: E402  (shared, light imports)

SLUG = "meta-llama_Meta-Llama-3.1-8B"
INDIR = ROOT / "results" / "sensitivity" / "token_state_alignment"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
# ⚠️ report / Hidden Failures rung order. LOO before SameTask. Do not silently reorder.
RUNGS = ["ID", "LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
OOD = RUNGS[1:]
METHODS = ["HAPE", "HAPES"]
ALIGN = ["post_token", "pre_token"]
GATE_4DP = 1e-4


def load():
    """{(method, alignment, dataset, rung): mean-over-seeds PRR}, plus gates and coverage."""
    prr = defaultdict(list)          # per-seed values
    stated_mean, gate2, gate3, n_eval = {}, [], [], {}
    files = sorted(p for p in INDIR.glob(f"alignment_prr_*__{SLUG}.csv") if "SMOKE" not in p.name)
    for p in files:
        for r in _csv.DictReader(open(p)):
            key = (r["method"], r["alignment"], r["dataset"], r["rung"])
            if r["alignment"] == "CONTROL2_vs_master":
                gate2.append((r["dataset"], r["rung"], r["method"], float(r["prr"])))
            elif r["alignment"] == "CONTROL3_distinctness":
                gate3.append((r["dataset"], r["rung"], r["method"], float(r["prr"])))
            elif r["seed"] == "mean":
                stated_mean[key] = float(r["prr"])
            else:
                prr[key].append(float(r["prr"]))
                n_eval[(r["dataset"], r["rung"])] = int(r["n_eval"])
    mean = {k: float(np.mean(v)) for k, v in prr.items()}
    nseeds = {k: len(v) for k, v in prr.items()}
    return mean, nseeds, stated_mean, gate2, gate3, n_eval, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(INDIR))
    args = ap.parse_args()

    mean, nseeds, stated, gate2, gate3, n_eval, files = load()
    print("=" * 100)
    print("W-Align -- post_token (l_t<->h_t, current) vs pre_token (l_t<->h_(t-1))")
    print("POST-HOC SENSITIVITY ANALYSIS. Unit of analysis = DATASET, n = 8.")
    print(f"Read {len(files)} per-eval files from {args.indir}")
    print("=" * 100)

    # ---------------------------------------------------------------- coverage
    have = sorted({d for (_m, _a, d, _r) in mean})
    missing = [f"{d}/{rg}/{m}/{a}" for d in LONG for rg in RUNGS for m in METHODS for a in ALIGN
               if (m, a, d, rg) not in mean]
    n_cells = len({(d, rg) for (_m, _a, d, rg) in mean})
    print(f"\nCOVERAGE: {len(have)}/8 datasets, {n_cells}/40 dataset x rung cells, "
          f"{len(mean)}/160 method x alignment x cell combinations")
    if missing:
        print(f"⚠️ INCOMPLETE -- {len(missing)} combinations absent. Datasets present: {have}")
        print("   A partial grid is never reported as if it were the whole one.")
    bad_seeds = {k: v for k, v in nseeds.items() if v != 3}
    if bad_seeds:
        print(f"⚠️ {len(bad_seeds)} combinations do not have 3 seeds: "
              f"{list(bad_seeds.items())[:5]}")

    # ---------------------------------------------------------------- gates re-asserted
    g2_bad = [g for g in gate2 if abs(g[3]) > GATE_4DP]
    g3_bad = [g for g in gate3 if g[3] == 0.0]
    print(f"\nCONTROL 2 (post_token reproduces canonical master, bar {GATE_4DP:g}): "
          f"{'✅ PASS' if not g2_bad else '❌ FAIL'} "
          f"({len(gate2)} checks, max |Δ| = {max((abs(g[3]) for g in gate2), default=float('nan')):.2e})")
    for g in g2_bad[:10]:
        print(f"    ✗ {g[0]}/{g[1]}/{g[2]}: Δ {g[3]:+.2e}")
    print(f"CONTROL 3 (arms differ): {'✅ PASS' if not g3_bad else '❌ FAIL'} ({len(gate3)} checks)")
    for g in g3_bad[:10]:
        print(f"    ✗ {g[0]}/{g[1]}/{g[2]}: max|q_pre - q_post| = 0")
    # cross-check the recomputed means against the driver's own stated means
    dmax = max((abs(mean[k] - stated[k]) for k in mean if k in stated), default=0.0)
    print(f"seed-mean cross-check vs the driver's stated means: max |Δ| = {dmax:.2e} "
          f"{'✅' if dmax < 1e-6 else '❌'}")
    if g2_bad or g3_bad:
        print("\n⛔ A GATE FAILED -- the pre_token numbers are not interpretable. Stopping.")
        return

    # ---------------------------------------------------------------- rung profile
    print("\n" + "-" * 100)
    print("RUNG PROFILE -- macro mean PRR over the datasets present, report rung order")
    print("-" * 100)
    print(f"{'method':10s}{'alignment':13s}" + "".join(f"{r.replace('-long',''):>13s}" for r in RUNGS)
          + f"{'OODmacro':>12s}")
    prof_rows = []
    for m in METHODS:
        for a in ALIGN:
            vals = []
            for rg in RUNGS:
                v = [mean[(m, a, d, rg)] for d in have if (m, a, d, rg) in mean]
                vals.append(float(np.mean(v)) if v else float("nan"))
            ood = float(np.nanmean(vals[1:]))
            print(f"{m:10s}{a:13s}" + "".join(f"{v:>+13.4f}" for v in vals) + f"{ood:>+12.4f}")
            for rg, v in zip(RUNGS, vals):
                prof_rows.append({"section": "rung_profile", "method": m, "alignment": a,
                                  "rung": rg, "value": round(v, 6)})
            prof_rows.append({"section": "rung_profile", "method": m, "alignment": a,
                              "rung": "OOD_macro", "value": round(ood, 6)})
    # the delta view, which is what the question is actually about
    print(f"\n{'method':10s}{'delta (pre - post)':13s}" +
          "".join(f"{r.replace('-long',''):>13s}" for r in RUNGS) + f"{'OODmacro':>12s}")
    for m in METHODS:
        vals = []
        for rg in RUNGS:
            v = [mean[(m, "pre_token", d, rg)] - mean[(m, "post_token", d, rg)]
                 for d in have if (m, "pre_token", d, rg) in mean and (m, "post_token", d, rg) in mean]
            vals.append(float(np.mean(v)) if v else float("nan"))
        print(f"{m:10s}{'':13s}" + "".join(f"{v:>+13.4f}" for v in vals)
              + f"{float(np.nanmean(vals[1:])):>+12.4f}")

    # ---------------------------------------------------------------- per-dataset + paired stats
    summary_rows, cmp_md = [], {}
    print("\n" + "=" * 100)
    print("PAIRED SUMMARY -- per dataset, mean PRR over its 4 OOD rungs; unit = dataset")
    print("=" * 100)
    for m in METHODS:
        per_ds = []
        for d in have:
            po = [mean[(m, "post_token", d, rg)] for rg in OOD if (m, "post_token", d, rg) in mean]
            pr = [mean[(m, "pre_token", d, rg)] for rg in OOD if (m, "pre_token", d, rg) in mean]
            if len(po) == len(OOD) and len(pr) == len(OOD):
                per_ds.append((d, float(np.mean(po)), float(np.mean(pr))))
        if not per_ds:
            print(f"\n### {m}: no dataset has all four OOD rungs yet"); continue
        dv = np.array([pr - po for _d, po, pr in per_ds])
        lo, hi = boot_ci_mean(dv)
        p, n_nz = wilcoxon_exact(dv)
        pos = int(np.sum(dv > 0))
        print(f"\n### {m}   (n = {len(per_ds)} datasets)")
        print(f"{'dataset':16s}{'post OOD':>11s}{'pre OOD':>11s}{'delta':>11s}")
        lines = [f"| dataset | post-token mean OOD PRR | pre-token mean OOD PRR | delta (pre − post) |",
                 "|---|---|---|---|"]
        for d, po, pr in per_ds:
            print(f"{d:16s}{po:>+11.4f}{pr:>+11.4f}{pr-po:>+11.4f}")
            lines.append(f"| {d} | {po:+.4f} | {pr:+.4f} | {pr-po:+.4f} |")
            summary_rows.append({"section": "per_dataset", "method": m, "dataset": d,
                                 "post_ood": round(po, 6), "pre_ood": round(pr, 6),
                                 "delta": round(pr - po, 6)})
        macro_po = float(np.mean([po for _d, po, _pr in per_ds]))
        macro_pr = float(np.mean([pr for _d, _po, pr in per_ds]))
        print(f"\n  macro mean OOD PRR   post {macro_po:+.4f}   pre {macro_pr:+.4f}")
        print(f"  mean paired delta    {dv.mean():+.4f}      median {np.median(dv):+.4f}")
        print(f"  datasets pre > post  {pos}/{len(dv)}")
        print(f"  exact paired Wilcoxon p = {p:.4f} (n={n_nz} non-zero)")
        print(f"  bootstrap 95% CI on the mean delta [{lo:+.4f}, {hi:+.4f}]")
        lines += ["", f"- macro mean OOD PRR: post **{macro_po:+.4f}**, pre **{macro_pr:+.4f}**",
                  f"- mean paired delta **{dv.mean():+.4f}**, median {np.median(dv):+.4f}",
                  f"- datasets where pre > post: **{pos}/{len(dv)}**",
                  f"- exact paired Wilcoxon p = **{p:.4f}**",
                  f"- bootstrap 95% CI on the mean delta **[{lo:+.4f}, {hi:+.4f}]**"]
        cmp_md[m] = "\n".join(lines)
        summary_rows.append({"section": "paired_summary", "method": m, "n_datasets": len(dv),
                             "macro_post": round(macro_po, 6), "macro_pre": round(macro_pr, 6),
                             "mean_delta": round(float(dv.mean()), 6),
                             "median_delta": round(float(np.median(dv)), 6),
                             "n_pre_gt_post": pos, "wilcoxon_p": round(p, 6),
                             "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)})

    # ---------------------------------------------------------------- write
    out_csv = INDIR / "alignment_summary.csv"
    fields = ["section", "method", "alignment", "dataset", "rung", "value", "post_ood", "pre_ood",
              "delta", "n_datasets", "macro_post", "macro_pre", "mean_delta", "median_delta",
              "n_pre_gt_post", "wilcoxon_p", "ci_lo", "ci_hi"]
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(prof_rows + summary_rows)
    # the tidy long CSV: one row per model x method x alignment x dataset x rung x seed
    combined = INDIR / f"alignment_prr_ALL__{SLUG}.csv"
    seen_hdr, n_rows = None, 0
    with open(combined, "w", newline="") as out:
        w = None
        for p in files:
            with open(p) as f:
                for r in _csv.DictReader(f):
                    if w is None:
                        seen_hdr = list(r.keys())
                        w = _csv.DictWriter(out, fieldnames=seen_hdr); w.writeheader()
                    w.writerow(r); n_rows += 1
    print(f"wrote {combined}  ({n_rows} rows from {len(files)} per-eval files)")

    for m, body in cmp_md.items():
        p = INDIR / f"comparison_{m}.md"
        p.write_text(f"# {m} — token/state alignment comparison\n\n"
                     f"Post-hoc sensitivity analysis. Unit of analysis = dataset, n = "
                     f"{len(have)}. Mean over the four OOD rungs per dataset.\n\n{body}\n")
        print(f"wrote {p}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
