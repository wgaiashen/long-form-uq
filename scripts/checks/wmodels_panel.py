"""Score the W-Models reduced six-dataset panel exactly as pre-registered in M5.

WHAT THIS COMPUTES (prereg/multimodel_far_ood_panel.md §5)
-------------------------------------------------------
Primary, per population and per far-OOD rung:

    Delta_shrink = macro over the six datasets of [ PRR(wmsp_shrink2) - PRR(wmsp_norm) ]

    Directional model-level replication  <=>  macro Delta_shrink > 0.

There is deliberately NO additional per-dataset bar (no 4/6, no 5/6). Strength is conveyed by the
reported statistics -- macro, median, positive count, per-dataset deltas, a dataset bootstrap CI and
a DESCRIPTIVE Wilcoxon -- so a tiny macro carried by two datasets reads as weak on its face.

Secondary, reported SEPARATELY and never combined:

    Delta_SAPLMA    = PRR(wmsp_shrink2) - PRR(saplma)
    Delta_attention = PRR(wmsp_shrink2) - PRR(attention)

These are never collapsed into a synthetic max(SAPLMA, attention) comparator. Which learned
baseline is stronger is itself model-dependent and must stay visible.

WHAT IT REFUSES TO DO
---------------------
  * It asserts the Stage-A cell count (18 = 6 evals x 3 rungs) before reporting anything.
    `probedriftlong` prints "<d>: no pertok cache -> skip" and carries on, so a missing dataset
    yields a quietly SMALLER grid rather than an error. A blank cell reads as "not measured" and a
    number reads as "measured"; those must never be confusable.
  * It never pools model x dataset cells into one significance test. The unit is the DATASET within
    a model (n = 6), and the MODEL across populations (n = 3 replication populations).
  * It reports a 3/3 sign result as directional replication at p = 0.125, and does not offer a
    pooled five-model p-value.

    python scripts/checks/wmodels_panel.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/checks/wmodels_panel.py --all            # every population found
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

PANEL = ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "asqa", "factscore"]
STAGE_A = ["ID", "DiffTask-long", "1ds-Diff-long"]
FAR_OOD = ["DiffTask-long", "1ds-Diff-long"]

# Driver arm names (probedriftlong writes these, not the Llama master's display names).
SHRINK, NORM, SAPLMA, ATTN = "wmsp_shrink2", "wmsp_norm", "saplma", "attention"

N_BOOT = 10000
BOOT_SEED = 0


def load(slug):
    """(rung, eval, method) -> prr_mean, from this population's Stage-A CSVs."""
    g = {}
    files = sorted(glob.glob(str(RES / f"wmodels_stageA__{slug}__*.csv")))
    if not files:
        sys.exit(f"no Stage-A CSVs for {slug} under {RES}")
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["method"].startswith("VERDICT:"):
                continue
            try:
                g[(r["rung"], r["eval"], r["method"])] = float(r["prr_mean"])
            except (TypeError, ValueError):
                continue      # blank stays ABSENT, never 0
    return g, files


def assert_cells(g, slug):
    cells = {(rung, ev) for (rung, ev, _) in g}
    want = {(r, e) for r in STAGE_A for e in PANEL}
    missing = sorted(want - cells)
    if missing:
        sys.exit(f"{slug}: Stage A is {len(want & cells)}/18 cells, MISSING {missing}\n"
                 f"  Refusing to report a partial grid as if it were complete.")
    print(f"  cell count OK: 18/18 Stage-A cells", flush=True)


def deltas(g, rung, a, b):
    """Per-dataset a - b on `rung`; None where either arm is absent."""
    out = []
    for ev in PANEL:
        x, y = g.get((rung, ev, a)), g.get((rung, ev, b))
        out.append(None if x is None or y is None else x - y)
    return out


def boot_ci(d, seed=BOOT_SEED):
    """Bootstrap the MACRO over datasets, resampling DATASETS (the unit of analysis)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = np.asarray(d)[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_desc(d):
    """DESCRIPTIVE only at n=6 -- never the verdict (prereg §5.3)."""
    try:
        from scipy import stats
        return float(stats.wilcoxon(d, alternative="greater").pvalue)
    except Exception:
        return float("nan")


def report_rung(g, rung):
    lines = []
    for label, a, b in (("Delta_shrink", SHRINK, NORM),
                        ("Delta_SAPLMA", SHRINK, SAPLMA),
                        ("Delta_attention", SHRINK, ATTN)):
        d = deltas(g, rung, a, b)
        if any(v is None for v in d):
            absent = [PANEL[i] for i, v in enumerate(d) if v is None]
            lines.append(f"    {label:16s} NOT COMPUTABLE - arm absent on {absent}")
            continue
        arr = np.asarray(d, float)
        lo, hi = boot_ci(arr)
        lines.append(
            f"    {label:16s} macro {arr.mean():+.4f}  median {np.median(arr):+.4f}  "
            f"pos {int((arr > 0).sum())}/6  CI [{lo:+.4f}, {hi:+.4f}]  "
            f"wilcoxon(desc) p={wilcoxon_desc(arr):.3f}")
        lines.append("      per-dataset: " + "  ".join(
            f"{ev}={v:+.3f}" for ev, v in zip(PANEL, arr)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="e.g. meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--all", action="store_true", help="every population with Stage-A CSVs")
    args = ap.parse_args()

    if args.all:
        slugs = sorted({Path(f).name.split("__")[1]
                        for f in glob.glob(str(RES / "wmodels_stageA__*__*.csv"))})
    elif args.model:
        slugs = [args.model.replace("/", "_")]
    else:
        sys.exit("pass --model or --all")

    summary = []
    for slug in slugs:
        print(f"\n=== {slug} ===", flush=True)
        g, files = load(slug)
        print(f"  {len(files)} Stage-A CSVs", flush=True)
        assert_cells(g, slug)
        for rung in FAR_OOD:
            print(f"  [{rung}]", flush=True)
            for ln in report_rung(g, rung):
                print(ln, flush=True)
        ds = {r: deltas(g, r, SHRINK, NORM) for r in FAR_OOD}
        summary.append((slug, {r: (np.mean(v) if all(x is not None for x in v) else None)
                               for r, v in ds.items()}))

    # Cross-model framing -- deliberately modest (prereg §5.4).
    print("\n=== primary verdict: Delta_shrink macro > 0 per population ===", flush=True)
    for slug, m in summary:
        bits = "  ".join(f"{r}={'n/a' if m[r] is None else f'{m[r]:+.4f}'} "
                         f"{'PASS' if (m[r] is not None and m[r] > 0) else 'fail'}" for r in FAR_OOD)
        print(f"  {slug:38s} {bits}", flush=True)
    print("\n  Reminder (prereg §5.4): the three NEW populations are the replication set. 3/3 is"
          "\n  p = 0.125 under a one-sided sign test -- DIRECTIONAL replication, not conventional"
          "\n  significance. Development models are descriptive consistency evidence only, and no"
          "\n  pooled five-model p-value is computed.", flush=True)


if __name__ == "__main__":
    main()
