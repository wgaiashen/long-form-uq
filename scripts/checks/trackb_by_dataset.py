"""R2 — re-read every Track B result BY DATASET rather than by the cross-dataset mean.

WHY THIS EXISTS
---------------
Track B (B.1 auxiliary-loss attention supervision, B.2 selective heads, B.3 entropy penalty) was recorded
as "three clean negatives". That framing came from reading the results as a pooled verdict. Read per
dataset they are not null — they are **null on one dataset and actively harmful on the other**, and the
direction tracks a property of the dataset.

⚠️ SCOPE, STATED BEFORE THE NUMBERS. Every Track B experiment ran on **pubmed_qa and xsum only, ID rung
only, seed 1 only**. Two datasets is not a grid. In the regime taxonomy those two are CONCENTRATED and
NOT-IN-PROBABILITIES, so **no SPREAD dataset (cnn, asqa, samsum) was ever tested** — and SPREAD is
precisely the regime where an averaging/sharpness intervention has a mechanism. Nothing here is a
cross-dataset claim; it is a description of two cells plus an explicit statement of what is missing.

This script reads the committed Track B CSVs and reports them per dataset. No refitting, no new runs.

    python scripts/checks/trackb_by_dataset.py
"""
import csv as _csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
RES = ROOT / "results"

# Dataset properties measured elsewhere, carried here ONLY as context for reading the deltas.
# Sources: results/pool_peaks_9dataset.csv (punct mass, entropy ratio) and
# results/floor_taxonomy__*.csv (regime). Not recomputed here -- quoted, with the source named.
CONTEXT = {
    "pubmed_qa": {"regime": "CONCENTRATED", "punct_mass_ID": 0.515, "entropy_ratio_ID": 0.606},
    "xsum": {"regime": "NOT-IN-PROBABILITIES", "punct_mass_ID": 0.069, "entropy_ratio_ID": None},
}

B1_FILES = [("stage1", "aux_attention_stage1"),
            ("stage1b_lowlambda", "aux_attention_stage1b_lowlambda"),
            ("stage1b_dropsched", "aux_attention_stage1b_dropsched")]
B3_FILES = [("tau0.9", "b3_entropy_penalty"), ("low_tau", "b3_entropy_penalty_lowtau")]


def read(stem):
    p = RES / f"{stem}__{SLUG}.csv"
    if not p.exists():
        sys.exit(f"missing {p} -- refusing to report a Track B summary with a file absent.")
    with open(p) as fh:
        return list(_csv.DictReader(fh))


def main():
    print("R2 — Track B re-read BY DATASET (not by the mean)")
    print("population: pubmed_qa + xsum, ID rung, seed 1. NO SPREAD dataset was ever tested.\n")

    rows = []

    print("=" * 78)
    print("B.1 auxiliary-loss attention supervision — real minus baseline, per dataset")
    print("=" * 78)
    print(f"{'variant':20s} {'dataset':12s} {'target':13s} {'lambda':>7s} {'vs base':>9s} {'vs shuf':>9s}")
    for variant, stem in B1_FILES:
        for r in read(stem):
            rows.append({"experiment": "B.1", "variant": variant, "dataset": r["eval"],
                         "arm": r["target"], "delta_vs_baseline": float(r["real_minus_baseline"]),
                         "delta_vs_shuffled": float(r["real_minus_shuffled"])})
            print(f"{variant:20s} {r['eval']:12s} {r['target']:13s} {r['best_lambda']:>7s} "
                  f"{float(r['real_minus_baseline']):+9.4f} {float(r['real_minus_shuffled']):+9.4f}")

    print("\n" + "=" * 78)
    print("B.3 one-sided entropy penalty — penalty minus baseline, per dataset")
    print("=" * 78)
    print(f"{'variant':20s} {'dataset':12s} {'tau':>5s} {'lambda':>7s} {'delta':>9s} "
          f"{'ent before':>11s} {'ent after':>10s}")
    for variant, stem in B3_FILES:
        for r in read(stem):
            rows.append({"experiment": "B.3", "variant": variant, "dataset": r["eval"],
                         "arm": f"tau={r['tau']}", "delta_vs_baseline": float(r["delta"]),
                         "delta_vs_shuffled": ""})
            print(f"{variant:20s} {r['eval']:12s} {r['tau']:>5s} {r['lambda']:>7s} "
                  f"{float(r['delta']):+9.4f} {float(r['ent_before']):>11.4f} "
                  f"{float(r['ent_after']):>10.4f}")

    # ---- The per-dataset summary, which is the actual point.
    print("\n" + "=" * 78)
    print("THE RE-READ")
    print("=" * 78)
    for ds in ("pubmed_qa", "xsum"):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        deltas = [r["delta_vs_baseline"] for r in ds_rows]
        worst = min(deltas)
        n_harm = sum(1 for d in deltas if d <= -0.01)
        ctx = CONTEXT[ds]
        print(f"\n{ds}  [{ctx['regime']}, punct mass {ctx['punct_mass_ID']}]")
        print(f"  {len(deltas)} interventions: worst {worst:+.4f}, best {max(deltas):+.4f}, "
              f"mean {sum(deltas)/len(deltas):+.4f}")
        print(f"  {n_harm} of {len(deltas)} cost more than 0.01 PRR")

    print("\n⭐ THE PATTERN, STATED AT THE STRENGTH THE EVIDENCE SUPPORTS:")
    print("  Every intervention that moved attention AWAY from pubmed's learned configuration cost PRR")
    print("  there (worst -0.0513), while the SAME interventions did nothing measurable on xsum")
    print("  (|delta| <= 0.006 throughout). pubmed is the dataset with 0.515 punctuation mass and the")
    print("  sharpest pooling; xsum has 0.069 and was already diffuse -- there was little to move.")
    print("\n  So Track B is NOT 'three methods that do nothing'. It is 'interventions that damage a")
    print("  concentrated attention configuration, and are inert where attention is already spread'.")
    print("\n⚠️ WHAT THIS IS NOT. n=2 datasets, 1 rung, 1 seed. The dataset property and the outcome are")
    print("  confounded -- pubmed differs from xsum in punctuation mass, sharpness, task, length and")
    print("  baseline PRR at once, and with two points those cannot be separated. The pattern is a")
    print("  HYPOTHESIS that predicts what a SPREAD dataset should do; it is not established until one")
    print("  is run. B.3 on cnn/samsum is the missing arm and it is cheap.")
    print("\n  Its truth also depends on R0: 'damage a concentrated configuration' presumes that")
    print("  configuration is informative, which is exactly what R0 measures. If R0 says punctuation")
    print("  ablation does NOT hurt pubmed, this reading is wrong and B.1's failure needs another cause.")

    out = RES / f"regime_R2_trackb_by_dataset__{SLUG}.csv"
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
