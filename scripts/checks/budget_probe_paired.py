"""Did the capped tail want more room, or does it just never stop? A PAIRED budget comparison.

WHY PAIRED, AND WHY THIS EXISTS AS A SCRIPT. `generation_quality.py` describes one cache at a time.
Answering the budget question means comparing two, and the obvious comparison is wrong in two ways
that both produce a plausible number:

  1. POPULATION. The v1 rows are the whole dataset (n=3800) while a probe run is `--limit 200` per
     split (n=400). Reading one CSV line against the other compares different populations. Here the
     two caches are joined on (split, idx) and only matched rows are used.
  2. PROMPT. Two caches can be built by different `probe_drift` checkouts, in which case the budget
     is not the only thing that changed. That is not hypothetical: it is what happened to xsum on
     2026-08-01, where v1 and the probe shared 0 of 400 prompts. This asserts prompt equality on the
     matched rows and REFUSES to report if it fails, rather than emitting a confounded number.

THE STATISTIC THAT ANSWERS THE QUESTION. Not the median, and not the truncation rate. Take the rows
that were CAPPED at the old budget and ask how many stop on their own once the cap is raised:

  - most of them finish  -> the cap was cutting off answers that had somewhere to end. Truncation.
  - most still run to the new cap -> they have no natural stopping point in that range, and a bigger
    budget just buys more text. Over-generation.

The median cannot distinguish these. On both xsum and cnn the median does not move at all while the
tail behaves completely differently from what "the model wanted room" would predict.

THE CONTROL is the rows that were NOT capped at the old budget. Greedy decoding on an identical
prompt must reproduce them exactly, so that column has to be ~100%. When it is not, the comparison
is broken and the number should not be read -- it is what exposed the xsum prompt change (1.7%
against cnn's 100.0%).

    python scripts/checks/budget_probe_paired.py                       # print the table
    python scripts/checks/budget_probe_paired.py --out results/budget_probe_paired.csv

Reads records only. CPU, no model, no GPU, no judge calls.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, degeneracy  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"

# (dataset, base regime, base budget, probe regime, probe budget)
# xsum's base is v2probe_xsum_base56, NOT the v1 cache: v1 xsum was generated under a different
# probe_drift, so it cannot serve as a one-variable baseline at any budget. cnn's v1 cache was built
# under the same checkout as its probe, so it can.
PAIRS = [("xsum", "v2probe_xsum_base56", 56, "v2probe_xsum", 96),
         ("cnn_dailymail", "", 128, "v2probe_cnn", 256)]


def load(dataset: str, regime: str):
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID", prompt_regime=regime)
    path = Path(cfg.cache_dir) / "records" / f"{cache.run_key(MODEL, dataset, 'ID')}.jsonl"
    if not path.exists():
        return None
    out = {}
    for line in path.open():
        r = json.loads(line)
        out[(r["split"], r["idx"])] = r
    return out


def compare(dataset, base_regime, base_budget, probe_regime, probe_budget):
    A, B = load(dataset, base_regime), load(dataset, probe_regime)
    if A is None or B is None:
        print(f"  {dataset}: missing a cache -> SKIPPED (blank, not zero)")
        return None
    keys = sorted(set(A) & set(B))
    if not keys:
        print(f"  {dataset}: no overlapping (split, idx) -> SKIPPED")
        return None

    same_prompt = sum(A[k]["prompt"] == B[k]["prompt"] for k in keys)
    if same_prompt != len(keys):
        # Refuse rather than report. A prompt difference means the budget was not the only variable,
        # and a number produced here would look exactly like a valid one.
        print(f"  {dataset}: prompts differ on {len(keys)-same_prompt}/{len(keys)} matched rows "
              f"-> REFUSING to report a budget effect; this is not a one-variable comparison.")
        return None

    g1 = np.array([len(A[k]["gen_token_ids"]) for k in keys])
    g2 = np.array([len(B[k]["gen_token_ids"]) for k in keys])
    c1, c2 = g1 >= base_budget - 1, g2 >= probe_budget - 1
    unc = ~c1
    control = 100.0 * (g1[unc] == g2[unc]).mean() if unc.any() else float("nan")
    fin, still = int((c1 & ~c2).sum()), int((c1 & c2).sum())
    sev = np.array([degeneracy.is_severe(B[k]["gen_text"]) for k in keys])
    empty = np.array([len(B[k]["gen_text"].strip()) == 0 for k in keys])
    return {
        "dataset": dataset, "n_paired": len(keys),
        "base_regime": base_regime or "v1", "base_budget": base_budget,
        "probe_regime": probe_regime, "probe_budget": probe_budget,
        "prompts_identical": f"{same_prompt}/{len(keys)}",
        "control_pct_identical": round(control, 1),
        "p50_base": float(np.percentile(g1, 50)), "p50_probe": float(np.percentile(g2, 50)),
        "pct_capped_base": round(100 * float(c1.mean()), 1),
        "pct_capped_probe": round(100 * float(c2.mean()), 1),
        "n_capped_base": int(c1.sum()),
        "n_finished_when_uncapped": fin,
        "pct_finished_when_uncapped": round(100 * fin / max(int(c1.sum()), 1), 1),
        "n_still_capped": still,
        "probe_pct_empty": round(100 * float(empty.mean()), 2),
        "probe_pct_severe": round(100 * float(sev.mean()), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [r for r in (compare(*p) for p in PAIRS) if r]
    if not rows:
        sys.exit("no comparable pairs")

    print("\nPAIRED BUDGET COMPARISON -- matched on (split, idx), prompts asserted identical\n")
    print(f"{'dataset':14s} {'n':>5s} {'budgets':>10s} {'p50':>11s} {'%capped':>13s} "
          f"{'control':>8s} {'capped rows that finished':>26s}")
    for r in rows:
        budgets = f"{r['base_budget']}->{r['probe_budget']}"
        p50 = f"{r['p50_base']:.0f}->{r['p50_probe']:.0f}"
        capped = f"{r['pct_capped_base']}->{r['pct_capped_probe']}"
        finished = (f"{r['n_finished_when_uncapped']}/{r['n_capped_base']} "
                    f"({r['pct_finished_when_uncapped']}%)")
        print(f"{r['dataset']:14s} {r['n_paired']:>5d} {budgets:>10s} {p50:>11s} {capped:>13s} "
              f"{r['control_pct_identical']:>7.1f}% {finished:>26s}")
    print("\n  control = % of rows NOT capped at the base budget whose length is IDENTICAL at the")
    print("  larger budget. Greedy decoding forces this to ~100%; anything less means the two caches")
    print("  differ by more than the budget and the row must not be read as a budget effect.")
    print("\n  'capped rows that finished' is the statistic that answers the question: a LOW value")
    print("  means the tail has no natural stopping point (over-generation), a HIGH value means the")
    print("  cap was cutting off answers that had somewhere to end (truncation).")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
