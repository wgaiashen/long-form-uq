"""Evaluate the W-Models prompt-regime validity gate mechanically (prereg/M5 §6).

WHY THIS IS A SCRIPT AND NOT A JUDGEMENT CALL
---------------------------------------------
The gate decides whether a population keeps RAW FEW-SHOT prompting or moves to its native chat
template. That decision changes the population, so it must be made on pre-registered thresholds and
nothing else. Reading six rows of a CSV and deciding by eye is exactly how a threshold quietly
becomes "close enough".

⚠️ IT NEVER READS PRR. Only generation-validity signals: degeneracy, empties, invented
continuations, cap behaviour. Choosing a prompt regime on downstream uncertainty performance would
be selecting the population on the outcome.

⚠️ THE DECISION IS PER MODEL, INDEPENDENTLY. One instruct model failing must not push another onto a
chat template (prereg §6). So this scores exactly one population per invocation.

⚠️ A PARTIAL SMOKE CANNOT PASS. All six panel datasets must be present. A missing dataset is not a
silent pass -- it is a FAIL with the missing names listed.

    python scripts/checks/wmodels_gate.py --smoke-csv <path>            # thresholds only
    python scripts/checks/wmodels_gate.py --smoke-csv <path> --baseline-csv results/generation_quality.csv
"""
import argparse
import csv
import sys

PANEL = ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "asqa", "factscore"]

# Pre-registered thresholds (prereg/M5_multimodel_far_ood.md §6), anchored to the accepted
# Llama-3.1-8B base population measured 2026-08-15.
MAX_SEVERE = 5.0
MAX_DEGRADED = 10.0
MAX_EMPTY = 1.0
MAX_FABRICATED = 10.0
MIN_ANSWER_FRAC = 0.90
MAX_CAPPED_OVER_BASE = 20.0     # percentage POINTS above the base model on the same dataset


def rows_by_dataset(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["dataset"]] = r
    return out


def f(row, key, default=None):
    v = (row or {}).get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-csv", required=True)
    ap.add_argument("--baseline-csv", default=None,
                    help="base-model generation_quality.csv, for the %%capped comparison. Without it "
                         "the cap check is reported as NOT EVALUATED rather than silently passed.")
    ap.add_argument("--min-rows", type=int, default=40,
                    help="refuse to decide on fewer than this many rows per dataset. Matters for the "
                         "INLINE gate, which reads a partial cache while generation is still running: "
                         "a percentage over 5 rows is noise, and a regime decision made on noise is "
                         "worse than no gate at all. 40 = the pre-registered smoke size.")
    args = ap.parse_args()

    smoke = rows_by_dataset(args.smoke_csv)
    base = rows_by_dataset(args.baseline_csv) if args.baseline_csv else {}

    missing = [d for d in PANEL if d not in smoke]
    print(f"gate: {args.smoke_csv}")
    print(f"  datasets present: {len(smoke)}/6")
    if missing:
        print(f"  ❌ FAIL — smoke is incomplete, missing {missing}")
        print("  A partial smoke is not a pass. Re-run the missing datasets.")
        sys.exit(1)

    thin = [(d, int(float(smoke[d].get("n", 0) or 0))) for d in PANEL
            if float(smoke[d].get("n", 0) or 0) < args.min_rows]
    if thin:
        print(f"  ⏳ UNDECIDED — too few rows to judge: {thin} (need >= {args.min_rows} each)")
        print("  Not a pass and not a fail. Let generation add rows, then re-run this gate.")
        sys.exit(3)

    hdr = f"  {'dataset':15s} {'severe':>7s} {'degrad':>7s} {'empty':>6s} {'fabric':>7s} {'ansfrc':>7s} {'capped':>7s}  verdict"
    print(hdr)
    failures = []
    uncompared = []
    for d in PANEL:
        r = smoke[d]
        sev, deg = f(r, "pct_severe"), f(r, "pct_degraded")
        emp, fab = f(r, "pct_empty"), f(r, "pct_fabricated")
        afr, cap = f(r, "mean_answer_frac"), f(r, "pct_capped")
        why = []
        if sev is None or sev > MAX_SEVERE:
            why.append(f"severe {sev}>{MAX_SEVERE}")
        if deg is None or deg > MAX_DEGRADED:
            why.append(f"degraded {deg}>{MAX_DEGRADED}")
        if emp is None or emp > MAX_EMPTY:
            why.append(f"empty {emp}>{MAX_EMPTY}")
        if fab is None or fab > MAX_FABRICATED:
            why.append(f"fabricated {fab}>{MAX_FABRICATED}")
        if afr is None or afr < MIN_ANSWER_FRAC:
            why.append(f"answer_frac {afr}<{MIN_ANSWER_FRAC}")
        # ⚠️ A cap check with no baseline value must READ as unevaluated, not pass silently. The `~`
        # marks exactly that: the number is the smoke's own %capped, with nothing to compare it to.
        bcap = f(base.get(d), "pct_capped") if base else None
        if cap is None:
            capstr = "      ?"
        elif bcap is None:
            capstr = f"{cap:6.1f}~"
            uncompared.append(d)
        else:
            capstr = f"{cap:7.1f}"
            if cap - bcap > MAX_CAPPED_OVER_BASE:
                why.append(f"capped {cap:.1f} vs base {bcap:.1f} (+{cap - bcap:.1f}pt)")
        verdict = "pass" if not why else "FAIL: " + "; ".join(why)
        if why:
            failures.append(d)
        print(f"  {d:15s} {sev:7.2f} {deg:7.2f} {emp:6.2f} {fab:7.1f} {afr:7.3f} {capstr}  {verdict}")

    if uncompared:
        print(f"  ⚠️ %capped marked `~` = NOT COMPARED (no baseline row): {uncompared}. "
              f"Reported, never silently passed.")

    print()
    if failures:
        print(f"  ❌ RAW FEW-SHOT FAILS on {failures}")
        print("  Per prereg §6: THIS population moves to its native chat template. Do not move any")
        print("  other population. Record the regime and make no causal claim about instruction tuning.")
        sys.exit(2)
    print("  ✅ RAW FEW-SHOT PASSES on all six — keep raw few-shot for this population.")
    print("  (Decision made on generation-validity signals only; no PRR was read.)")


if __name__ == "__main__":
    main()
