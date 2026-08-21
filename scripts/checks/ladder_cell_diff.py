"""LADDER CELL DIFF (Round-3 Step 2b): narrow-pool control vs widened-pool re-run, per cell.

TWO JOBS:
  1. REGRESSION GUARD. `contribution_ladder` was refactored onto the shared `build_rows` in the SAME change as
     the Task-A sampler fix -- a behaviour-changing refactor shipped with a bug fix (silent-second-change risk).
     Only the 4 XL evals' SameTask/LOO pools legitimately changed (they gained the previously-0-row eval-only
     sources); EVERY OTHER cell has an unchanged pool and, at identical seeds, MUST reproduce. So:
       - assert every SHOULD-BE-EXACT cell reproduces (|Δprr_mean| < tol);
       - a CORE-5 cell that moves is a STOP condition (that is the refactor, not the fix);
       - report the legitimately-changed XL SameTask/LOO cells separately (the Task-A effect).
  2. COMPARISON. Print the per-cell narrow-vs-widened Δ for the changed cells (the actual result).

Joins on (rung, eval, method) -- NOT `train` (labels now carry realised counts, which differ by design).
Works for §C.2 (contribution_ladder rungs) and §C.3 (probedriftlong `-long` rungs; suffix stripped).

    python scripts/checks/ladder_cell_diff.py --narrow 'results/contribution_ladder*.csv' \
        --widened results/contribution_ladder_widened.csv
"""
import argparse
import csv as _csv
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
CORE5 = {"sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail"}
# the ONLY cells whose pool legitimately changed under Task A: the 4 XL evals' SameTask + LOO rungs.
CHANGED = {(e, r) for e in ("med_quad", "samsum", "expertqa", "asqa") for r in ("SameTask", "LOO")}


def base_rung(r):
    return (r or "").replace("-long", "")


def load(globs):
    """{(rung, eval, method): (prr_mean, file)} keeping the most-recent file per cell (mtime)."""
    cell = {}
    for g in globs:
        for f in glob.glob(g):
            if MODEL_SLUG not in os.path.basename(f):     # pin the model (never ingest a non-Llama CSV)
                continue
            mt = os.path.getmtime(f)
            try:
                rows = list(_csv.DictReader(open(f)))
            except Exception:
                continue
            for r in rows:
                m = r.get("method", "")
                if not m or m.startswith("VERDICT") or m.startswith("fair_floor"):
                    continue
                try:
                    v = float(r["prr_mean"])
                except (KeyError, TypeError, ValueError):
                    continue
                k = (base_rung(r.get("rung", "")), r.get("eval", ""), m)
                if k not in cell or mt > cell[k][1]:
                    cell[k] = (v, mt)
    return {k: v[0] for k, v in cell.items()}


def main():
    apr = argparse.ArgumentParser()
    apr.add_argument("--narrow", nargs="+", required=True, help="narrow-pool control CSV glob(s)")
    apr.add_argument("--widened", nargs="+", required=True, help="widened-pool re-run CSV glob(s)")
    # 2026-07-28: tol raised 5e-4 -> 3e-3 after VERIFYING the flagged core-5 moves were all ±0.0005 and came
    # from the OLD committed control being float/version-STALE (sciq perplexity floor recomputes to 0.54050 =
    # the widened value, not the committed 0.5410; a FLOOR cell can't be touched by the build_rows refactor).
    # A real refactor regression changes the training POOL -> a LARGE move (the legit XL changes are 0.05-0.25),
    # so 3e-3 cleanly separates verified version-drift (≤0.0005) from material breakage (≥0.01).
    apr.add_argument("--tol", type=float, default=3e-3, help="|Δ| below this = reproduces (verified drift ≤5e-4)")
    args = apr.parse_args()
    N = load(args.narrow); W = load(args.widened)
    common = sorted(set(N) & set(W))
    if not common:
        raise SystemExit("no (rung,eval,method) cells in common -- check the globs/paths")

    exact_fail, core5_stop, changed, reproduced = [], [], [], 0
    for k in common:
        rung, eval_, method = k
        delta = W[k] - N[k]
        should_change = (eval_, rung) in CHANGED
        if should_change:
            changed.append((k, N[k], W[k], delta))
        elif abs(delta) < args.tol:
            reproduced += 1
        else:
            exact_fail.append((k, N[k], W[k], delta))
            if eval_ in CORE5:
                core5_stop.append((k, N[k], W[k], delta))

    print(f"cells compared: {len(common)}  |  reproduced (should-be-exact): {reproduced}  |  "
          f"legitimately-changed: {len(changed)}  |  UNEXPECTED moves: {len(exact_fail)}")
    if exact_fail:
        print("\nUNEXPECTED MOVES (should-be-exact cells that changed) — investigate before trusting the run:")
        for (rung, e, m), n, w, d in sorted(exact_fail, key=lambda x: -abs(x[3])):
            print(f"    {e:14s} {rung:18s} {m:22s} {n:+.4f} -> {w:+.4f}  Δ{d:+.4f}"
                  + ("   <== CORE-5 STOP" if e in CORE5 else ""))
    if core5_stop:
        print(f"\nSTOP: {len(core5_stop)} CORE-5 cell(s) moved — that is the refactor, not the fix. "
              f"Nothing from the widened run is usable until explained.")
    print("\nLEGITIMATELY-CHANGED (XL SameTask/LOO — the Task-A effect):")
    for (rung, e, m), n, w, d in sorted(changed):
        if m in ("attention", "uniform", "msp_min", "weighted_msp_norm"):   # the headline methods
            print(f"    {e:14s} {rung:18s} {m:22s} {n:+.4f} -> {w:+.4f}  Δ{d:+.4f}")
    print(f"\nREGRESSION: {'PASS' if not exact_fail else 'FAIL'}")
    sys.exit(1 if core5_stop else 0)


if __name__ == "__main__":
    main()
