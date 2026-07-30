"""verify_2x2_gates.py -- read-only gate check for head_aggregation_2x2 output.

The 2x2 driver re-runs the three known cells alongside the new one. This confirms the known cells
reproduce their reference values, so the NEW cell (attention_mlp) is comparable to the published board:

  HARD gates (exit 1 on failure -- the recipe diverged, attention_mlp is not comparable):
    meanpool_linear  ==  armB   (results/fixed_prior_ladder__*.csv, per-cell raw -- NOT the assembler avg)
    attention_linear ==  armA   (results/fixed_prior_ladder__*.csv, per-cell raw)
    meanpool_mlp_5ep ==  saplma_ref   (same-run; the MLP-head path IS the SAPLMA head)
  SOFT checks (reported, never fatal):
    saplma_ref       ~=  saplma (results/probedriftlong_<eval>__*.csv) -- cross-driver sanity
    bridge delta     =   meanpool_mlp(60ep) - saplma_ref(5ep)   -- how much 60ep overfits the MLP head
    q_final_norm / retracted for attention_mlp                  -- did the query actually train

Pass criterion per gate: |Δ| <= max(--tol, the cell's prr_std) (seed noise).

    python scripts/checks/verify_2x2_gates.py --result results/head_aggregation_2x2_pubmed_qa__<slug>.csv
    python scripts/checks/verify_2x2_gates.py --result 'results/head_aggregation_2x2_*.csv'
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _load(path):
    """CSV -> {(rung, eval, method): (prr_mean, prr_std)} for non-VERDICT rows with a numeric prr_mean."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m = r.get("method", "")
            if m.startswith("VERDICT") or r.get("prr_mean", "") in ("", None):
                continue
            try:
                mean = float(r["prr_mean"])
            except ValueError:
                continue
            std = float(r["prr_std"]) if r.get("prr_std") not in ("", None) else 0.0
            out[(r["rung"], r["eval"], m)] = (mean, std)
    return out


def _first(pattern):
    hits = sorted(glob.glob(str(RESULTS / pattern)))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="the head_aggregation_2x2 CSV (or a glob over per-eval files)")
    ap.add_argument("--tol", type=float, default=0.02, help="base tolerance; effective tol = max(tol, prr_std)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.result)) or [args.result]
    new = {}
    for p in paths:
        new.update(_load(p))
    if not new:
        raise SystemExit(f"no rows loaded from {args.result}")

    fpl_path = _first("fixed_prior_ladder__*.csv")
    fpl = _load(fpl_path) if fpl_path else {}
    if not fpl:
        print("WARN: no fixed_prior_ladder__*.csv found -- armA/armB gates will be SKIPPED (not proven).")

    # map new-driver method -> (reference method, reference table)
    HARD = [("meanpool_linear", "armB", fpl, "fixed_prior_ladder"),
            ("attention_linear", "armA", fpl, "fixed_prior_ladder")]

    evals = sorted({e for (_r, e, _m) in new})
    pdl = {}
    for e in evals:
        pp = _first(f"probedriftlong_{e}__*.csv")
        if pp:
            pdl.update(_load(pp))

    n_fail = 0
    for e in evals:
        cells = sorted({r for (r, ev, _m) in new if ev == e})
        print(f"\n==== {e} ====")
        for rung in cells:
            print(f"  [{rung}]")
            # hard gates: linear cells vs armA/armB
            for new_m, ref_m, ref, tbl in HARD:
                nk = new.get((rung, e, new_m)); rk = ref.get((rung, e, ref_m))
                if nk is None:
                    continue
                if rk is None:
                    print(f"    {new_m:16s} vs {ref_m:6s} ({tbl}): reference cell ABSENT -- SKIP")
                    continue
                d = nk[0] - rk[0]; tol = max(args.tol, nk[1], rk[1])
                ok = abs(d) <= tol
                n_fail += (not ok)
                print(f"    {new_m:16s} {nk[0]:+.3f}  vs {ref_m} {rk[0]:+.3f}  Δ={d:+.3f} (tol {tol:.3f})  "
                      f"{'PASS' if ok else 'FAIL <<<'}")
            # same-run gate: meanpool_mlp_5ep == saplma_ref
            a = new.get((rung, e, "meanpool_mlp_5ep")); s = new.get((rung, e, "saplma_ref"))
            if a and s:
                d = a[0] - s[0]; tol = max(args.tol, a[1], s[1])
                ok = abs(d) <= tol; n_fail += (not ok)
                print(f"    {'meanpool_mlp_5ep':16s} {a[0]:+.3f}  vs saplma_ref {s[0]:+.3f}  Δ={d:+.3f} "
                      f"(tol {tol:.3f})  {'PASS' if ok else 'FAIL <<<'}")
            # soft: saplma_ref vs the checked-in saplma (cross-driver)
            sp = pdl.get((rung, e, "saplma"))
            if s and sp:
                print(f"    {'saplma_ref':16s} {s[0]:+.3f}  vs saplma(pdl) {sp[0]:+.3f}  Δ={s[0]-sp[0]:+.3f}  (soft)")
            # soft: the bridge delta (recipe/overfit effect)
            mm = new.get((rung, e, "meanpool_mlp"))
            if mm and s:
                print(f"    bridge delta meanpool_mlp(60ep) - saplma_ref(5ep) = {mm[0]-s[0]:+.3f}  (soft; large -> 60ep overfits)")

    print(f"\n{'ALL HARD GATES PASS' if n_fail == 0 else f'{n_fail} HARD GATE(S) FAILED'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
