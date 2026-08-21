"""Cell-level paired significance for ANY method pair, read post-hoc from the per-example sidecars.

Why this script exists: `pdl_master` carries one PRR per cell and nothing else, so every "is that
difference significant?" question used to need the whole ladder re-run. `probedriftlong.py --perex-dir`
now persists the per-example uncertainty vectors that were always in memory, and this script turns them
into a paired bootstrap. Adding a new comparison is now free.

THE BOOTSTRAP (fixed in advance, 2026-07-31):
  - the unit of resampling is the CELL, n = 32 OOD cells (8 evals x 4 OOD rungs)
  - the per-cell value is the 3-SEED MEAN of (PRR_a - PRR_b), matching pdl_master's 3-seed regime
  - report the mean margin, the percentile CI, and whether it excludes zero

Seed-regime discipline: this refuses to mix regimes. If two methods in a pair were run with different
seed sets in a cell, that cell is dropped LOUDLY rather than compared -- the seed-1-pooler-vs-3-seed-SAPLMA
mix is exactly the class of error that produced the retracted cross-population router number.

    python scripts/checks/pdl_significance.py
    python scripts/checks/pdl_significance.py --pairs saplma:floor_min,saplma:fair_floor,attention:floor_min
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
OOD_RUNGS = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
LONG_EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
# master alias -> sidecar method key, so a cross-check against pdl_master is possible without guessing
MASTER_ALIAS = {"saplma": "SAPLMA", "floor_min": "msp_min", "floor_sum": "msp_sum",
                "floor_ppl": "perplexity", "attention": "armA(attention)", "uniform": "armB(mean-pool)"}


def load_cells(perex_dir):
    """{(eval, rung): npz} for every sidecar present. Absent cells stay absent -- never zero-filled."""
    cells = {}
    for p in sorted(Path(perex_dir).glob(f"*__{SLUG}.npz")):
        z = np.load(p, allow_pickle=False)
        ev = str(z["meta__eval"]); rg = str(z["meta__rung"])
        cells[(ev, rg)] = z
    return cells


FLOORS3 = ["floor_sum", "floor_ppl", "floor_min"]


def _prr_vec(z, m):
    """Per-seed PRR vector for method `m`, or None if absent.

    `maxof3` is DERIVED here rather than stored: the per-seed max over the three MSP floors, i.e. the
    best-of-three bar. That bar was REJECTED on 2026-07-24 ("gives the baseline three shots at being
    good"), so it is a ROBUSTNESS check and never the primary comparison -- but it has to be testable,
    and it cannot be read off the CSV because `fair_floor` in probedriftlong is hard-wired to the
    pre-registered `msp_min`, not to a max. Computing it from the stored vectors is exactly the kind of
    after-the-fact question the sidecars exist to answer.
    """
    if m == "maxof3":
        vs = [z[f"prr__{f}"] for f in FLOORS3 if f"prr__{f}" in z]
        return np.max(np.stack(vs), axis=0) if len(vs) == len(FLOORS3) else None
    return z[f"prr__{m}"] if f"prr__{m}" in z else None


def per_cell_margin(z, a, b):
    """3-seed mean of (PRR_a - PRR_b) for one cell, or None if either method is absent / regimes differ."""
    pa, pb = _prr_vec(z, a), _prr_vec(z, b)
    if pa is None or pb is None:
        return None, "absent"
    if pa.shape != pb.shape:
        return None, f"seed-regime mismatch ({pa.shape[0]} vs {pb.shape[0]} seeds)"
    return float(np.mean(pa - pb)), None


def bootstrap_cells(margins, n_boot, seed=1):
    """Resample CELLS (not examples): the cell is the unit of replication for a cross-dataset claim."""
    m = np.asarray(margins, float)
    rs = np.random.RandomState(seed)
    boot = np.array([m[rs.randint(0, len(m), len(m))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(m.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0), int((m > 0).sum()), len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perex-dir", default=str(ROOT / "results" / "pdl_perex"))
    ap.add_argument("--pairs", default="saplma:floor_min,saplma:fair_floor",
                    help="comma-separated a:b pairs; the reported margin is a - b")
    ap.add_argument("--rungs", default="OOD", help="'OOD' (the 4 OOD rungs), 'ALL', or a comma list")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--master", default=str(ROOT / "results" / f"pdl_master__{SLUG}.csv"),
                    help="cross-check the sidecars' per-cell PRR against the master table")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = load_cells(args.perex_dir)
    if not cells:
        raise SystemExit(f"no sidecars in {args.perex_dir} -- run probedriftlong.py --perex-dir first")
    rungs = OOD_RUNGS if args.rungs == "OOD" else (None if args.rungs == "ALL" else args.rungs.split(","))
    want = [(e, r) for e in LONG_EVALS for r in (rungs or sorted({k[1] for k in cells}))]
    have = [c for c in want if c in cells]
    missing = [c for c in want if c not in cells]
    print(f"sidecars: {len(cells)} on disk | this analysis wants {len(want)} cells, has {len(have)}")
    if missing:
        print(f"  MISSING {len(missing)} (left out of the analysis, NOT zero-filled): "
              + ", ".join(f"{e}/{r}" for e, r in missing))

    # ---- cross-check against pdl_master before trusting anything ------------------------------------
    if Path(args.master).exists():
        m = {}
        for row in _csv.DictReader(open(args.master)):
            try:
                m[(row["method"], row["eval"], row["rung"])] = float(row["prr"])
            except (ValueError, KeyError):
                pass
        worst, worst_at, n_chk = 0.0, None, 0
        for (ev, rg) in have:
            z = cells[(ev, rg)]
            for skey, mkey in MASTER_ALIAS.items():
                if f"prr_mean__{skey}" in z and (mkey, ev, rg) in m:
                    d = abs(float(z[f"prr_mean__{skey}"]) - m[(mkey, ev, rg)])
                    n_chk += 1
                    if d > worst:
                        worst, worst_at = d, f"{mkey} {ev}/{rg}"
        # the master rounds to 4dp, so 5e-5 is the rounding floor; anything above 1e-3 is a real disagreement
        verdict = "OK" if worst < 1e-3 else "*** DISAGREES WITH MASTER ***"
        print(f"master cross-check: {n_chk} cells, max|Δ| = {worst:.2e} at {worst_at}  {verdict}")
        if worst >= 1e-3:
            raise SystemExit("sidecars disagree with pdl_master beyond rounding -- refusing to report "
                             "significance built on them.")

    # ---- the bootstrap ------------------------------------------------------------------------------
    rows = []
    print(f"\npaired bootstrap over CELLS (n_boot={args.n_boot}, resampling cells, 3-seed mean per cell)\n")
    print(f"{'comparison':<28} {'n':>4} {'mean Δ':>9} {'95% CI':>20} {'a>b':>7}  verdict")
    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        margins, skipped = [], []
        for (ev, rg) in have:
            mg, why = per_cell_margin(cells[(ev, rg)], a, b)
            (margins if mg is not None else skipped).append(mg if mg is not None else f"{ev}/{rg} ({why})")
        if not margins:
            print(f"{a} vs {b}: no usable cells ({len(skipped)} skipped)")
            continue
        mean, lo, hi, sig, wins, n = bootstrap_cells(margins, args.n_boot)
        print(f"{a+' vs '+b:<28} {n:>4} {mean:>+9.4f} {f'[{lo:+.4f},{hi:+.4f}]':>20} {wins:>3}/{n}  "
              f"{'SIGNIFICANT' if sig else 'ns'}")
        if skipped:
            print(f"{'':28} skipped {len(skipped)}: " + "; ".join(map(str, skipped[:4])))
        rows.append({"comparison": f"{a}_vs_{b}", "n_cells": n, "mean_margin": round(mean, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "a_higher": f"{wins}/{n}",
                     "significant": sig, "n_boot": args.n_boot, "rungs": args.rungs,
                     "n_skipped": len(skipped)})

    if rows:
        out = Path(args.out) if args.out else ROOT / "results" / f"pdl_significance__{SLUG}.csv"
        with open(out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
