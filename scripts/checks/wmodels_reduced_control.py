"""The FREE CONTROL for the W-Models reduced six-dataset panel.

WHAT IT CHECKS AND WHY IT IS FREE
---------------------------------
The reduced panel (prereg M5 §2) drops `med_quad` and `expertqa` from the SOURCE pool. That changes
which datasets a probe trains on, so it changes some rungs and leaves others untouched:

    ID            train = the eval's own rows            -> pool spec IDENTICAL to the full grid
    1ds-Diff-long train = ONE opposite-family long set   -> pool spec IDENTICAL to the full grid
    DiffTask-long train = the whole opposite family      -> pool spec genuinely CHANGED
    SameTask/LOO  train = family / everything-else       -> pool spec genuinely CHANGED

So `ID` and `1ds-Diff-long` MUST reproduce the published master exactly. Nothing has to be computed
to earn that check -- the reduced ladder runs those two rungs anyway on its way to `DiffTask-long`,
which is why it is free. If they drift, the restriction is implemented wrongly and every new-model
number built on it is suspect. Catching that costs nothing and catching it late costs everything.

WHY IT COMPARES VECTORS AND NOT PRR
-----------------------------------
PRR is a rank statistic over 2,000 examples and moves ~1e-4 on 1e-7 of floating-point round-off, so
"the PRRs agree to 3 dp" would pass runs that are not actually the same computation. The comparison
is therefore on the PER-EXAMPLE uncertainty vectors, per method and per seed, plus the label vector
and the training-spec string.

⚠️ THE TRAINING-SPEC STRING IS THE PREMISE, NOT A DETAIL. `meta__train` records exactly which
sources fed the probe and at what cap. If the reduced run's `ID` spec is not character-identical to
the master's, the two runs are not the same cell and comparing their vectors is meaningless. It is
checked FIRST and a mismatch is fatal.

⚠️ WHAT THIS CONTROL CAN AND CANNOT SEE (read before quoting a PASS)
The master's per-example sidecars in `results/pdl_perex/` carry only five methods:
`floor_sum, floor_ppl, floor_min, fair_floor, saplma`. The poolers and the wMSP variants were never
dumped there, so they are NOT COMPARABLE and are reported as such rather than silently omitted.
Of the five that are comparable, four are TRAINING-POOL-INDEPENDENT: the floors are unsupervised and
index only the eval rows, and `fair_floor` is a max over them. Those four would match even if the
pool had changed, so they are a weak control on their own. **`saplma` is the one that carries the
check** -- it is trained on the pool, so an identical vector is real evidence the pool is identical.
The verdict therefore REFUSES to pass a cell in which no pool-dependent method was compared.

    python scripts/checks/wmodels_reduced_control.py
    python scripts/checks/wmodels_reduced_control.py --model meta-llama/Meta-Llama-3.1-8B --tol 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# The two rungs whose training-pool spec is unchanged by the six-dataset restriction. Do NOT add
# DiffTask/SameTask/LOO here: those genuinely differ, and a "control" that expected them to match
# would fail for the right reason in a way that reads as a bug.
CONTROL_RUNGS = ["ID", "1ds-Diff-long"]

PANEL6 = ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "asqa", "factscore"]

# Trained on the pool, so an identical vector is evidence the pool is identical.
POOL_DEPENDENT = {"saplma", "uniform", "attention", "wmsp_norm", "wmsp_shrink2"}


def _slug(model: str) -> str:
    return model.replace("/", "_")


def _methods(z) -> set:
    return {k[len("unc__"):] for k in z.keys() if k.startswith("unc__")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--evals", default=",".join(PANEL6))
    ap.add_argument("--reduced-dir", default=None,
                    help="default results/perex_wmodels/<slug>")
    ap.add_argument("--master-dir", default="results/pdl_perex")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max |Δ| tolerated per vector. Default 0 = bit-identical, which is what the "
                         "same computation on the same inputs should give. Raise it only with a "
                         "reason, and say what the reason is.")
    args = ap.parse_args()

    slug = _slug(args.model)
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]
    red_dir = Path(args.reduced_dir) if args.reduced_dir else ROOT / "results" / "perex_wmodels" / slug
    mas_dir = Path(args.master_dir)
    if not mas_dir.is_absolute():
        mas_dir = ROOT / mas_dir

    print("=== W-Models reduced-panel FREE CONTROL ===")
    print(f"  model    {args.model}")
    print(f"  rungs    {CONTROL_RUNGS}   (the two with an unchanged training-pool spec)")
    print(f"  reduced  {red_dir}")
    print(f"  master   {mas_dir}")
    print(f"  tol      {args.tol:g}\n")

    if not mas_dir.is_dir():
        print(f"!!! master sidecar directory not found: {mas_dir}", file=sys.stderr)
        return 2

    n_cells = n_cmp = 0
    n_pool_dep = 0
    worst = 0.0
    worst_where = "-"
    failures, not_run, uncomparable = [], [], set()

    for ev in evals:
        for rung in CONTROL_RUNGS:
            fname = f"{ev}__{rung}__{slug}.npz"
            red_p, mas_p = red_dir / fname, mas_dir / fname

            # A reduced cell that has not run yet is NOT a failure -- it is named and excluded, so a
            # partial re-score cannot masquerade as a complete pass.
            if not red_p.exists():
                not_run.append(f"{ev}/{rung}")
                continue
            if not mas_p.exists():
                failures.append(f"{ev}/{rung}: no MASTER sidecar at {mas_p.name} -- nothing to "
                                f"control against")
                continue

            red = np.load(red_p, allow_pickle=True)
            mas = np.load(mas_p, allow_pickle=True)

            # 1. THE PREMISE. If the training spec differs these are not the same cell.
            r_train, m_train = str(red["meta__train"]), str(mas["meta__train"])
            if r_train != m_train:
                failures.append(f"{ev}/{rung}: TRAINING SPEC DIFFERS -- reduced '{r_train}' vs "
                                f"master '{m_train}'. The rung restriction is wrong.")
                continue

            # 2. Same eval rows, same order, same seeds.
            if red["y"].shape != mas["y"].shape or not np.array_equal(red["y"], mas["y"]):
                failures.append(f"{ev}/{rung}: label vectors differ (shapes "
                                f"{red['y'].shape} vs {mas['y'].shape}) -- the eval split moved")
                continue
            if not np.array_equal(red["seeds"], mas["seeds"]):
                failures.append(f"{ev}/{rung}: seed lists differ, {red['seeds']} vs {mas['seeds']}")
                continue

            shared = sorted(_methods(red) & _methods(mas))
            uncomparable |= (_methods(red) - _methods(mas))
            if not shared:
                failures.append(f"{ev}/{rung}: no method is present in BOTH sidecars")
                continue

            cell_worst, cell_worst_m = 0.0, "-"
            cell_pool_dep = 0
            for m in shared:
                a, b = red[f"unc__{m}"], mas[f"unc__{m}"]
                if a.shape != b.shape:
                    failures.append(f"{ev}/{rung}/{m}: shape {a.shape} vs {b.shape}")
                    continue
                d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
                n_cmp += 1
                if m in POOL_DEPENDENT:
                    cell_pool_dep += 1
                    n_pool_dep += 1
                if d > cell_worst:
                    cell_worst, cell_worst_m = d, m
                if d > args.tol:
                    failures.append(f"{ev}/{rung}/{m}: max|Δ| = {d:.3e} > tol {args.tol:g}")

            # A cell in which only pool-INDEPENDENT methods were compared proves nothing about the
            # pool, so it must not be reported as a passing control.
            if cell_pool_dep == 0:
                failures.append(f"{ev}/{rung}: only training-pool-INDEPENDENT methods were "
                                f"comparable ({shared}); this cell cannot control the restriction")

            if cell_worst > worst:
                worst, worst_where = cell_worst, f"{ev}/{rung}/{cell_worst_m}"
            n_cells += 1
            print(f"  {ev:15s} {rung:15s} n={red['y'].shape[0]:5d}  methods={len(shared)} "
                  f"(pool-dep {cell_pool_dep})  max|Δ|={cell_worst:.3e}  train={r_train}")

    print("\n--- summary ---")
    print(f"  cells compared          {n_cells}")
    print(f"  vector comparisons      {n_cmp}  (of which pool-dependent: {n_pool_dep})")
    print(f"  worst max|Δ|            {worst:.10f}   "
          f"{'(every vector bit-identical)' if worst == 0.0 else 'at ' + worst_where}")
    if uncomparable:
        print(f"  NOT comparable          {sorted(uncomparable)} -- absent from the master sidecars, "
              f"so this control says nothing about them")
    if not_run:
        print(f"  not yet run ({len(not_run)})       {not_run}")

    if failures:
        print(f"\n!!! CONTROL FAILED -- {len(failures)} problem(s):", file=sys.stderr)
        for f in failures:
            print(f"    - {f}", file=sys.stderr)
        return 1

    if n_cells == 0:
        print("\n!!! nothing was compared -- this is NOT a pass", file=sys.stderr)
        return 1

    print(f"\n✅ CONTROL PASSED: the reduced panel reproduces the published master exactly on "
          f"{n_cells} unchanged-pool cells.")
    if not_run:
        print(f"   ⚠️ PARTIAL: {len(not_run)} cell(s) have not run yet and are excluded above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
