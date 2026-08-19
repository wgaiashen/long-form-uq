"""Reproduction control for the λ = 1.5 re-run (prereg D5).

Adding `wmsp_shrink1_5` to the WMSP registry must not disturb the two variants that were already
there. The re-run recomputes `wmsp_norm` and `wmsp_shrink2` from the SAME caches with the SAME seeds,
so both must come back **bit-identical** to the committed values. If they do not, the registry edit
changed something it should not have, and the λ = 1.5 numbers must not be read at all.

⚠️ COMPARES PER-EXAMPLE VECTORS, NOT PRR. PRR moves ~1e-4 on 1e-7 of round-off, so agreeing PRRs
would pass runs that are not the same computation.

⚠️ A cell present in one run and absent from the other is reported, never skipped silently -- that is
the "blank reads as measured" failure this project bans.

    python scripts/checks/wmodels_lambda_control.py --model google/gemma-2-9b
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CARRIED = ["wmsp_norm", "wmsp_shrink2"]   # must reproduce exactly
NEW = "wmsp_shrink1_5"                    # must be PRESENT in the new run and ABSENT from the old


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b")
    ap.add_argument("--old-dir", default=None, help="default results/perex_wmodels/<slug>")
    ap.add_argument("--new-dir", default=None, help="default results/perex_wmodels_lam3/<slug>")
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()

    slug = a.model.replace("/", "_")
    old = Path(a.old_dir) if a.old_dir else ROOT / "results/perex_wmodels" / slug
    new = Path(a.new_dir) if a.new_dir else ROOT / "results/perex_wmodels_lam3" / slug

    print("=== λ = 1.5 re-run REPRODUCTION CONTROL ===")
    print(f"  model {a.model}\n  old   {old}\n  new   {new}\n  tol   {a.tol:g}\n")
    if not new.is_dir():
        print(f"!!! new sidecar dir missing: {new}", file=sys.stderr); return 2

    n_cells = n_cmp = 0
    worst = 0.0
    fails, missing_new, missing_old = [], [], []

    for p_old in sorted(old.glob(f"*__{slug}.npz")):
        p_new = new / p_old.name
        if not p_new.exists():
            missing_new.append(p_old.stem); continue
        zo = np.load(p_old, allow_pickle=True); zn = np.load(p_new, allow_pickle=True)
        tag = p_old.name.replace(f"__{slug}.npz", "")

        if not np.array_equal(zo["y"], zn["y"]):
            fails.append(f"{tag}: label vectors differ -- different eval rows"); continue
        if not np.array_equal(zo["seeds"], zn["seeds"]):
            fails.append(f"{tag}: seed lists differ"); continue
        if str(zo["meta__train"]) != str(zn["meta__train"]):
            fails.append(f"{tag}: training spec differs -- '{zo['meta__train']}' vs "
                         f"'{zn['meta__train']}'"); continue

        # the new variant must actually be there, and must NOT be in the old run
        if f"unc__{NEW}" not in zn:
            fails.append(f"{tag}: {NEW} ABSENT from the new run -- the λ set did not reach the driver")
        if f"unc__{NEW}" in zo:
            fails.append(f"{tag}: {NEW} unexpectedly present in the OLD run")

        cell_worst = 0.0
        for m in CARRIED:
            ko = f"unc__{m}"
            if ko not in zo or ko not in zn:
                fails.append(f"{tag}: {m} missing from one run"); continue
            d = float(np.nanmax(np.abs(zo[ko] - zn[ko])))
            n_cmp += 1
            cell_worst = max(cell_worst, d)
            if d > a.tol:
                fails.append(f"{tag}/{m}: max|Δ| = {d:.3e} > tol {a.tol:g}")
        worst = max(worst, cell_worst)
        n_cells += 1
        print(f"  {tag:34s} n={zn['y'].shape[0]:5d}  carried max|Δ|={cell_worst:.3e}  "
              f"{NEW}={'present' if f'unc__{NEW}' in zn else 'MISSING'}")

    for p_new in sorted(new.glob(f"*__{slug}.npz")):
        if not (old / p_new.name).exists():
            missing_old.append(p_new.stem)

    print(f"\n--- summary ---\n  cells compared     {n_cells}\n  vector comparisons {n_cmp}")
    print(f"  worst max|Δ|       {worst:.10f}"
          f"{'   (every carried vector bit-identical)' if worst == 0.0 else ''}")
    if missing_new: print(f"  ⏳ not re-run yet ({len(missing_new)}): {missing_new}")
    if missing_old: print(f"  ℹ️ new only ({len(missing_old)}): {missing_old}")

    if fails:
        print(f"\n!!! CONTROL FAILED -- {len(fails)} problem(s):", file=sys.stderr)
        for f in fails: print(f"    - {f}", file=sys.stderr)
        print("    ⛔ Do NOT read or record any λ = 1.5 number.", file=sys.stderr)
        return 1
    if n_cells == 0:
        print("\n!!! nothing compared -- NOT a pass", file=sys.stderr); return 1
    print(f"\n✅ CONTROL PASSED on {n_cells} cells: λ=0 and λ=2 reproduce exactly, and "
          f"{NEW} is present only in the new run.")
    if missing_new:
        print(f"   ⚠️ PARTIAL: {len(missing_new)} cell(s) have not been re-run yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
