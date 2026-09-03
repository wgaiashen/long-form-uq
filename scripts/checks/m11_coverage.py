"""Is every layer of the requested set complete, before anything is combined?

A layer present at some cells and absent at others would give the regression a different feature set
per cell, which is not the registered method. A layer missing entirely would silently shrink the
layer set, turning a reproduction into a sensitivity without anyone deciding to. Neither is visible
in the combined output afterwards, so both are checked here first.

The count that matters is per layer: eight evaluation datasets by five training conditions is forty
cells, and a layer with thirty-nine is not a slightly smaller layer, it is an incomplete one.

    python scripts/checks/m11_coverage.py --scan results/hybrids/mdscan_refwin__<slug> \\
        --layers 0,1,2,...,30,32
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--expect-cells", type=int, default=40)
    ap.add_argument("--expect-seeds", type=int, default=3)
    ap.add_argument("--require-rmd", action="store_true", default=True,
                    help="the relative distance must be present, since the relative family is part "
                         "of what this reproduces")
    args = ap.parse_args()

    scan = ROOT / args.scan
    if not scan.is_dir():
        sys.exit(f"FATAL: no scan directory at {scan}. Nothing to combine.")
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    cells = collections.defaultdict(set)
    short_seeds, no_rmd = [], []
    for p in sorted(scan.glob("L*__*.npz")):
        stem = p.name.split("__", 1)
        layer = int(stem[0][1:])
        cells[layer].add(stem[1])
        z = np.load(p, allow_pickle=True)
        seeds = [int(s) for s in np.asarray(z["seeds"]).ravel()]
        present = [sd for sd in seeds if f"test_md__{sd}" in z.files]
        if len(present) != args.expect_seeds:
            short_seeds.append(f"L{layer} {stem[1]}: {len(present)} of {args.expect_seeds} seeds")
        if args.require_rmd and not any(f"test_rmd__{sd}" in z.files for sd in seeds):
            no_rmd.append(f"L{layer} {stem[1]}")

    missing_layers = [L for L in layers if L not in cells]
    incomplete = [(L, len(cells[L])) for L in layers if L in cells
                  and len(cells[L]) != args.expect_cells]

    print(f"requested {len(layers)} layers | present {len([L for L in layers if L in cells])}")
    if missing_layers:
        print(f"\nMISSING ENTIRELY: {missing_layers}")
    if incomplete:
        print(f"\nINCOMPLETE LAYERS (expected {args.expect_cells} cells):")
        for L, n in incomplete:
            print(f"  L{L}: {n} cells")
    if short_seeds:
        print(f"\n{len(short_seeds)} cell(s) short of {args.expect_seeds} seeds:")
        for line in short_seeds[:10]:
            print(f"  {line}")
    if no_rmd:
        print(f"\n{len(no_rmd)} cell(s) carry no relative distance:")
        for line in no_rmd[:10]:
            print(f"  {line}")

    # A cell missing from only some layers is the subtle case: the combining pass would drop it, so
    # the result would rest on fewer cells than the grid without saying so.
    if cells:
        every = set.intersection(*(cells[L] for L in layers if L in cells))
        union = set().union(*(cells[L] for L in layers if L in cells))
        ragged = sorted(union - every)
        if ragged:
            print(f"\n{len(ragged)} cell(s) are absent from at least one layer and would be dropped "
                  f"from the combination:")
            for c in ragged[:10]:
                print(f"  {c}")
        print(f"\ncells present at EVERY requested layer: {len(every)} of {args.expect_cells}")

    ok = not (missing_layers or incomplete or short_seeds or no_rmd)
    print("\nCOVERAGE: " + ("PASS, the layer set is complete" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
