#!/usr/bin/env python
"""Assemble the report-facing clean per-example sidecar view for one model, as symlinks.

WHY A VIEW AND NOT A COPY
-------------------------
The corrected-span correction touches exactly the cells whose eval target or training pool contains
med_quad. Every other cell was proved byte-identical to the uncorrected population by the
corrected-span identity gate, so its sidecar is INHERITED rather than recomputed. Copying the files
would duplicate several hundred MB and, worse, would create a second artifact that can silently drift
from its source. A symlink view keeps one copy of each vector and records where it came from.

WHAT IS PROVED HERE RATHER THAN ASSUMED
---------------------------------------
The 19/21 partition is recomputed from `cells_long` at run time -- a cell is "corrected" if and only if
med_quad is its eval target or appears in its training spec. It is NOT read off a stored list, and it is
NOT taken on trust from the identity gate. The two must agree; if the corrected directory holds a cell
this rule says is untouched (or vice versa) the script exits non-zero rather than building a view.

Every sidecar is also opened and checked for the keys the downstream ensemble read needs, so a
truncated or half-written npz fails here rather than producing a plausible number later.

    python scripts/checks/build_clean_perex_view.py --model meta-llama/Meta-Llama-3.1-8B \
        --corrected results/cleanv2/perex --inherited results/pdl_perex_ens \
        --out results/perex_clean__meta-llama_Meta-Llama-3.1-8B
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from probe_drift_long.dataset_configs import LONG_SRC  # noqa: E402
from probe_drift_long.ood_settings import cells_long  # noqa: E402

# The vectors the ensemble read consumes. A sidecar missing any of these is unusable for this purpose,
# so it is a hard failure and not a skipped cell.
REQUIRED = ["y", "seeds", "unc__saplma", "unc__wmsp_shrink2", "unc__attention",
            "unc__wmsp_norm", "unc__floor_min", "unc__floor_sum", "unc__floor_ppl"]

CORRECTED_SOURCE = "med_quad"


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            if limit and h.name and fh.tell() > limit:
                break
    return h.hexdigest()


def resolve_cells():
    """[(rung, eval, sources, touches_corrected)] for the 40-cell long grid."""
    srcs = set(LONG_SRC)
    out = []
    for rung, X, spec in cells_long(srcs, sorted(srcs)):
        if rung == "Long->Short":
            continue
        sources = [d for d, _ in spec]
        out.append((rung, X, sources, X == CORRECTED_SOURCE or CORRECTED_SOURCE in sources))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corrected", required=True, help="dir holding the recomputed corrected-span cells")
    ap.add_argument("--inherited", required=True, help="dir holding the uncorrected-population cells")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slug = cache._slug(args.model)
    corrected_dir = ROOT / args.corrected
    inherited_dir = ROOT / args.inherited
    out_dir = ROOT / args.out
    for d in (corrected_dir, inherited_dir):
        if not d.is_dir():
            sys.exit(f"FATAL: not a directory: {d}")

    cells = resolve_cells()
    n_corr = sum(1 for c in cells if c[3])
    print(f"model {args.model} (slug {slug})")
    print(f"grid: {len(cells)} cells | corrected (touch {CORRECTED_SOURCE}): {n_corr} | "
          f"inherited: {len(cells) - n_corr}")
    if len(cells) != 40:
        sys.exit(f"FATAL: expected 40 cells, resolved {len(cells)}")

    # Cross-check the partition against what the corrected directory actually holds, in BOTH directions.
    on_disk = {p.name for p in corrected_dir.glob(f"*__{slug}.npz")}
    expected = {f"{X}__{rung}__{slug}.npz" for rung, X, _, corr in cells if corr}
    if on_disk != expected:
        only_disk, only_rule = sorted(on_disk - expected), sorted(expected - on_disk)
        print(f"FATAL: corrected directory disagrees with the cells_long partition.")
        if only_disk:
            print(f"  present on disk but the rule says untouched: {only_disk}")
        if only_rule:
            print(f"  the rule says corrected but absent from disk: {only_rule}")
        sys.exit(1)
    print(f"partition cross-check: corrected directory matches the cells_long rule exactly "
          f"({len(expected)} files)")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, missing = [], []
    for rung, X, sources, corr in sorted(cells):
        fname = f"{X}__{rung}__{slug}.npz"
        src = (corrected_dir if corr else inherited_dir) / fname
        if not src.exists():
            missing.append((fname, "corrected" if corr else "inherited"))
            continue
        # Open it. A sidecar that cannot be read, or lacks a required vector, is a hard failure --
        # never a silently skipped cell, which would read downstream as "not measured".
        with np.load(src, allow_pickle=True) as z:
            keys = set(z.keys())
            lack = [k for k in REQUIRED if k not in keys]
            if lack:
                sys.exit(f"FATAL {fname}: missing required vectors {lack}")
            n_seeds, n_te = z["unc__saplma"].shape
            n_y = len(z["y"])
            if n_te != n_y:
                sys.exit(f"FATAL {fname}: {n_te} score columns vs {n_y} labels")
        link = out_dir / fname
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src.resolve())
        manifest.append({"cell": fname, "rung": rung, "eval": X, "sources": sources,
                         "population": "corrected" if corr else "inherited",
                         "source_path": str(src.relative_to(ROOT)),
                         "sha256": sha256(src), "n_seeds": int(n_seeds), "n_test": int(n_te)})

    if missing:
        print(f"\nFATAL: {len(missing)} sidecars absent -- the view is INCOMPLETE and is not written "
              f"as if it were whole:")
        for f, w in missing:
            print(f"  {f}  (expected in the {w} directory)")
        sys.exit(1)

    mpath = out_dir / "PROVENANCE.json"
    mpath.write_text(json.dumps({"model": args.model, "slug": slug,
                                 "corrected_dir": str(corrected_dir.relative_to(ROOT)),
                                 "inherited_dir": str(inherited_dir.relative_to(ROOT)),
                                 "corrected_source": CORRECTED_SOURCE,
                                 "n_cells": len(manifest),
                                 "n_corrected": sum(1 for m in manifest if m["population"] == "corrected"),
                                 "cells": manifest}, indent=2))
    print(f"\nview complete: {len(manifest)}/40 cells linked into {out_dir.relative_to(ROOT)}")
    print(f"provenance written to {mpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
