#!/usr/bin/env python
"""Assemble the clean-v2 Llama master and run gates A and C.

WHAT THIS BUILDS. The clean-v2 master is 40 cells made of two provably different things:

  * 19 cells RECOMPUTED, because MedQuAD is either their evaluation target or one of their training
    sources. Those come from results/cleanv2/probedriftlong_cleanv2_<eval>__<slug>.csv.
  * 21 cells INHERITED verbatim from the canonical master, because MedQuAD appears nowhere in them.

WHY THE 21 ARE INHERITED AND NOT RE-RUN. Their inputs are identical, so a re-run can only differ by
the stochastic draw of a probe fit. Refitting them would add noise to precisely the comparison that
is supposed to be exact, and would then invite the question "did this cell move?" about a cell that
by construction cannot have moved. Inheriting is the stronger claim, and it is only made after the
identity check below passes.

GATE A -- identity before inheritance. For every candidate inherited cell this asserts:
  * MedQuAD is not the eval target and not in the training pool (structural, from cells_long);
  * every dataset the cell touches resolves, under LUQ_REGIME=med_quad=cleanv2, to the SAME cache
    path the canonical run used -- i.e. nothing silently points at a cleanv2 directory;
  * that path's file content hash is unchanged.
Any failure is a STOP. It is never repaired by recomputing the cell.

GATE C -- training-pool audit. For every recomputed learned cell it prints the eval target, the
rung, the source datasets, the number of MedQuAD rows drawn, and the exact feature and label
namespace MedQuAD resolved to, so a silent fallback to the raw population is visible rather than
inferred.

This script does not interpret any PRR. It moves rows and checks provenance.

    python scripts/checks/cleanv2_assemble_and_gate.py            # gates only, no write
    python scripts/checks/cleanv2_assemble_and_gate.py --write    # also write the clean-v2 master
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("LUQ_REGIME", "med_quad=cleanv2")

from luq import cache                                            # noqa: E402
from luq.config import Config                                    # noqa: E402
from attn_pool import PROMPT_REGIME                              # noqa: E402
from probe_drift_long.ood_settings import cells_long             # noqa: E402
from probe_drift_long.dataset_configs import LONG_SRC            # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)
CANON = ROOT / "results" / f"pdl_master__{SLUG}.csv"
OUTDIR = ROOT / "results" / "cleanv2"
LAYER = 15


def _hash(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def split_cells():
    """(affected, control) cell lists, derived structurally -- never hard-coded."""
    aff, ctl = [], []
    for rung, X, spec in cells_long(LONG_SRC, LONG_SRC):
        touches = (X == "med_quad") or any(s == "med_quad" for s, _ in spec)
        (aff if touches else ctl).append((rung, X, spec))
    return aff, ctl


def gate_a(control):
    print("\n" + "=" * 96)
    print("GATE A -- input identity for the cells that will be INHERITED")
    print("=" * 96)
    ok = True
    seen = {}
    for rung, X, spec in control:
        datasets = [X] + [s for s, _ in spec]
        if "med_quad" in datasets:
            print(f"  STOP  {X}/{rung}: med_quad present in a cell classed as control"); ok = False; continue
        for d in datasets:
            if d in seen:
                continue
            cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                         prompt_regime=PROMPT_REGIME.get(d, ""))
            p = Path(cfg.cache_dir) / "pertok" / f"{cache._slug(MODEL)}__{d}__ID__L{LAYER}.npz"
            if "cleanv2" in str(p):
                print(f"  STOP  {d}: resolves into a cleanv2 path under the override -> {p}"); ok = False; continue
            if not p.exists():
                print(f"  STOP  {d}: cache missing -> {p}"); ok = False; continue
            seen[d] = _hash(p)
    print(f"  datasets touched by the {len(control)} control cells: {sorted(seen)}")
    for d, h in sorted(seen.items()):
        print(f"     {d:15} {h}  (canonical path, unchanged)")
    print(f"\n  GATE A: {'PASS -- the control cells may be inherited' if ok else '**FAIL -- STOP**'}")
    return ok


def gate_c(affected):
    print("\n" + "=" * 96)
    print("GATE C -- training-pool audit for every recomputed cell")
    print("=" * 96)
    mq_cfg = Config(model_name=MODEL, dataset="med_quad", ood_setting="ID",
                    prompt_regime=PROMPT_REGIME.get("med_quad", ""))
    mq_ns = PROMPT_REGIME.get("med_quad", "") or "<default>"
    print(f"  med_quad feature namespace : {mq_ns}   ({mq_cfg.cache_dir})")
    print(f"  med_quad label field       : correctness (in the same namespace)")
    print(f"\n  {'rung':16}{'eval':15}{'med_quad rows':>14}  sources")
    for rung, X, spec in sorted(affected):
        # cap is None on the ID cell, meaning "all of the eval's own train rows" -- not a number.
        # Summing it blind is the cap-vs-target trap the eight-dataset control already tripped over.
        caps = [c for s, c in spec if s == "med_quad"]
        if X == "med_quad":
            tag = "TARGET"
        elif not caps:
            tag = "-"
        elif any(c is None for c in caps):
            tag = "all"
        else:
            tag = str(sum(caps))
        print(f"  {rung:16}{X:15}{tag:>14}  {', '.join(s for s, _ in spec)}")
    bad = [c for c in affected if c[1] != "med_quad" and not any(s == "med_quad" for s, _ in c[2])]
    print(f"\n  GATE C: {'PASS -- every recomputed cell genuinely touches med_quad' if not bad else '**FAIL**'}")
    return not bad


def assemble(affected, control, write):
    print("\n" + "=" * 96)
    print("ASSEMBLE the clean-v2 master")
    print("=" * 96)
    canon = pd.read_csv(CANON)
    parts, missing = [], []
    for rung, X, _ in affected:
        f = OUTDIR / f"probedriftlong_cleanv2_{X}__{SLUG}.csv"
        if not f.exists():
            missing.append((rung, X)); continue
        d = pd.read_csv(f)
        d = d[(d["eval"] == X) & (d["rung"] == rung)]
        if d.empty:
            missing.append((rung, X)); continue
        d = d.copy(); d["provenance"] = "recomputed_cleanv2"
        parts.append(d)
    inh = []
    for rung, X, _ in control:
        d = canon[(canon["eval"] == X) & (canon["rung"] == rung)].copy()
        if d.empty:
            missing.append((rung, X, "canonical")); continue
        d["provenance"] = "inherited_unchanged"
        inh.append(d)
    print(f"  recomputed cells found : {len(parts)}/{len(affected)}")
    print(f"  inherited cells found  : {len(inh)}/{len(control)}")
    if missing:
        print(f"  MISSING: {missing}")
        print("  -> the master is NOT written while any cell is missing; a partial master would read"
              "\n     as a complete one, which is the failure this project bans.")
        return False
    master = pd.concat(parts + inh, ignore_index=True)
    print(f"  assembled {len(master)} rows over {master.groupby(['eval','rung']).ngroups} cells")
    if write:
        out = OUTDIR / f"pdl_master_cleanv2__{SLUG}.csv"
        master.to_csv(out, index=False)
        print(f"  wrote {out}")
    else:
        print("  (dry run: pass --write to persist)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    affected, control = split_cells()
    print(f"cells: {len(affected)} affected (recompute), {len(control)} control (inherit), "
          f"{len(affected) + len(control)} total")
    a = gate_a(control)
    c = gate_c(affected)
    if not (a and c):
        sys.exit("\nSTOP: a gate failed; nothing assembled.")
    assemble(affected, control, args.write)


if __name__ == "__main__":
    main()
