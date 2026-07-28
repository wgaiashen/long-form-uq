"""S1/P0 MIGRATION (2026-07-28): re-stamp the OOD attention sidecars whose rung label was STRIPPED of "-long".

`dump_ood_attention.py` used to write `base_rung = rung.replace("-long","")`, so a sidecar trained on the
LONG-ladder `LOO-long` / `DiffTask-long` pool was stamped (name + metadata) as the STANDARD "LOO"/"DiffTask"
rung -- a false assertion that let a later join silently mix long-pool and standard-pool quantities. The
writer is now fixed; this migrates the artifacts ALREADY on disk so name and content agree.

EVERY suffixed sidecar in cache/viz was written by dump_ood_attention (dump_viz_attention writes only the
unsuffixed ID sidecar), and dump_ood_attention ALWAYS builds LONG pools (cells_long/LONG_SRC), so every
stripped OOD suffix is a LONG-ladder rung -> append "-long" and stamp ladder_family="LONG".

This is a PURE RELABELLING: the pool_w / record_pos_all arrays are byte-identical before and after (asserted).
The .pkl poolers (dead C1 path) are only renamed (no torch unpickle on the login node); the sidecars carry
the authoritative rung.

    python scripts/tools/restamp_ood_sidecars.py            # dry-run (prints the plan)
    python scripts/tools/restamp_ood_sidecars.py --apply    # perform the migration
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIZ = ROOT / "cache" / "viz"
PROBES = ROOT / "cache" / "probes"
STRIPPED = {"LOO", "DiffTask", "SameTask", "OneDatasetDiffTask"}   # standard-looking names to correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform the migration (default: dry-run)")
    args = ap.parse_args()

    # --- sidecars (npz): re-stamp metadata + rename, asserting the payload is unchanged ---
    sidecars = []
    for p in sorted(glob.glob(str(VIZ / "*__attn__*.npz"))):
        name = Path(p).name
        suffix = name.split("__attn__", 1)[1][:-len(".npz")]      # e.g. "LOO", already "LOO-long", ...
        if suffix.endswith("-long") or suffix not in STRIPPED:
            continue                                              # already honest, or not a stripped OOD rung
        sidecars.append((p, suffix))

    print(f"sidecars to re-stamp: {len(sidecars)}")
    for p, suffix in sidecars:
        true_rung = f"{suffix}-long"
        newp = p.replace(f"__attn__{suffix}.npz", f"__attn__{true_rung}.npz")
        print(f"  {Path(p).name}  ->  {Path(newp).name}   (rung {suffix} -> {true_rung}, ladder_family=LONG)")
        if not args.apply:
            continue
        z = np.load(p, allow_pickle=True)
        payload = {k: z[k] for k in z.files}
        old_pool_w = payload["pool_w"]; old_rpos = payload["record_pos_all"]
        payload["rung"] = np.array(true_rung); payload["base_rung"] = np.array(suffix)
        payload["ladder_family"] = np.array("LONG")
        np.savez_compressed(newp, **payload)
        # PURE-RELABEL ASSERTION: the re-saved payload's arrays must be byte-identical to the original.
        chk = np.load(newp, allow_pickle=True)
        assert len(chk["pool_w"]) == len(old_pool_w) and np.array_equal(chk["record_pos_all"], old_rpos), \
            f"payload changed for {newp}!"
        for a, b in zip(chk["pool_w"], old_pool_w):
            assert np.array_equal(np.asarray(a), np.asarray(b)), f"pool_w changed for {newp}!"
        if os.path.abspath(newp) != os.path.abspath(p):
            os.remove(p)

    # --- pkls (saved poolers, dead C1 path): rename only ---
    pkls = []
    for p in sorted(glob.glob(str(PROBES / "*__attnpool_*.pkl"))):
        name = Path(p).name
        tag = name.split("__attnpool_", 1)[1].split("_s")[0]      # the rung tag before "_s<seed>"
        if tag.endswith("-long") or tag not in STRIPPED:
            continue
        pkls.append((p, tag))
    print(f"\npkls to rename: {len(pkls)}")
    for p, tag in pkls:
        newp = p.replace(f"__attnpool_{tag}_s", f"__attnpool_{tag}-long_s")
        print(f"  {Path(p).name}  ->  {Path(newp).name}")
        if args.apply and os.path.abspath(newp) != os.path.abspath(p):
            os.rename(p, newp)

    print("\n" + ("APPLIED." if args.apply else "DRY-RUN (pass --apply to migrate)."))


if __name__ == "__main__":
    main()
