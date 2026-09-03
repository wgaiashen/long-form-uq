"""Which machine produced each layer, and did every cell of a layer agree?

This is the provenance diagnostic of prereg/M11_alllayer_published_distance_baselines.md section 5.
It is a REPORT, never a stop rule.

WHY IT IS A REPORT. Hidden states recomputed on a different accelerator generation differ from cached
ones in their last bits. That is a fact about where a tensor was computed, not about whether a method
was implemented correctly, and the acceptance gates that can stop this experiment are placed on the
per-example distances and on the downstream rows instead. The measurement is still recorded, because
a layer set assembled on more than one kind of card should say so rather than leave a reader to
reconstruct it from job logs.

The hidden-state comparison itself was already made, by scripts/checks/m9_extraction_gate.py, and its
numbers are quoted in the write-up from that run. Nothing here re-measures it.

    python scripts/checks/m11_provenance.py --scan results/hybrids/mdscan_refwin__<slug>
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
    ap.add_argument("--expect-cells", type=int, default=40)
    args = ap.parse_args()

    scan = ROOT / args.scan
    if not scan.is_dir():
        sys.exit(f"FATAL: {scan} is not a directory.")

    per_layer = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    counts = collections.Counter()
    for p in sorted(scan.glob("L*__*.npz")):
        layer = int(p.name.split("__", 1)[0][1:])
        counts[layer] += 1
        z = np.load(p, allow_pickle=True)
        for field in ("card", "commit", "window", "has_background"):
            if field in z.files:
                per_layer[layer][field][str(z[field])] += 1

    if not counts:
        sys.exit(f"FATAL: no cell files in {scan}.")

    print(f"{len(counts)} layers | {sum(counts.values())} cell files\n")
    print(f"{'layer':>5}  {'cells':>5}  {'window':<8} {'bg':>4}  accelerator")
    split = []
    for L in sorted(counts):
        f = per_layer[L]
        card = f["card"].most_common()
        win = "/".join(sorted(f["window"]))
        bg = f["has_background"].most_common(1)
        bg_s = "all" if bg and bg[0][0] == "1" and bg[0][1] == counts[L] else "PART"
        cards = ", ".join(f"{c} ({n})" for c, n in card) or "unrecorded"
        flag = ""
        if counts[L] != args.expect_cells:
            flag += f"  <- {counts[L]} of {args.expect_cells} cells"
        if len(card) > 1:
            flag += "  <- MIXED within the layer"
            split.append(L)
        print(f"{L:>5}  {counts[L]:>5}  {win:<8} {bg_s:>4}  {cards}{flag}")

    all_cards = collections.Counter()
    for L in per_layer:
        all_cards.update(per_layer[L]["card"])
    print("\naccelerators across the layer set:")
    for c, n in all_cards.most_common():
        print(f"  {c}: {n} cell files")
    if len(all_cards) > 1:
        print("\nThe layer set was assembled on more than one kind of accelerator. Report this. It is\n"
              "not a defect: the acceptance gates are on the distances, not on the hidden states.")
    if split:
        print(f"\nLayers whose own cells disagree about the accelerator: {split}. That means a layer\n"
              "was scanned in more than one job, which is allowed but worth knowing.")

    short = [L for L in counts if counts[L] != args.expect_cells]
    if short:
        print(f"\nINCOMPLETE LAYERS: {sorted(short)}. These must not enter a combined result.")
    else:
        print(f"\nEvery layer has {args.expect_cells} cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
