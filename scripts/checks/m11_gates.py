"""The acceptance gates for the full-layer reproduction.

Registered in prereg/M11_alllayer_published_distance_baselines.md section 5. Gate A is a diagnostic
and is reported by scripts/checks/m9_extraction_gate.py; the two gates here are stop rules.

GATE B, DISTANCE INVARIANCE. The per-example mean distance vector at layer 15 produced by the layer
scan must agree with the vector the single-layer driver computes, to a relative tolerance of 1e-4.
The comparison is on vectors, never on a rank statistic derived from them: a rank statistic moves on
round-off that a distance does not care about, and it can also stay put while individual scores move,
so it answers neither question. Both sides must have been run under the SAME window; a window
difference is a deliberate change and would swamp everything else.

GATE C, PIPELINE INVARIANCE. The layer-combining pass restricted to layer 15 alone must reproduce the
corrected single-layer rows already on record, including which cells were left blank for degeneracy.
A pipeline that cannot reproduce the one-layer result it extends cannot be trusted on the layers that
have nothing to check against.

    python scripts/checks/m11_gates.py b --scan results/hybrids/mdscan__<slug> \\
        --dump results/hybrids/mdhybdump__<slug>
    python scripts/checks/m11_gates.py c --control results/hybrids/pdl_m11_1layer__<slug>.csv \\
        --reference results/hybrids/pdl_hybrids_unconstrained__<slug>.csv
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# The single-layer driver's names for the two methods the combining pass also produces.
METHOD_MAP = {"satmd": "satmd_mid", "huq_satmd": "huq_satmd_mid"}


def gate_b(args):
    scan, dump = ROOT / args.scan, ROOT / args.dump
    for d in (scan, dump):
        if not d.is_dir():
            sys.exit(f"FATAL: {d} is not a directory. Gate B has nothing to compare.")

    scan_files = {p.name for p in scan.glob("L*__*.npz")}
    dump_files = {p.name for p in dump.glob("L*__*.npz")}
    common = sorted(scan_files & dump_files)
    only_scan, only_dump = sorted(scan_files - dump_files), sorted(dump_files - scan_files)
    print(f"scan {len(scan_files)} cells | driver dump {len(dump_files)} cells | "
          f"comparable {len(common)}")
    if only_scan or only_dump:
        print(f"  only in the scan: {len(only_scan)} | only in the driver dump: {len(only_dump)}")
    if not common:
        sys.exit("FATAL: no cell appears on both sides.")

    worst, worst_where, n_cmp, bad = 0.0, "", 0, []
    windows = set()
    for name in common:
        a, b = np.load(scan / name, allow_pickle=True), np.load(dump / name, allow_pickle=True)
        for z in (a, b):
            if "window" in z.files:
                windows.add(str(z["window"]))
        seeds = [int(s) for s in np.asarray(a["seeds"]).ravel()]
        for sd in seeds:
            for key in (f"dev_md__{sd}", f"test_md__{sd}"):
                if key not in a.files or key not in b.files:
                    continue
                x, y = np.asarray(a[key], float), np.asarray(b[key], float)
                if x.shape != y.shape:
                    bad.append(f"{name} {key}: shape {x.shape} vs {y.shape}")
                    continue
                scale = np.maximum(np.abs(y), 1e-12)
                rel = float(np.max(np.abs(x - y) / scale))
                n_cmp += 1
                if rel > worst:
                    worst, worst_where = rel, f"{name} {key}"
    if len(windows) > 1:
        sys.exit(f"FATAL: the two sides were run under different windows {sorted(windows)}. "
                 "Gate B compares implementations, not windows.")

    print(f"\ncompared {n_cmp} vectors | worst relative difference {worst:.3e} at {worst_where}")
    print(f"pre-registered bar: {args.tol:.1e} relative")
    for line in bad[:10]:
        print(f"  MISMATCH {line}")
    ok = (not bad) and worst <= args.tol
    print("GATE B: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _rows(path, methods):
    out = {}
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            m = r["method"]
            if m not in methods:
                continue
            out[(r["eval"], r["rung"], methods[m] if isinstance(methods, dict) else m)] = r
    return out


def gate_c(args):
    ctrl_path, ref_path = ROOT / args.control, ROOT / args.reference
    ctrl_raw, ref_raw = {}, {}
    with open(ctrl_path) as fh:
        for r in _csv.DictReader(fh):
            base = r["method"].rsplit("_", 1)[0] if "_" in r["method"] else r["method"]
            if base in METHOD_MAP:
                ctrl_raw[(r["eval"], r["rung"], METHOD_MAP[base])] = r
    with open(ref_path) as fh:
        for r in _csv.DictReader(fh):
            if r["method"] in METHOD_MAP.values():
                ref_raw[(r["eval"], r["rung"], r["method"])] = r

    keys = sorted(set(ctrl_raw) & set(ref_raw))
    missing = sorted(set(ref_raw) - set(ctrl_raw))
    print(f"control {len(ctrl_raw)} rows | reference {len(ref_raw)} rows | comparable {len(keys)}")
    if missing:
        print(f"  absent from the control: {len(missing)}")
        for k in missing[:8]:
            print(f"    {k}")

    worst, worst_where, bad = 0.0, "", []
    for k in keys:
        c, r = ctrl_raw[k], ref_raw[k]
        # A blank on either side is a degeneracy decision, and the two sides must agree about which
        # cells are blank. A number opposite a blank is a disagreement about whether a cell was
        # measurable at all, which matters more than any tolerance.
        cb, rb = (c["prr_mean"] == ""), (r["prr_mean"] == "")
        if cb != rb:
            bad.append(f"{k}: blank on one side only (control blank {cb}, reference blank {rb})")
            continue
        if cb:
            continue
        d = abs(float(c["prr_mean"]) - float(r["prr_mean"]))
        if d > worst:
            worst, worst_where = d, str(k)
        if int(c["n_seeds"]) != int(r["n_seeds"]):
            bad.append(f"{k}: seed count {c['n_seeds']} vs {r['n_seeds']}")

    print(f"\nworst absolute difference in PRR {worst:.3e} at {worst_where}")
    print(f"pre-registered bar: {args.tol:.1e} absolute, and identical degeneracy decisions")
    for line in bad[:10]:
        print(f"  MISMATCH {line}")
    ok = (not bad) and (not missing) and worst <= args.tol
    print("GATE C: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="gate", required=True)

    b = sub.add_parser("b", help="distance invariance, on per-example vectors")
    b.add_argument("--scan", required=True)
    b.add_argument("--dump", required=True)
    b.add_argument("--tol", type=float, default=1e-4)

    c = sub.add_parser("c", help="pipeline invariance, against the recorded single-layer rows")
    c.add_argument("--control", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--tol", type=float, default=1e-6)

    args = ap.parse_args()
    return gate_b(args) if args.gate == "b" else gate_c(args)


if __name__ == "__main__":
    raise SystemExit(main())
