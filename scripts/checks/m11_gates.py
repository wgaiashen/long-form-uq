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


def gate_d(args):
    """The cross-cluster sentinel: one layer computed independently on two machines.

    Registered in amendment A1.4. The earlier layer registration compared hidden states recomputed on
    a different accelerator against cached ones and stopped when they differed in their last bits. It
    never asked whether such a difference survives into a distance. This does, end to end, on the
    per-example mean distance AND on the relative distance, which is the quantity a background is
    subtracted from and therefore the one with the most room to amplify anything.
    """
    a_dir, b_dir = ROOT / args.a, ROOT / args.b
    for d in (a_dir, b_dir):
        if not d.is_dir():
            sys.exit(f"FATAL: {d} is not a directory. Gate D has nothing to compare.")

    pat = f"L{args.layer}__*.npz"
    a_files = {p.name for p in a_dir.glob(pat)}
    b_files = {p.name for p in b_dir.glob(pat)}
    common = sorted(a_files & b_files)
    print(f"layer {args.layer} | {args.a}: {len(a_files)} cells | {args.b}: {len(b_files)} cells | "
          f"comparable {len(common)}")
    if len(common) != args.expect_cells:
        print(f"  expected {args.expect_cells} comparable cells")
    if not common:
        sys.exit("FATAL: no cell of this layer appears on both sides.")

    cards, worst, n_cmp, bad = {}, {}, 0, []
    for key in ("md", "rmd"):
        worst[key] = (0.0, "")
    for name in common:
        a, b = np.load(a_dir / name, allow_pickle=True), np.load(b_dir / name, allow_pickle=True)
        for side, z in (("a", a), ("b", b)):
            if "card" in z.files:
                cards.setdefault(side, set()).add(str(z["card"]))
        wa = {str(z["window"]) for z in (a, b) if "window" in z.files}
        if len(wa) > 1:
            sys.exit(f"FATAL [{name}]: the two sides used different windows {sorted(wa)}.")
        seeds = [int(s) for s in np.asarray(a["seeds"]).ravel()]
        for sd in seeds:
            for key, keys in (("md", (f"dev_md__{sd}", f"test_md__{sd}")),
                              ("rmd", (f"dev_rmd__{sd}", f"test_rmd__{sd}"))):
                for k in keys:
                    if k not in a.files or k not in b.files:
                        continue
                    x, y = np.asarray(a[k], float), np.asarray(b[k], float)
                    if x.shape != y.shape:
                        bad.append(f"{name} {k}: shape {x.shape} vs {y.shape}")
                        continue
                    scale = np.maximum(np.abs(y), 1e-12)
                    rel = float(np.max(np.abs(x - y) / scale))
                    n_cmp += 1
                    if rel > worst[key][0]:
                        worst[key] = (rel, f"{name} {k}")

    print(f"accelerators: {args.a} = {sorted(cards.get('a', {'unrecorded'}))} | "
          f"{args.b} = {sorted(cards.get('b', {'unrecorded'}))}")
    print(f"\ncompared {n_cmp} vectors")
    for key in ("md", "rmd"):
        w, where = worst[key]
        print(f"  {key:<4} worst relative difference {w:.3e} at {where or 'nothing compared'}")
    print(f"pre-registered bar: {args.tol:.1e} relative, on both")
    for line in bad[:10]:
        print(f"  MISMATCH {line}")
    ok = (not bad) and all(worst[k][0] <= args.tol for k in worst) and n_cmp > 0
    if worst["rmd"][1] == "":
        print("  the relative distance was not present on both sides, so it was NOT tested")
        ok = False
    print("GATE D: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def gate_bg(args):
    """SUPERSEDED by gate_bgdist. Kept because it produced a recorded result.

    This compares two background statistics by their largest elementwise relative difference. It was
    the registered background check, it failed, and the failure was traced to the comparison rather
    than to the data: a perturbation that moves every distance by about one part in a million, and
    leaves their ordering exactly unchanged, registers as a factor of eighty here. It measures the
    conditioning of a near-singular matrix inverse, not the fidelity of a method. See amendment A2 of
    prereg/M11_alllayer_published_distance_baselines.md and
    scripts/checks/m11_bg_sensitivity.py.

    Do not use it to accept or reject a background. Use bgdist.

    The original description follows.

    The background verification of amendment A1.5.

    The layer-15 background can be derived from per-token states already on disk, with no model and no
    recomputation. The recomputed statistics that the other thirty-one layers rely on must reproduce
    that one. This is the only check available on the recomputation path.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from luq import mahalanobis as _MD
    status, budgets = 0, [int(b) for b in args.budgets.split(",") if b.strip()]
    for b in budgets:
        fa, fb = ROOT / args.exact.format(b=b), ROOT / args.recomputed.format(b=b)
        if not fa.exists() or not fb.exists():
            print(f"b{b}: missing {'exact' if not fa.exists() else 'recomputed'} statistic -> NOT CHECKED")
            status = 1
            continue
        x, y = _MD.load_stats(fa), _MD.load_stats(fb)
        dc = float(np.max(np.abs(x.centroid - y.centroid) / np.maximum(np.abs(x.centroid), 1e-12)))
        ds = float(np.max(np.abs(x.sigma_inv - y.sigma_inv) / np.maximum(np.abs(x.sigma_inv), 1e-12)))
        print(f"b{b}: centroid {dc:.3e} | sigma_inv {ds:.3e} | tokens {x.n_tokens} vs {y.n_tokens}")
        if max(dc, ds) > args.tol or x.n_tokens != y.n_tokens:
            status = 1
    print(f"\npre-registered bar: {args.tol:.1e} relative")
    print("BACKGROUND VERIFICATION: " + ("PASS" if status == 0 else "FAIL"))
    return status


def gate_bgdist(args):
    """Compare two background statistics by the DISTANCES they produce, not by their entries.

    This is what a background is for. The inverse covariance is consumed as a quadratic form over
    sixteen million terms, so the question that decides a result is whether the per-row distances
    move, and by how much.

    NOT A WEAKER TEST THAN THE ELEMENTWISE ONE. It is a different one, on the quantity the
    pre-registration says gates belong on. If the two backgrounds genuinely disagree in a way that
    matters, the distances move and this fails; the elementwise comparison can fail for conditioning
    of the matrix inverse alone, which no result depends on.

    Both statistics are scored against the SAME rows on one machine, so nothing here depends on where
    either was fitted.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from luq import mahalanobis as _MD
    from scipy.stats import spearmanr

    z = np.load(ROOT / args.states, allow_pickle=True)
    rows = []
    for st in z["states"]:
        st = np.asarray(st, dtype=np.float32)
        if len(st):
            rows.append(st)
    if args.n_score:
        rows = rows[:args.n_score]
    print(f"scoring {len(rows)} rows from {args.states}")

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    status = 0
    print(f"\n{'budget':>7} {'centroid rel':>13} {'sigma_inv rel':>14} | "
          f"{'DISTANCE rel':>13} {'rank corr':>10}  verdict")
    for b in budgets:
        fa, fb = ROOT / args.exact.format(b=b), ROOT / args.recomputed.format(b=b)
        if not fa.exists() or not fb.exists():
            print(f"{b:>7} {'missing a statistic':>40}")
            status = 1
            continue
        A, B = _MD.load_stats(fa), _MD.load_stats(fb)
        dc = float(np.max(np.abs(A.centroid - B.centroid) /
                          np.maximum(np.abs(A.centroid), 1e-12)))
        ds = float(np.max(np.abs(A.sigma_inv - B.sigma_inv) /
                          np.maximum(np.abs(A.sigma_inv), 1e-12)))
        sliced = [r[:b + 1] for r in rows]
        da, db = _MD.md_mean(sliced, A), _MD.md_mean(sliced, B)
        ok = np.isfinite(da) & np.isfinite(db)
        rel = float(np.max(np.abs(db[ok] - da[ok]) / np.maximum(np.abs(da[ok]), 1e-12)))
        rho = float(spearmanr(da[ok], db[ok]).statistic)
        good = rel <= args.tol
        status = status or (0 if good else 1)
        print(f"{b:>7} {dc:>13.3e} {ds:>14.3e} | {rel:>13.3e} {rho:>10.6f}  "
              + ("ok" if good else "FAIL"))

    print(f"\nbar: {args.tol:.1e} relative on the per-row distances")
    print("The two left columns are reported for the record, not gated on.")
    print("BACKGROUND DISTANCE CHECK: " + ("PASS" if status == 0 else "FAIL"))
    return status


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

    d = sub.add_parser("d", help="cross-cluster sentinel, one layer computed on two machines")
    d.add_argument("--a", required=True, help="scan directory from one cluster")
    d.add_argument("--b", required=True, help="scan directory from the other")
    d.add_argument("--layer", type=int, default=15)
    d.add_argument("--expect-cells", type=int, default=40)
    d.add_argument("--tol", type=float, default=1e-4)

    g = sub.add_parser("bg", help="background verification, recomputed against the exact path")
    g.add_argument("--exact", required=True, help="path template with {b} for the budget")
    g.add_argument("--recomputed", required=True, help="path template with {b} for the budget")
    g.add_argument("--budgets", default="56,128,256,384")
    g.add_argument("--tol", type=float, default=1e-4)

    bd = sub.add_parser("bgdist", help="two backgrounds compared by the distances they produce")
    bd.add_argument("--exact", required=True, help="path template with {b} for the budget")
    bd.add_argument("--recomputed", required=True, help="path template with {b} for the budget")
    bd.add_argument("--states", required=True, help="per-token states to score both against")
    bd.add_argument("--budgets", default="56,128,256,384")
    bd.add_argument("--n-score", type=int, default=400)
    bd.add_argument("--tol", type=float, default=1e-4)

    args = ap.parse_args()
    return {"b": gate_b, "c": gate_c, "d": gate_d, "bg": gate_bg,
            "bgdist": gate_bgdist}[args.gate](args)


if __name__ == "__main__":
    raise SystemExit(main())
