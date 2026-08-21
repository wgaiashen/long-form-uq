#!/usr/bin/env python
"""The XL grid's headline numbers, computed the SAME way as the Long grid's, so the two are comparable.

Mirrors the analysis written into the project results log for ProbeDriftLong:
  * ID mean and OOD mean per method, with the cell count behind each.
  * The comparison against BOTH bars — `msp_min` (pre-registered, the agreed bar) and the strongest
    free floor chosen PER DATASET. On the Long grid these disagree about whether training beats free at
    all, so reporting only one is not an option.
  * Per-dataset floor winners, because the standing rule is to report which floor won per cell.

REFUSES to report a method that does not cover the full grid, and refuses any cell that is not a finite
number. A mean over a partial cell set is not comparable to a mean over a full one, and quietly averaging
whatever happens to be present is how a cross-population comparison gets made.

    python scripts/checks/xl_headline.py
    python scripts/checks/xl_headline.py --allow-partial    # prints partial rows, clearly marked
"""
import argparse
import collections
import csv
import glob
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from cohort import CANONICAL_10, XL_RUNGS  # noqa: E402

RES = ROOT / "results"
OOD = [r for r in XL_RUNGS if r != "ID"]
# display label -> raw method name(s) as written by the drivers
SHOW = [("msp_min (floor)", ["msp_min"]), ("perplexity (floor)", ["perplexity"]),
        ("msp_sum (floor)", ["msp_sum"]), ("P(True)-unsup", ["ptrue_unsup"]),
        ("SAPLMA", ["mean-pool+MLP", "saplma"]), ("P(True) probe", ["ptrue_accurate", "ptrue"]),
        ("Lookback Lens", ["lookback"]), ("armB mean-pool", ["uniform"]),
        ("armA attention", ["attention"]), ("wMSP-norm", ["weighted_msp_norm"]),
        ("wMSP-unconstrained", ["weighted_msp_unc"]),
        ("last-token", ["last-token"]), ("per-sentence", ["per-sentence"]), ("per-token", ["per-token"])]


def load():
    C = collections.defaultdict(dict)
    for pat in ("xlcontrib_fam_*meta-llama*.csv", "xlonegrid_fam_*meta-llama*.csv"):
        for f in sorted(glob.glob(str(RES / pat))):
            for r in csv.DictReader(open(f)):
                rk = "rung" if r.get("rung") else "setting"
                try:
                    v = float(r["prr_mean"])
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(v):
                    C[(r[rk], r["eval"])][r["method"]] = v
    return C


def pick(cell, names):
    for n in names:
        if n in cell:
            return cell[n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    C = load()

    print(f"  {'method':22s} {'ID mean':>9s} {'n':>3s} {'OOD mean':>9s} {'n':>3s}  "
          f"{'vs msp_min':>11s} {'vs best free':>13s}")
    fm_ood = [C[(rg, e)]["msp_min"] for e in CANONICAL_10 for rg in OOD if "msp_min" in C.get((rg, e), {})]
    bf_ood = []
    for e in CANONICAL_10:
        cand = [C[("ID", e)].get(k) for k in ("msp_min", "perplexity", "msp_sum")]
        cand = [c for c in cand if c is not None]
        if cand:
            bf_ood += [max(cand)] * len(OOD)
    rows = []
    for lab, names in SHOW:
        idv = [v for e in CANONICAL_10 if (v := pick(C.get(("ID", e), {}), names)) is not None]
        ov = [v for e in CANONICAL_10 for rg in OOD if (v := pick(C.get((rg, e), {}), names)) is not None]
        full = len(idv) == len(CANONICAL_10) and len(ov) == len(CANONICAL_10) * len(OOD)
        if not full and not args.allow_partial:
            print(f"  {'SKIP ' + lab:22s} {'':>9s} {len(idv):3d} {'':>9s} {len(ov):3d}   "
                  f"PARTIAL — not comparable, use --allow-partial to see it")
            continue
        mark = "" if full else "  PARTIAL"
        i, o = (st.mean(idv) if idv else float('nan')), (st.mean(ov) if ov else float('nan'))
        rows.append((lab, o, full))
        print(f"  {lab:22s} {i:+9.4f} {len(idv):3d} {o:+9.4f} {len(ov):3d}  "
              f"{o - st.mean(fm_ood):+11.4f} {o - st.mean(bf_ood):+13.4f}{mark}")

    print("\n  per-dataset floors (ID rung; a floor is constant across rungs by construction):")
    print(f"    {'eval':15s} {'msp_min':>9s} {'perplexity':>11s} {'msp_sum':>9s}  strongest free")
    win = collections.Counter()
    for e in CANONICAL_10:
        c = C.get(("ID", e), {})
        vals = {k: c.get(k) for k in ("msp_min", "perplexity", "msp_sum")}
        if any(v is None for v in vals.values()):
            print(f"    {e:15s} (floors incomplete)")
            continue
        b = max(vals, key=vals.get)
        win[b] += 1
        print(f"    {e:15s} {vals['msp_min']:+9.4f} {vals['perplexity']:+11.4f} {vals['msp_sum']:+9.4f}"
              f"  {b} ({vals[b]:+.4f})")
    print(f"\n  floor wins across the {sum(win.values())} evals: {dict(win)}")


if __name__ == "__main__":
    main()
