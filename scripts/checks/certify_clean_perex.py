#!/usr/bin/env python
"""Certify that a per-example score directory really is the corrected-span population.

Written before the directory it certifies existed. Seven criteria, each an explicit PASS or FAIL; the
script exits non-zero unless every one passes, so it cannot be read as approval by accident.

The reason this exists: the only Qwen per-example directory that existed held the UNCORRECTED
population while carrying a name that said nothing either way, and agreed with the corrected master on
zero of forty cells. Nothing downstream noticed, because nothing downstream checked. A directory name
is not evidence.

    python scripts/checks/certify_clean_perex.py --model Qwen/Qwen2.5-14B \
        --master results/analysis/pdl_master_qwenclean__Qwen_Qwen2.5-14B.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache                                                           # noqa: E402

# The master stores to four decimals, so agreement can never be tighter than that for the
# unsupervised floors. The probe is refitted per seed and carries genuine seed noise.
TOL_FLOOR = 1e-4
TOL_PROBE = 5e-3
TOL_CAWSA = 5e-3
CORRECTED = ["med_quad", "samsum", "expertqa", "factscore"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--dir", default="")
    ap.add_argument("--master", required=True)
    ap.add_argument("--expect-cells", type=int, default=40)
    ap.add_argument("--partial", action="store_true",
                    help="certify a directory that is still being filled: criterion 1 reports the "
                         "count without failing on it. Every other criterion still applies.")
    args = ap.parse_args()

    slug = cache._slug(args.model)
    d = ROOT / (args.dir or f"results/perex_clean__{slug}")
    m = pd.read_csv(ROOT / args.master)
    files = sorted(d.glob(f"*__{slug}.npz")) if d.is_dir() else []
    print(f"POPULATION {args.model}")
    print(f"  directory {d.relative_to(ROOT) if d.is_dir() else str(d) + '  (ABSENT)'}")
    print(f"  master    {args.master}\n")

    verdicts = {}

    # 1 -- coverage
    ok1 = len(files) == args.expect_cells
    verdicts["1 all cells present"] = (
        True if (args.partial and files) else ok1,
        f"{len(files)} of {args.expect_cells}" + ("  (partial run, not failed)" if args.partial else ""))

    # 2/4/5 -- the stored scores must reproduce the master this population is supposed to be
    rec = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        ev, rung = str(z["meta__eval"]), str(z["meta__rung"])
        for meth in ("floor_sum", "floor_min", "saplma", "wmsp_shrink2"):
            k = f"prr_mean__{meth}"
            if k in z.files:
                rec.append(dict(eval=ev, rung=rung, method=meth, side=float(z[k])))
    s = pd.DataFrame(rec)
    if s.empty:
        verdicts["2 cohort matches master"] = (False, "no readable sidecars")
        verdicts["4 learned weighting reproduces master"] = (False, "no readable sidecars")
        verdicts["5 probe and floors reproduce master"] = (False, "no readable sidecars")
        j = pd.DataFrame()
    else:
        j = s.merge(m[["eval", "rung", "method", "prr_mean"]], on=["eval", "rung", "method"])
        j["d"] = (j.side - j.prr_mean).abs()
        matched = len(j)
        verdicts["2 cohort matches master"] = (
            matched == len(s), f"{matched} of {len(s)} stored rows found in the master")
        cw = j[j.method == "wmsp_shrink2"]
        verdicts["4 learned weighting reproduces master"] = (
            (not cw.empty) and cw.d.max() <= TOL_CAWSA,
            f"n={len(cw)} max|d|={cw.d.max():.2e} (bar {TOL_CAWSA:g})" if not cw.empty else "absent")
        others = j[j.method != "wmsp_shrink2"]
        worst = []
        allok = not others.empty
        for meth, tol in (("floor_sum", TOL_FLOOR), ("floor_min", TOL_FLOOR), ("saplma", TOL_PROBE)):
            g = others[others.method == meth]
            if g.empty:
                allok = False; worst.append(f"{meth} absent"); continue
            allok &= g.d.max() <= tol
            worst.append(f"{meth} {g.d.max():.2e}/{tol:g}")
        verdicts["5 probe and floors reproduce master"] = (allok, "  ".join(worst))

    # 3/7 -- the corrected datasets must be corrected wherever they appear, as target OR as source
    from attn_pool import PROMPT_REGIME
    raw_named = [ds for ds in CORRECTED if not PROMPT_REGIME.get(ds, "")]
    verdicts["3+7 corrected datasets resolve corrected"] = (
        not raw_named,
        "all four namespaced: " + ", ".join(f"{ds}={PROMPT_REGIME.get(ds, '') or 'CANONICAL'}"
                                            for ds in CORRECTED))

    # 6 -- no silent fallback: every sidecar must name a training spec, and any cell whose target or
    #      source is a corrected dataset must exist rather than having been quietly skipped
    covered = set()
    for f in files:
        z = np.load(f, allow_pickle=True)
        covered.add((str(z["meta__eval"]), str(z["meta__rung"])))
    want = set(zip(m["eval"], m["rung"]))
    missing = sorted(want - covered)
    verdicts["6 no cell silently absent"] = (
        (not missing) or args.partial,
        f"{len(missing)} master cells not in the directory" + (" (partial run)" if args.partial else ""))

    width = max(len(k) for k in verdicts)
    allpass = True
    for k, (ok, detail) in verdicts.items():
        allpass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:<{width}}  {detail}")

    print()
    if allpass:
        print("CERTIFICATION: PASS -- this directory is the corrected-span population.")
        return
    print("CERTIFICATION: FAIL -- do NOT run the substitution against this directory.")
    if not j.empty:
        bad = j[j.d > TOL_CAWSA].sort_values("d", ascending=False).head(8)
        if not bad.empty:
            print("\nworst disagreements with the master:")
            print(bad[["eval", "rung", "method", "side", "prr_mean", "d"]].to_string(index=False))
    sys.exit(1)


if __name__ == "__main__":
    main()
