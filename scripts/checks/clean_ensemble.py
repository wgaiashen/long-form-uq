#!/usr/bin/env python
"""Run the CAWSA + SAPLMA complementarity read against a CORRECTED-SPAN master.

WHY THIS WRAPPER EXISTS
-----------------------
`complementary_ensemble.py` gates every component against `results/pdl_master__<slug>.csv`, the
uncorrected-population master. That is the right reference for the uncorrected population and the wrong
one here: on the corrected-span population the 19 cells whose eval target or training pool contains
med_quad are *supposed* to differ, so running the existing gate unchanged reports 19 spurious failures
and refuses to produce a result.

The gate is not relaxed and no tolerance is widened. Only the reference file changes, to the master that
actually describes the population being read. Everything else -- coverage, seed handling, combiners,
gate 2, the pre-registered estimands -- is the committed code, called unmodified.

`complementary_ensemble.py` is deliberately NOT edited: a ladder job is queued, and the provenance guard
aborts a queued job if a tracked file changes after submission.

SELF-TEST
---------
`_load_master_at` mirrors `complementary_ensemble.load_master`, so it can drift from it. `--self-test`
loads the UNCORRECTED master through both paths and asserts the two dictionaries are identical, which
fails loudly if the upstream reader ever changes shape.

    python scripts/checks/clean_ensemble.py --model meta-llama/Meta-Llama-3.1-8B \
        --perex-dir results/perex_clean__meta-llama_Meta-Llama-3.1-8B \
        --master results/cleanv2/pdl_cleanv2_master__meta-llama_Meta-Llama-3.1-8B.csv \
        --out results/hybrids/clean_ensemble__meta-llama_Meta-Llama-3.1-8B.csv
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import complementary_ensemble as ce  # noqa: E402


def _load_master_at(path):
    """Mirror of ce.load_master with the path supplied rather than derived. Kept in step by --self-test."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"GATE: master not found at {path}. Refusing to run without the reference "
                         "the continuity gate exists to check against.")
    with open(path) as f:
        rdr = _csv.DictReader(f)
        cols = rdr.fieldnames or []
        prr_col = "prr" if "prr" in cols else ("prr_mean" if "prr_mean" in cols else None)
        std_col = "prr_std" if "prr_std" in cols else None
        if prr_col is None:
            raise SystemExit(f"GATE: {path.name} has neither a 'prr' nor a 'prr_mean' column ({cols}).")
        rows = [r for r in rdr if r.get("rung") != "rung"]
    methods = {r["method"] for r in rows}
    display_keyed = bool(methods & set(ce.ALIAS_TO_MASTER.values()))
    out = {}
    for r in rows:
        try:
            v = float(r[prr_col])
        except (ValueError, KeyError, TypeError):
            continue
        sd = None
        if std_col:
            try:
                sd = float(r[std_col])
            except (ValueError, TypeError, KeyError):
                sd = None
        for raw, disp in ce.ALIAS_TO_MASTER.items():
            if r["method"] == (disp if display_keyed else raw):
                out[(raw, r["eval"], r["rung"])] = (v, sd)
    print(f"  master {path.name}: {prr_col!r} column, "
          f"{'display' if display_keyed else 'raw'}-keyed methods, {len(out)} component cells"
          + (f", {std_col!r} available" if std_col else ", no per-cell sd (strict bar only)"))
    return out


def self_test(slug):
    raw = ROOT / "results" / f"pdl_master__{slug}.csv"
    if not raw.exists():
        sys.exit(f"self-test needs the uncorrected master at {raw}")
    mine = _load_master_at(raw)
    theirs = ce.load_master(slug)
    if mine != theirs:
        only_a = sorted(set(mine) - set(theirs))[:5]
        only_b = sorted(set(theirs) - set(mine))[:5]
        diff = [k for k in set(mine) & set(theirs) if mine[k] != theirs[k]][:5]
        sys.exit(f"SELF-TEST FAILED -- the local reader has drifted from complementary_ensemble."
                 f"load_master. only-mine {only_a} only-theirs {only_b} differing {diff}")
    print(f"SELF-TEST PASS: local reader identical to complementary_ensemble.load_master "
          f"({len(mine)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--perex-dir", required=True)
    ap.add_argument("--master", required=True, help="the master describing THIS population")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    from luq import cache
    slug = cache._slug(args.model)
    if args.self_test:
        self_test(slug)
        return

    master_path = ROOT / args.master if not Path(args.master).is_absolute() else Path(args.master)
    print(f"CORRECTED-SPAN RUN: gate reference redirected to {master_path.relative_to(ROOT)}\n")
    ce.load_master = lambda _slug, _p=master_path: _load_master_at(_p)

    argv = ["complementary_ensemble.py", "--model", args.model,
            "--perex-dir", args.perex_dir, "--out", args.out]
    if args.exclude:
        argv += ["--exclude", args.exclude]
    sys.argv = argv
    ce.main()


if __name__ == "__main__":
    main()
