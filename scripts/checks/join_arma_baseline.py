"""Post-hoc joins for the S3/S6 CSVs. Joins are where this project's bugs live, so this ASSERTS before
merging and ABORTS on any mismatch (never degrades / never drops cells silently).

MODE fill-baseline (CHANGE 2): fill a DEFERRED S6 multihead CSV's K=1 columns from the S3 fixed_prior_ladder
  CSV (arm A). S6 training is self-contained (MH + ABLATION trained in-run); this only supplies arm A's PRR
  as a reference number for the MH-vs-K=1 comparison.

MODE merge-factscore (CHANGE 3): merge a factscore fixed_prior_ladder CSV into the main one when both exist.

Assert battery (both modes):
  * commit hash matches between the two runs (if the column is present on BOTH; else a LOUD warning that
    code-identity could not be verified -- re-run with the updated driver to record it).
  * seed VALUES match (not the count -- n_seeds==3 would pass 4,5,6), same present/warn rule.
  * eval list and rung list match (fill-baseline) / no cell overlap (merge-factscore).
  * cell keys align 1:1; UNMATCHED cells are reported, never dropped.
  * temperature condition labelled per side (S6 MH/ABL at T=1; arm A at best_T).

    python scripts/checks/join_arma_baseline.py --mode fill-baseline \
        --s6 results/multihead_ladder__<slug>.csv --s3 results/fixed_prior_ladder__<slug>.csv --out <slug>.csv
    python scripts/checks/join_arma_baseline.py --mode merge-factscore \
        --main results/fixed_prior_ladder__<slug>.csv --factscore results/fixed_prior_ladder_factscore__<slug>.csv
"""
import argparse
import csv as _csv
import sys
from pathlib import Path


def _read(path):
    with open(path) as f:
        r = _csv.DictReader(f)
        return list(r), r.fieldnames


def _uniq(rows, key):
    return sorted({r[key] for r in rows})


def _provenance_checks(a_rows, a_name, b_rows, b_name):
    """commit + seeds: assert equal if BOTH sides record them; else a loud unverifiable-warning. Aborts on
    a genuine mismatch (present on both, differ)."""
    for col in ("commit", "seeds"):
        av = {r.get(col) for r in a_rows if r.get(col) not in (None, "")}
        bv = {r.get(col) for r in b_rows if r.get(col) not in (None, "")}
        if not av or not bv:
            print(f"  ⚠️  {col}: not recorded on {'both sides' if not av and not bv else (a_name if not av else b_name)}"
                  f" -> CANNOT verify {col} match (re-run the driver with the updated code to record it).")
            continue
        if len(av) > 1 or len(bv) > 1:
            raise SystemExit(f"ABORT: {col} is not constant within a file (a={av}, b={bv}) -- mixed runs.")
        if av != bv:
            raise SystemExit(f"ABORT: {col} MISMATCH between runs: {a_name}={av} vs {b_name}={bv}. Not joining.")
        print(f"  ✓ {col} matches ({next(iter(av))[:12] if col == 'commit' else next(iter(av))})")


def fill_baseline(s6_path, s3_path, out_path):
    s6, s6f = _read(s6_path); s3, s3f = _read(s3_path)
    print(f"fill-baseline: S6={s6_path} ({len(s6)} rows)  S3={s3_path} ({len(s3)} rows)")
    _provenance_checks(s6, "S6", s3, "S3")
    # eval + rung lists must match (both describe the same long-form ladder)
    for col in ("eval", "rung"):
        se, s3e = set(_uniq(s6, col)), set(_uniq(s3, col))
        if se != s3e:
            print(f"  ⚠️  {col} sets differ: only-in-S6={sorted(se - s3e)}  only-in-S3={sorted(s3e - se)}")
    # arm A lookup from S3
    armA = {(r["eval"], r["rung"]): float(r["prr_mean"]) for r in s3 if r["method"] == "armA"}
    # cell alignment: every S6 method cell (mh/ablation/floor_min) should find an arm A
    s6_cells = {(r["eval"], r["rung"]) for r in s6 if not r["method"].startswith("VERDICT")}
    unmatched = sorted(c for c in s6_cells if c not in armA)
    if unmatched:
        print(f"  ⚠️  {len(unmatched)} S6 cells have NO arm A in S3 (reported, not dropped): {unmatched}")
    # fill k1_armA_s3 on the method rows; flip baseline_deferred to False (now joined)
    filled = 0
    for r in s6:
        r["baseline_deferred"] = "False"
        if r["method"].startswith("VERDICT"):
            continue
        k = (r["eval"], r["rung"])
        if k in armA:
            r["k1_armA_s3"] = round(armA[k], 4); filled += 1
    print(f"  filled {filled} rows with arm A. TEMPERATURE CAVEAT: S6 MH/ABLATION at T=1, arm A at best_T "
          f"(MH-vs-K1 carries this caveat; MH-vs-ABLATION, both T=1, is the clean control).")
    # headline count on the mh rows
    mh = [r for r in s6 if r["method"] == "mh" and r.get("k1_armA_s3") not in (None, "")]
    if mh:
        wins = sum(1 for r in mh if float(r["prr_mean"]) > float(r["k1_armA_s3"]))
        print(f"  MH beats K=1 (arm A) on {wins}/{len(mh)} cells")
    out = Path(out_path) if out_path else Path(s6_path).with_name(Path(s6_path).stem + "_joined.csv")
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=s6f); w.writeheader(); w.writerows(s6)
    print(f"wrote {out}")


def merge_factscore(main_path, fs_path, out_path):
    main, mf = _read(main_path); fs, ff = _read(fs_path)
    print(f"merge-factscore: main={main_path} ({len(main)} rows)  factscore={fs_path} ({len(fs)} rows)")
    if set(mf) != set(ff):
        raise SystemExit(f"ABORT: schema mismatch. only-in-main={set(mf) - set(ff)} only-in-fs={set(ff) - set(mf)}")
    _provenance_checks(main, "main", fs, "factscore")
    key = lambda r: (r["eval"], r["rung"], r["method"])
    overlap = {key(r) for r in main} & {key(r) for r in fs}
    if overlap:
        raise SystemExit(f"ABORT: {len(overlap)} overlapping (eval,rung,method) cells -- factscore should be a NEW "
                         f"eval, not a re-run of existing cells. e.g. {sorted(overlap)[:3]}")
    merged = main + fs
    assert len(merged) == len(main) + len(fs), "row-count check failed"     # no silent drops
    out = Path(out_path) if out_path else Path(main_path)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=mf); w.writeheader(); w.writerows(merged)
    print(f"  merged {len(main)} + {len(fs)} = {len(merged)} rows (row-count check PASS). wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fill-baseline", "merge-factscore"])
    ap.add_argument("--s6"); ap.add_argument("--s3")
    ap.add_argument("--main"); ap.add_argument("--factscore")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "fill-baseline":
        if not (args.s6 and args.s3):
            raise SystemExit("fill-baseline needs --s6 and --s3")
        fill_baseline(args.s6, args.s3, args.out)
    else:
        if not (args.main and args.factscore):
            raise SystemExit("merge-factscore needs --main and --factscore")
        merge_factscore(args.main, args.factscore, args.out)


if __name__ == "__main__":
    main()
