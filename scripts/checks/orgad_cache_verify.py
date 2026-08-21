"""Gate the Orgad mask cache BEFORE any Orgad run. Presence is not enough.

WHY THIS EXISTS. `build_soft_orgad` only checks that `{SLUG}__{d}__ID__broad.json` EXISTS. A cache that
exists, parses, and is near-empty passes that check and then degrades silently: every record reads as
"nothing located", `build_masks` returns all-ones, and `build_soft_orgad` writes a LOUD-but-tolerated
uniform fallback per row. The "Orgad" arm then IS the unmasked arm, emits a full set of rows, and reads
as "Orgad ~= unmasked, no harm done". That exact contamination is on the record at
`weighted_msp_orgad_ladder.py:59-64`: of {pubmed_qa, sciq, trivia_qa, xsum} only pubmed_qa had a
__broad.json, so 3 of 4 evals were silently unmasked and dragged the margin to ~0.

It is also the precise signature of the bug fixed in 83108cf: asqa and factscore were missing from
`LONGFORM_QA_DATASETS`, so `--variant broad` would have used the SHORT-ANSWER prompt and written a
near-empty cache. Hence the checks below are aimed at that failure, not at file existence.

THREE CHECKS PER DATASET
  1. the broad JSON exists, parses, and is non-trivial
  2. VALUE TYPE -- the span variants cache a `list`. A `str` is the short-answer-prompt signature.
     (`locate_important_rows` dispatches on `isinstance(cached, list)`, so this is the read-side tell.)
  3. LOCATED FRACTION -- share of records where spans were actually found in the generation, from
     `build_masks(...)`'s own `located` return, plus the mean mask density.

PRE-REGISTERED THRESHOLD: located fraction < 0.50 => the dataset is treated as NOT COVERED. It is
reported and excluded, and any cell needing it is skipped. An effectively-uniform "Orgad" arm must never
be run under the Orgad label.

Exits non-zero if any present dataset fails, so it can gate a job script.

    python scripts/checks/orgad_cache_verify.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from transformers import AutoTokenizer  # noqa: E402

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import PROMPT_REGIME  # noqa: E402
import probedriftlong as PDL  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
MIN_LOCATED = 0.50          # pre-registered; below this the dataset is NOT COVERED
MIN_BYTES = 10_000


def broad_path(d):
    return ROOT / "cache" / "orgad_llm" / f"{SLUG}__{d}__ID__broad.json"


def check_one(d, tok, limit):
    """Return a dict describing this dataset's cache, or None if the file is absent."""
    p = broad_path(d)
    if not p.exists():
        return None
    row = {"dataset": d, "bytes": p.stat().st_size, "problems": []}
    if row["bytes"] < MIN_BYTES:
        row["problems"].append(f"file only {row['bytes']}B")

    try:
        ex = json.loads(p.read_text())
    except Exception as e:
        row["problems"].append(f"does not parse ({type(e).__name__})")
        return row
    row["entries"] = len(ex)

    # CHECK 2 -- value type. A str means the short-answer prompt wrote this cache (the 83108cf bug).
    vals = list(ex.values())
    n_list = sum(1 for v in vals if isinstance(v, list))
    n_str = sum(1 for v in vals if isinstance(v, str))
    n_noans = sum(1 for v in vals if v == "NO ANSWER" or (isinstance(v, list) and not v))
    row["pct_list"] = n_list / max(1, len(vals))
    row["pct_noanswer"] = n_noans / max(1, len(vals))
    if n_list == 0 and n_str > 0:
        row["problems"].append("values are STRINGS not span lists -> short-answer prompt (83108cf)")

    # CHECK 3 -- located fraction, from build_masks' own return, on the real records
    try:
        from weighted_msp_orgad_ladder import build_masks
        cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                     prompt_regime=PROMPT_REGIME.get(d, ""))
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
        # SUBSAMPLE RANDOMLY, NEVER recs[:limit]. Records are not in random order, and a HEAD slice
        # gives a biased located-fraction: on med_quad the first 200 records read 0.995 while the full
        # 1800 read 0.747 (455 empty span lists, all sitting past the head). A gate that under-reports
        # the very degradation it exists to catch is worse than no gate -- a dataset genuinely below the
        # threshold would have passed. Default is ALL records; --limit is for a quick look only.
        if limit and limit < len(recs):
            sel = np.random.RandomState(0).choice(len(recs), limit, replace=False)
            recs = [recs[i] for i in sorted(sel)]
        for r in recs:
            r["_dataset"] = d
        masks, located = build_masks(tok, recs, variant="broad", floor=0.0)
        row["n_records"] = len(recs)
        row["located_frac"] = float(np.mean(located))
        # mean share of generated tokens marked important, over rows where something WAS located --
        # a located row whose mask is ~everything is not a mask, it is uniform under another name
        dens = [float(np.mean(np.asarray(m) > 0)) for m, lc in zip(masks, located) if lc]
        row["mask_density"] = float(np.mean(dens)) if dens else float("nan")
    except Exception as e:
        row["problems"].append(f"build_masks failed ({type(e).__name__}: {e})")
        return row

    if row["located_frac"] < MIN_LOCATED:
        row["problems"].append(f"located {row['located_frac']:.2f} < {MIN_LOCATED} -> NOT COVERED")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(PDL.LONG_SRC))
    ap.add_argument("--limit", type=int, default=0,
                    help="RANDOM subsample per dataset for a quick look. Default 0 = ALL records, which "
                         "is what the gate verdict must be based on.")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"Orgad broad-mask cache: {ROOT / 'cache' / 'orgad_llm'}")
    print(f"pre-registered threshold: located fraction >= {MIN_LOCATED}")
    if args.limit:
        print(f"ESTIMATE ONLY: random subsample of {args.limit} records/dataset (seed 0). "
              f"Re-run with --limit 0 before trusting a pass/fail verdict.")
    print()
    print(f"{'dataset':<15} {'bytes':>10} {'entries':>8} {'%list':>7} {'%noans':>7} "
          f"{'located':>8} {'density':>8}  status")
    print("-" * 96)

    covered, absent, failed = [], [], []
    for d in datasets:
        row = check_one(d, tok, args.limit)
        if row is None:
            absent.append(d)
            print(f"{d:<15} {'--':>10} {'--':>8} {'--':>7} {'--':>7} {'--':>8} {'--':>8}  ABSENT")
            continue
        ok = not row["problems"]
        (covered if ok else failed).append(d)
        print(f"{d:<15} {row['bytes']:>10} {row.get('entries', 0):>8} "
              f"{row.get('pct_list', float('nan')):>7.2f} {row.get('pct_noanswer', float('nan')):>7.2f} "
              f"{row.get('located_frac', float('nan')):>8.3f} {row.get('mask_density', float('nan')):>8.3f}  "
              f"{'OK' if ok else '*** ' + '; '.join(row['problems']) + ' ***'}")

    print("-" * 96)
    print(f"covered: {len(covered)}  absent: {len(absent)}  failed: {len(failed)}")
    if absent:
        print(f"  absent : {', '.join(absent)}")
    if failed:
        print(f"  failed : {', '.join(failed)}")

    # Which cells this unlocks -- computed from build_soft_orgad's ACTUAL requirement, which is every
    # dataset in train_rows + test_rows, i.e. the pool sources UNION the eval target (not just the pool).
    usable = set(covered)
    print("\norgad-runnable cells under this cache (pool sources UNION eval must all be covered):")
    evals = ["pubmed_qa", "cnn_dailymail", "xsum", "factscore"]
    runnable = 0
    for rung, X, spec in PDL.cells_long(set(PDL.LONG_SRC), evals):
        req = {X} | {s for s, _ in spec}
        if req <= usable:
            runnable += 1
        else:
            print(f"    skip {rung:16s} {X:<15} missing {sorted(req - usable)}")
    print(f"  runnable: {runnable}/20")

    if failed:
        print("\nDO NOT RUN ORGAD -- a listed dataset would supply an effectively-uniform mask.")
        raise SystemExit(1)
    print("\nCACHE OK for the covered datasets.")


if __name__ == "__main__":
    main()
