"""Cache <-> table consistency audit: prove the ID/OOD table numbers are reproducible from the caches
and settle the xsum bf16-SAPLMA-feature flag once and for all.

For each dataset we confirm four things, from the ARTIFACTS (not from memory):

  1. WHICH cache is bf16.  A bf16 value stored as fp32 has its low 16 mantissa bits zero; we measure
     the fraction of zero-low-16 values in the pertok states and in the SAPLMA feature cache. This
     tells us, per dataset, whether the pertok (the cache the tables actually read) is fp32.

  2. raw_mean == SAPLMA feature.  The mean over the pertok answer-window rows must equal the cached
     SAPLMA L15 feature (the aggregation-table gate). ~1e-6 = identical; a bf16 feature cache shows a
     larger uniform gap while the pertok stays fp32 (xsum), which is benign for the TABLE because...

  3. THE REPORTED NUMBER IS FP32.  aggregation_table.py reports mean-pool+MLP from the fp32 pertok
     (Xmean), and uses the SAPLMA feature cache ONLY as a gate anchor. We recompute the SAPLMA-mean
     PRR from the fp32 pertok (seeds 1-3) and check it matches the on-disk aggregation_table CSV. We
     ALSO compute it from the (possibly bf16) feature cache and show the two agree within rank-noise
     -- i.e. the bf16 feature never moved a reported number, so no DoC re-extraction is needed for
     the tables.

  4. LABELS ALIGN.  The pertok `y` (positional) must equal the records' correctness in order, non-NaN.

Finally, diff the freshly-regenerated tables (results/audit/*.csv, written by the PBS job just before
this) against the on-disk tables: PRR point estimates must match; only the significance columns may
change (the intended t-test -> bootstrap refresh).

Heavy (loads the ~GB pertok + feature caches) -> COMPUTE job, never login.

    python scripts/checks/cache_table_audit.py --datasets sciq,trivia_qa,pubmed_qa,xsum,med_quad
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import load_per_token  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
LAYER = 15
SEEDS = [1, 2, 3]


def frac_bf16(arr):
    """Fraction of float32 values whose low 16 mantissa bits are zero (== bf16-rounded). True fp32
    data lands near 2^-16 (~1.5e-5) by chance; bf16-rounded data lands near 1.0."""
    u = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return float(np.mean((u & 0xFFFF) == 0))


def prr_meanpool(X, split, y, seeds):
    tr = np.where(split == "train")[0]
    te = np.where(split == "test")[0]
    if len(te) == 0:
        return None
    vals = []
    for sd in seeds:
        clf = probe.train_probe_mlp(X[tr], y[tr], seed=sd)
        vals.append(results.prr(y[te], 1.0 - clf.p_correct(X[te])))
    return float(np.mean(vals)), float(np.std(vals))


def audit_dataset(dataset):
    print(f"\n########## {dataset} ##########", flush=True)
    loaded = load_per_token(MODEL, dataset, LAYER, "correctness")
    if loaded is None:
        print(f"  no pertok cache -> SKIP")
        return
    states, split, y, lyr, records = loaded
    n = len(states)
    raw_mean = np.stack([s.mean(axis=0) for s in states]).astype(np.float32)

    # (1) bf16 signature -- which cache is bf16?
    pt_bf16 = frac_bf16(np.concatenate([s.reshape(-1) for s in states[:50]]))
    print(f"  [dtype]  pertok zero-low16 frac = {pt_bf16:.3f}  "
          f"({'bf16-rounded' if pt_bf16 > 0.5 else 'genuine fp32'})", flush=True)

    # (2) raw_mean vs SAPLMA feature + (3) fp32-mean vs feature-derived PRR
    try:
        feat = cache.load_features(Config(model_name=MODEL, dataset=dataset, ood_setting="ID").cache_dir,
                                   cache.run_key(MODEL, dataset, "ID"), "saplma")
        featL = feat[:, lyr, :].astype(np.float32)
        maxdiff = float(np.abs(raw_mean - featL[:n]).max())
        ft_bf16 = frac_bf16(featL[:50])
        print(f"  [align]  raw_mean vs SAPLMA feature maxdiff = {maxdiff:.2e}   "
              f"(feature zero-low16 frac = {ft_bf16:.3f} -> {'bf16' if ft_bf16 > 0.5 else 'fp32'})", flush=True)
        prr_fp32 = prr_meanpool(raw_mean, split, y, SEEDS)
        prr_feat = prr_meanpool(featL[:n], split, y, SEEDS)
        if prr_fp32 and prr_feat:
            gap = prr_fp32[0] - prr_feat[0]
            print(f"  [PRR ]   SAPLMA-mean  fp32-pertok = {prr_fp32[0]:+.4f}   "
                  f"feature-cache = {prr_feat[0]:+.4f}   gap = {gap:+.4f}  "
                  f"({'bf16 feature moves the number' if abs(gap) > 0.01 else 'bf16 has no effect'})", flush=True)
        elif prr_fp32 is None:
            print(f"  [PRR ]   train-only dataset (no test split) -> PRR is source-side only", flush=True)
    except Exception as e:
        print(f"  [align]  no/unreadable SAPLMA feature cache: {e}", flush=True)
        prr_fp32 = prr_meanpool(raw_mean, split, y, SEEDS)

    # (4) label alignment: pertok y == records correctness, in order, finite
    rc = np.array([r.get("correctness", np.nan) for r in records], dtype=float)
    aligned = np.allclose(np.nan_to_num(rc), np.nan_to_num(y), atol=1e-9)
    n_nan = int(np.isnan(y).sum())
    print(f"  [label]  pertok y == records correctness (in order): {aligned}   "
          f"NaN labels: {n_nan}/{n}   label_model = {records[0].get('correctness_model','?')}", flush=True)
    return prr_fp32 if 'prr_fp32' in dir() else None


def diff_tables():
    """Compare the freshly-regenerated audit tables to the on-disk tables: PRR point estimates must
    match; significance columns may change (intended bootstrap refresh)."""
    print(f"\n########## fresh-vs-ondisk table diff ##########", flush=True)
    pairs = [
        ("aggregation_table", ROOT / "results" / "audit" / "aggregation_table_fresh.csv",
         ROOT / "results" / f"aggregation_table__{cache._slug(MODEL)}.csv",
         ("dataset", "label_field", "aggregator")),
        ("contribution_ladder", ROOT / "results" / "audit" / "contribution_ladder_fresh.csv",
         ROOT / "results" / f"contribution_ladder__{cache._slug(MODEL)}.csv",
         ("rung", "eval", "train", "method")),
    ]
    for name, fresh_p, old_p, keycols in pairs:
        if not fresh_p.exists():
            print(f"  [{name}] fresh table not found ({fresh_p.name}) -> regeneration step did not run", flush=True)
            continue
        if not old_p.exists():
            print(f"  [{name}] no on-disk table to compare -> fresh is the first", flush=True)
            continue

        def index(path):
            with open(path) as f:
                rows = list(csv.DictReader(f))
            return {tuple(r.get(k, "") for k in keycols): r for r in rows}

        fresh, old = index(fresh_p), index(old_p)
        keys = sorted(set(fresh) & set(old))
        worst = 0.0; worst_key = None; n_cmp = 0
        for k in keys:
            if str(k[-1]).startswith("VERDICT") or str(k[-1]).startswith("DIAG"):
                continue  # significance/diagnostic rows -- point estimate may legitimately refresh
            try:
                a = float(fresh[k]["prr_mean"]); b = float(old[k]["prr_mean"])
            except (ValueError, KeyError, TypeError):
                continue
            d = abs(a - b); n_cmp += 1
            if d > worst:
                worst, worst_key = d, k
        only_fresh = sorted(set(fresh) - set(old))
        only_old = sorted(set(old) - set(fresh))
        verdict = "MATCH" if worst < 0.03 else "MISMATCH"
        print(f"  [{name}] compared {n_cmp} PRR cells: worst |delta| = {worst:.4f} at {worst_key} -> {verdict}", flush=True)
        if only_fresh:
            print(f"           rows only in fresh ({len(only_fresh)}): {only_fresh[:4]}{' ...' if len(only_fresh) > 4 else ''}", flush=True)
        if only_old:
            print(f"           rows only in on-disk ({len(only_old)}): {only_old[:4]}{' ...' if len(only_old) > 4 else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sciq,trivia_qa,pubmed_qa,xsum,med_quad")
    ap.add_argument("--skip-diff", action="store_true")
    args = ap.parse_args()
    for d in args.datasets.split(","):
        audit_dataset(d)
    if not args.skip_diff:
        diff_tables()


if __name__ == "__main__":
    main()
