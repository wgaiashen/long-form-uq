"""Format gate for the P(True) / lookback baseline feature caches.

WHY THIS EXISTS. `01b_ptrue.py` and `01c_lookback.py` have no skip-if-exists and no resume: every
run recomputes from scratch and overwrites whatever is at the output path. So two things can go
wrong quietly, and this script is the gate for both.

  (1) A NEW cache comes out in a DIFFERENT format from the caches already on disk -- wrong number
      of stored layers, wrong row count, a silent truncation. It would still load, still look like
      a feature matrix, and only fall over much later (or worse, not fall over at all).
  (2) A wrong `--dataset` argument OVERWRITES one of the reference caches. You would then be
      "verifying" a new file against a file you had just written, which checks nothing at all.

So the script has two jobs. `--guard-write` / `--guard-check` snapshot and re-check the reference
caches (sha256 + size + mtime) around a run; `--datasets` verifies feature files against sciq's
existing cache, which is treated as the definition of correct format rather than something assumed.

Nothing here needs a GPU or a model -- it reads the .npz files and the records .jsonl.

    # before any extraction: freeze the reference caches, and calibrate the gate on known-good files
    python scripts/checks/verify_baseline_features.py --guard-write ref_snapshot.json
    python scripts/checks/verify_baseline_features.py --datasets sciq trivia_qa pubmed_qa xsum

    # after an extraction: verify the new files, and prove the references were not touched
    python scripts/checks/verify_baseline_features.py --datasets asqa
    python scripts/checks/verify_baseline_features.py --guard-check ref_snapshot.json

Exits 1 on any failure, naming the file. A MISSING file is reported as missing, never as an empty
or zero result -- "not measured" and "measured and empty" must stay distinguishable.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import PROMPT_REGIME  # noqa: E402  (the canonical dataset -> cache-namespace map)

MODEL = "meta-llama/Meta-Llama-3.1-8B"

# The two baseline feature sets this script covers, and the cache method name each is stored under.
# NB the P(True) features live under `ptrue_accurate`, not `ptrue`: that name marks the unified
# "Is the above response accurate?" wording, and it is the name the ladder drivers read
# (scripts/checks/ood_onegrid.py). There is no `__ptrue.npz` on disk for any dataset.
METHODS = ("ptrue_accurate", "lookback")

# sciq is the format reference: its two caches were written 2026-07-01 and are what every existing
# result was computed from. We read the expected shape/dtype/layer-pattern OFF these files rather
# than hardcoding them, so the gate cannot drift away from the thing it is supposed to protect.
REFERENCE_DATASET = "sciq"
REFERENCE_DATASETS = ("sciq", "trivia_qa", "pubmed_qa", "xsum")


# --------------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------------
def paths_for(dataset: str, method: str):
    """Return (feature_path, records_path, saplma_path) for one dataset/method.

    `prompt_regime` is a cache NAMESPACE, not a filename suffix: a non-empty regime moves the whole
    cache into `cache/<regime>/` while the filename stays `<model>__<dataset>__ID.<ext>`. That is
    why asqa/expertqa/factscore files are not in the top-level cache dir.
    """
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    feats = Path(cfg.cache_dir) / "features" / f"{key}__{method}.npz"
    records = Path(cfg.cache_dir) / "records" / f"{key}.jsonl"
    saplma = Path(cfg.cache_dir) / "features" / f"{key}__saplma.npz"
    return feats, records, saplma


def load_feats(path: Path):
    """Load one feature file, checking the container itself before the contents.

    A .npz is a zip archive, so a truncated file raises here (BadZipFile) rather than producing a
    plausible-looking short array. We let that exception propagate -- a load failure IS the finding.
    """
    with np.load(path) as z:
        keys = list(z.keys())
        if keys != ["feats"]:
            raise ValueError(f"expected exactly one array named 'feats', found {keys}")
        return z["feats"]


def n_records(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def short_gen_rows(path: Path) -> int:
    """Count records whose generation is shorter than 2 tokens.

    `lookback.lookback_vector` returns an all-zero vector for exactly these rows (it drops the last
    generated token, so it needs at least 2). That gives us an EXACT integer we can predict from the
    records alone and compare against the extracted file -- a check that does not go through the
    extraction code path, so it catches misalignment and truncation rather than just restating them.
    """
    import json as _json
    n = 0
    with path.open() as f:
        for line in f:
            if len(_json.loads(line)["gen_token_ids"]) < 2:
                n += 1
    return n


# --------------------------------------------------------------------------------------------
# the reference guard: sha256 + size + mtime of the caches we compare against
# --------------------------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_state() -> dict:
    state = {}
    for ds in REFERENCE_DATASETS:
        for m in METHODS:
            p, _, _ = paths_for(ds, m)
            if not p.exists():
                continue
            st = p.stat()
            state[str(p.relative_to(ROOT))] = {
                "sha256": sha256_of(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return state


def guard_write(out: Path) -> int:
    state = reference_state()
    out.write_text(json.dumps(state, indent=2, sort_keys=True))
    print(f"GUARD SNAPSHOT: {len(state)} reference caches -> {out}")
    for k, v in sorted(state.items()):
        print(f"  {v['sha256'][:16]}  {v['size']:>12,}  {k}")
    return 0


def guard_check(snap: Path) -> int:
    """Re-check the reference caches against the snapshot. Any change is a FAILURE, not a note."""
    before = json.loads(snap.read_text())
    after = reference_state()
    bad = []
    for rel, exp in sorted(before.items()):
        got = after.get(rel)
        if got is None:
            bad.append(f"{rel}: DISAPPEARED since the snapshot")
        elif got["sha256"] != exp["sha256"]:
            bad.append(f"{rel}: sha256 CHANGED ({exp['sha256'][:16]} -> {got['sha256'][:16]}, "
                       f"size {exp['size']:,} -> {got['size']:,}) -- a reference cache was OVERWRITTEN")
        elif got["mtime_ns"] != exp["mtime_ns"]:
            bad.append(f"{rel}: content identical but mtime changed -- rewritten with the same bytes")
        else:
            print(f"  unchanged  {exp['sha256'][:16]}  {rel}")
    new = sorted(set(after) - set(before))
    for rel in new:
        print(f"  NEW (not in snapshot, not a reference file at snapshot time): {rel}")
    if bad:
        print("\nGUARD FAIL:")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"\nGUARD OK: all {len(before)} reference caches byte-identical to the snapshot")
    return 0


# --------------------------------------------------------------------------------------------
# the format gate
# --------------------------------------------------------------------------------------------
def reference_spec():
    """Read the expected format off sciq's own caches. Returns {method: spec dict}."""
    spec = {}
    for m in METHODS:
        p, _, _ = paths_for(REFERENCE_DATASET, m)
        if not p.exists():
            sys.exit(f"reference cache missing, cannot calibrate the gate: {p}")
        f = load_feats(p)
        s = {"tail_shape": f.shape[1:], "dtype": f.dtype}
        if m == "ptrue_accurate":
            # Which layers are actually stored. The extraction NaN-fills every layer it was not
            # asked to keep (`--store-layers 15`), so the array stays full height (33) and a probe
            # of an unstored layer fails loudly on the NaN instead of silently returning junk.
            s["finite_layers"] = tuple(np.isfinite(f[:, L, :]).any() for L in range(f.shape[1]))
        spec[m] = s
        del f
    return spec


def verify_one(dataset: str, method: str, spec: dict) -> list[str]:
    """Return a list of failure strings ([] means the file passed)."""
    fpath, rpath, spath = paths_for(dataset, method)
    tag = f"{dataset:10s} {method:15s}"

    if not fpath.exists():
        print(f"{tag} MISSING   {fpath}")
        return [f"{dataset}/{method}: feature file MISSING at {fpath}"]

    fails = []
    size = fpath.stat().st_size
    try:
        feats = load_feats(fpath)
    except Exception as e:                                   # noqa: BLE001 -- the failure IS the result
        print(f"{tag} LOAD FAIL {type(e).__name__}: {e}")
        return [f"{dataset}/{method}: failed to LOAD ({type(e).__name__}: {e})"]

    ref = spec[method]

    # 1. dtype and the non-row dimensions must match the reference exactly.
    if feats.dtype != ref["dtype"]:
        fails.append(f"{dataset}/{method}: dtype {feats.dtype} != reference {ref['dtype']}")
    if feats.shape[1:] != ref["tail_shape"]:
        fails.append(f"{dataset}/{method}: shape tail {feats.shape[1:]} != reference {ref['tail_shape']}")

    # 2. Row count must match the records this was extracted from. 01b/01c iterate the records in
    #    order with no filtering, so a mismatch means the file is not aligned with the records and
    #    every downstream row-indexed join would be silently wrong.
    n_rec = n_records(rpath) if rpath.exists() else None
    if n_rec is None:
        fails.append(f"{dataset}/{method}: records file MISSING at {rpath}")
    elif feats.shape[0] != n_rec:
        fails.append(f"{dataset}/{method}: {feats.shape[0]} rows != {n_rec} records ({rpath.name})")

    # 3. Cross-check against the SAPLMA cache already sitting in the same directory: an independent
    #    file, extracted at a different time, that must agree on the row count.
    n_sap = None
    if spath.exists():
        with np.load(spath) as z:
            n_sap = z["feats"].shape[0]
        if n_sap != feats.shape[0]:
            fails.append(f"{dataset}/{method}: {feats.shape[0]} rows != saplma cache {n_sap} rows")

    extra = ""
    if method == "ptrue_accurate":
        # 4. The stored-layer pattern must be identical to the reference. This is what catches a run
        #    that forgot `--store-layers 15`: same shape, same dtype, 46x the size, all 33 layers
        #    populated -- a file that would load fine and quietly differ from every existing cache.
        finite = tuple(np.isfinite(feats[:, L, :]).any() for L in range(feats.shape[1]))
        if finite != ref["finite_layers"]:
            got = [i for i, b in enumerate(finite) if b]
            want = [i for i, b in enumerate(ref["finite_layers"]) if b]
            fails.append(f"{dataset}/{method}: stored layers {got} != reference {want}")
        # 5. The layer the probe actually reads must be completely clean.
        L15 = feats[:, 15, :]
        n_bad = int((~np.isfinite(L15)).sum())
        if n_bad:
            fails.append(f"{dataset}/{method}: layer 15 has {n_bad} non-finite values")
        extra = f"L15 |mean| {np.abs(L15).mean():.4f}  std {L15.std():.4f}"

    if method == "lookback":
        # 6. The lookback ratio is ctx/(ctx+new) with both terms positive, so it is bounded in [0,1]
        #    by construction. Anything outside means the attention pass went wrong (e.g. fp16 NaNs).
        if not np.isfinite(feats).all():
            fails.append(f"{dataset}/{method}: contains non-finite values")
        else:
            lo, hi = float(feats.min()), float(feats.max())
            if lo < 0.0 or hi > 1.0:
                fails.append(f"{dataset}/{method}: values outside [0,1] (min {lo:.4f}, max {hi:.4f})")
        # 7. The all-zero rows are the documented `n_out < 2` fallback. Predict how many there
        #    should be straight from the records and demand an exact match.
        zero_rows = int((np.abs(feats).sum(axis=(1, 2)) == 0).sum())
        if rpath.exists():
            expect = short_gen_rows(rpath)
            if zero_rows != expect:
                fails.append(f"{dataset}/{method}: {zero_rows} all-zero rows but {expect} records "
                             f"have <2 generated tokens (the n_out<2 fallback) -- these must match")
            extra = f"all-zero rows {zero_rows} (expected {expect})  range [{feats.min():.4f}, {feats.max():.4f}]"

    print(f"{tag} {'OK  ' if not fails else 'FAIL'}  {str(feats.shape):>20s}  {feats.dtype}  "
          f"{size/1e6:8.2f} MB  rows={feats.shape[0]} recs={n_rec} saplma={n_sap}  {extra}")
    del feats
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[],
                    help="datasets to verify, e.g. asqa expertqa factscore")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--guard-write", type=Path, default=None,
                    help="snapshot the reference caches (sha256/size/mtime) to this JSON and exit")
    ap.add_argument("--guard-check", type=Path, default=None,
                    help="re-check the reference caches against a snapshot and exit")
    args = ap.parse_args()

    if args.guard_write:
        sys.exit(guard_write(args.guard_write))
    if args.guard_check:
        sys.exit(guard_check(args.guard_check))
    if not args.datasets:
        sys.exit("nothing to do: pass --datasets, --guard-write or --guard-check")

    spec = reference_spec()
    print(f"gate calibrated on {REFERENCE_DATASET}: "
          + "; ".join(f"{m} shape(*,{','.join(map(str, spec[m]['tail_shape']))}) {spec[m]['dtype']}"
                      + (f" layers={[i for i, b in enumerate(spec[m]['finite_layers']) if b]}"
                         if 'finite_layers' in spec[m] else "")
                      for m in args.methods if m in spec))
    print()

    fails = []
    for ds in args.datasets:
        for m in args.methods:
            fails += verify_one(ds, m, spec)

    print()
    if fails:
        print(f"VERIFY FAIL ({len(fails)}):")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    print(f"VERIFY OK: {len(args.datasets)} dataset(s) x {len(args.methods)} method(s) all pass")


if __name__ == "__main__":
    main()
