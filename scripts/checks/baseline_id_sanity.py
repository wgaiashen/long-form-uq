"""ID-cell SANITY CHECK for the P(True) / lookback feature caches -- NOT a results table.

WHAT THIS IS FOR. `verify_baseline_features.py` proves a feature file has the right shape, dtype and
row count. It cannot prove that feature row `i` describes the same example as label row `i`. A
silently misaligned cache passes every structural check and then produces a plausible-looking number
downstream, which is exactly the failure shape this project keeps hitting.

The test for alignment is to SCORE the features and see whether the numbers land where they should.
So this script does the same thing twice:

  1. CALIBRATE on the four datasets whose ID numbers are already known. `ood_onegrid.py` pins them in
     its ID_ANCHOR dict (sciq ptrue 0.727 / lookback 0.809, trivia 0.420 / 0.631, pubmed 0.532 /
     0.600). If this harness cannot reproduce those from the caches on disk, the harness is wrong and
     nothing it says about the new files means anything.
  2. REPORT the three new datasets through the identical path.

THE SECOND HALF IS A SANITY CHECK, NOT A RESULT. It is a single ID cell, seed-averaged, on one
population -- it is not the full ProbeDriftLong grid and must never be copied into a results table.
Its only job is to answer "do these features carry signal, and are they aligned with their labels?"
A PRR near zero where the sibling SAPLMA cache scores well is the alignment alarm.

    python scripts/checks/baseline_id_sanity.py

Reads cached features + records only. CPU, no model, no GPU, no judge calls.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import PROMPT_REGIME  # noqa: E402
from xl_rungs import eval_split  # noqa: E402  (the ladder's own train/test convention)

MODEL = "meta-llama/Meta-Llama-3.1-8B"
# SEED PROTOCOL (checked, not assumed). The published anchors come from 04_eval, and `03_probe.py:154`
# stamps `"seed": 1` -- they are SINGLE-SEED numbers. So the calibration compares seed 1 against the
# anchor, which is like-for-like; the extra seeds are reported beside it as spread, not averaged into
# the comparison. Averaging 5 seeds against a 1-seed anchor was the first version of this script and it
# manufactured a 0.036 "mismatch" on pubmed out of nothing but protocol.
ANCHOR_SEED = 1
SEEDS = (1, 2, 3, 4, 5)

# method -> (cache name, layer to read, standardize) -- copied from ood_onegrid.BASE so this harness
# trains the probe exactly the way the ladder does.
BASE = {"linear": ("saplma", 15, True),
        "ptrue_accurate": ("ptrue_accurate", 15, True),
        "lookback": ("lookback", 0, False)}

# The published ID cells this harness must reproduce (ood_onegrid.ID_ANCHOR, judge label).
ANCHOR = {"sciq": {"linear": 0.610, "ptrue_accurate": 0.727, "lookback": 0.809},
          "trivia_qa": {"linear": 0.732, "ptrue_accurate": 0.420, "lookback": 0.631},
          "pubmed_qa": {"linear": 0.550, "ptrue_accurate": 0.532, "lookback": 0.600}}
ANCHOR_TOL = 0.03                      # ood_onegrid's own GATE_TOL

CALIBRATE = ["sciq", "trivia_qa", "pubmed_qa", "xsum"]
NEW = [("asqa", "correctness"), ("expertqa", "factuality"), ("factscore", "factuality")]


def load_dataset(dataset: str, label_field: str):
    """Return (cfg, key, y, split, orig, n_dropped) for the LABELLED rows only.

    UNLABELLED-ROW HANDLING, copied from probedriftlong.py:268-275 so this harness sits on the same
    population as the ladder. Some label fields are genuinely absent on some rows (expertqa
    `factuality` on 292/2016, factscore on 45/500) and are correctly stored as null, never as 0. The
    ladder drops those rows FIRST and only then carves the train/test split, keeping an `orig` map back
    to the original record positions.

    `orig` is the part that matters downstream: the feature caches are indexed by ORIGINAL record
    position, so anything joining features to these filtered rows must index with `orig[i]`, not `i`.
    Getting that wrong shifts every expertqa row by up to 292 places and still produces a number.
    """
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    key = cache.run_key(MODEL, dataset, "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    y = np.array([r.get(label_field, np.nan) if r.get(label_field) is not None else np.nan
                  for r in recs], dtype=float)
    split = np.array([r["split"] for r in recs])
    orig = np.arange(len(recs))
    finite = np.isfinite(y)
    n_dropped = int((~finite).sum())
    if n_dropped:
        orig = np.where(finite)[0]
        y, split = y[orig], split[orig]
    return cfg, key, y, split, orig, n_dropped


def score(cfg, key, method, y, tr, te, orig):
    """Train the ladder's probe on the train rows and score PRR on the test rows.

    Returns (seed-1 PRR, mean over SEEDS, std over SEEDS), or (None, None, None) if the feature file
    is absent -- a missing cell must read as "not measured", never as a zero.
    """
    cname, layer, std = BASE[method]
    fpath = Path(cfg.cache_dir) / "features" / f"{key}__{cname}.npz"
    if not fpath.exists():
        return None, None, None
    arr = cache.load_features(cfg.cache_dir, key, cname)
    # index by ORIGINAL record position: `orig` maps filtered row -> raw cache row (see load_dataset)
    X = np.ascontiguousarray(arr[np.asarray(orig), layer, :])
    del arr
    vals = {}
    for sd in sorted(set(SEEDS) | {ANCHOR_SEED}):
        clf = probe.train_probe(X[tr], y[tr], standardize=std, seed=sd)
        vals[sd] = results.prr(list(y[te]), list(probe.uncertainty(clf, X[te])))
    del X
    multi = [vals[s] for s in SEEDS]
    return vals[ANCHOR_SEED], float(np.mean(multi)), float(np.std(multi))


def main():
    print("=" * 100)
    print("PART 1 -- CALIBRATION: reproduce the published ID anchors from the caches on disk")
    print("=" * 100)
    print(f"{'dataset':12s} {'method':16s} {'seed1':>8s} {'anchor':>8s} {'delta':>8s}  "
          f"{'5-seed mean±sd':>16s}  verdict")
    fails = []
    for ds in CALIBRATE:
        cfg, key, y, split, orig, n_drop = load_dataset(ds, "correctness")
        tr, te = eval_split(split)
        for m in BASE:
            s1, mean, sd = score(cfg, key, m, y, tr, te, orig)
            if s1 is None:
                print(f"{ds:12s} {m:16s} {'(no cache)':>8s} {'':>8s} {'':>8s} {'':>16s}  not measured")
                continue
            spread = f"{mean:>10.3f}±{sd:.3f}"
            a = ANCHOR.get(ds, {}).get(m)
            if a is None:
                print(f"{ds:12s} {m:16s} {s1:>8.3f} {'--':>8s} {'--':>8s} {spread:>16s}  no anchor published")
                continue
            d = s1 - a
            ok = abs(d) <= ANCHOR_TOL
            print(f"{ds:12s} {m:16s} {s1:>8.3f} {a:>8.3f} {d:>+8.3f} {spread:>16s}  "
                  f"{'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                fails.append(f"{ds}/{m}: seed-1 {s1:.3f} vs anchor {a:.3f} (delta {d:+.3f})")

    if fails:
        print("\nCALIBRATION FAILED -- the harness does not reproduce known ID cells, so nothing")
        print("it reports about the new caches is trustworthy:")
        for f in fails:
            print(f"  {f}")
        sys.exit(1)
    print("\nCALIBRATION OK: every published anchor reproduced within "
          f"{ANCHOR_TOL} from the caches on disk.\n")

    print("=" * 100)
    print("PART 2 -- SANITY CHECK on the three new caches  (single ID cell; NOT a results table)")
    print("=" * 100)
    print("ID cell for these three uses xl_rungs.eval_split's deterministic 70/30 carve, because")
    print("their records are test-only. `linear` (SAPLMA) is the reference column: it was extracted")
    print("separately, so if it scores well and the new columns collapse, the new caches are")
    print("misaligned with their labels rather than merely uninformative.\n")
    print(f"{'dataset':12s} {'label':14s} {'n(tr/te)':>12s} {'method':16s} {'seed1':>8s} "
          f"{'5-seed mean±sd':>16s}")
    for ds, lab in NEW:
        cfg, key, y, split, orig, n_drop = load_dataset(ds, lab)
        tr, te = eval_split(split)
        if n_drop:
            print(f"{ds:12s} {lab:14s} -- dropped {n_drop} unlabelled rows before the carve "
                  f"(ladder convention); {len(y)} labelled rows remain")
        for m in BASE:
            s1, mean, sd = score(cfg, key, m, y, tr, te, orig)
            a = f"{s1:>8.3f}" if s1 is not None else f"{'(no cache)':>8s}"
            b = f"{mean:>10.3f}±{sd:.3f}" if s1 is not None else ""
            print(f"{ds:12s} {lab:14s} {f'{len(tr)}/{len(te)}':>12s} {m:16s} {a} {b:>16s}")
    print("\nSANITY CHECK COMPLETE. These numbers are diagnostic only -- the reportable figures come")
    print("from the full ProbeDriftLong grid, not from this script.")


if __name__ == "__main__":
    main()
