"""ProbeDrift-light OOD for SAPLMA, using the reference pool composition (its code, not its README).

We drive `probe_drift.ood_settings.get_training_spec(eval, setting)` — the SAME function the
`run_polygraph.py` uses (via `load_datasets_via_probe_drift(..., seed=1)`) — so the training pool is
Reference: LEAVE_ONE_OUT = the other 9 datasets at 200 each; DIFF_TASK (QA eval) = samsum+xsum+cnn at 600
each. We keep that per-source CAP and restrict to the sources we have Llama features for
({sciq, trivia_qa, pubmed_qa, xsum}); omitted sources are logged. This is a SUBSET of ProbeDrift-light,
NOT a reproduction — see the note below on why cell-by-cell reproduction is impossible from a subset.

SAPLMA = the A&M MLP on the mean-pooled middle-layer (L15) hidden state, judge-labelled — the same
`full_sequence_saplma` / `hs_middle` probe whose ID numbers reproduce Table 14 exactly.

WHY SEED-AVERAGED (not the single seed=1): with only 3-of-9 LOO sources the pool is small, and the
OOD PRR is dominated by WHICH 200-example subsample gets drawn (measured: subsample-only std ~0.043 vs
training-only std ~0.014 on pubmed-LOO). the seed=1 draw is stable-but-arbitrary; restricting the
sources changes the RNG draw path, so the reference subsample, and thus its exact cell value, cannot be
reproduced from a subset EVEN IN PRINCIPLE. So we report a seed-averaged mean±std range, not a single
cell claimed to match it.

ID-diagonal GATE (mirrors 05_transfer): load the cached ID probe, score its eval features, assert the
uncertainties equal the cached 04_eval scores (allclose). Fail loudly if not.

    python scripts/checks/ood_refpools.py --model meta-llama/Meta-Llama-3.1-8B          # seed-averaged table
    python scripts/checks/ood_refpools.py --diagnose pubmed_qa                          # variance + paired xsum test
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, probe, results          # noqa: E402
from luq.config import Config                   # noqa: E402
from luq.features import saplma                 # noqa: E402
from probe_drift.ood_settings import get_training_spec  # noqa: E402  (reference code)

LAYER = 15
LABEL = "correctness"
AVAIL = ("sciq", "trivia_qa", "pubmed_qa", "xsum")   # datasets we have Llama L15 features for
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
SETTINGS = ["OOD_LEAVE_ONE_OUT", "OOD_DIFF_TASK"]


def _feats(cache_dir, model):
    """Load (L15 features, records) once per available dataset."""
    fe = {}
    for d in AVAIL:
        key = cache.run_key(model, d, "ID")
        f = saplma.select_layer(cache.load_features(cache_dir, key, "saplma"), LAYER)
        r = cache.load_records(cache_dir, key)
        if len(r) != len(f):
            raise SystemExit(f"{d}: records/features out of step — rerun 01/03.")
        fe[d] = (f, r)
    return fe


def _rows(records, split):
    return np.array([i for i, x in enumerate(records) if x["split"] == split])


def _eval_test(fe, E):
    f, r = fe[E]
    te = _rows(r, "test")
    return f[te], np.array([r[i][LABEL] for i in te], dtype=float)


def _prr(fe, E, specs, sub_seed, train_seed, batch):
    """Subsample controlled by sub_seed (fresh RNG per source -> order-independent); train by train_seed."""
    Xs, ys = [], []
    for d, n in specs:
        f, r = fe[d]
        tr = _rows(r, "train")
        pick = tr[np.random.RandomState(sub_seed).permutation(len(tr))[:min(n, len(tr))]]
        Xs.append(f[pick])
        ys.append(np.array([r[i][LABEL] for i in pick], dtype=float))
    clf = probe.train_probe_mlp(np.vstack(Xs), np.concatenate(ys), batch_size=batch, seed=train_seed)
    Xte, yte = _eval_test(fe, E)
    return results.prr(list(yte), list(probe.uncertainty(clf, Xte)))


def _restrict(E, setting):
    """the pool (src, n), restricted to available sources (drop the eval dataset and any absent)."""
    spec = get_training_spec(E, setting)
    kept = [(s, n) for s, n in spec if s in AVAIL and s != E]
    omitted = [s for s, n in spec if (s not in AVAIL) and s != E]
    return kept, omitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--batch", type=int, default=1, help="SAPLMA MLP batch size (1 = keystone/reference)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--diagnose", default=None,
                    help="eval dataset for the variance-decomposition + paired-xsum diagnostic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache_dir = Config(model_name=args.model, dataset="sciq", ood_setting="ID").cache_dir
    fe = _feats(cache_dir, args.model)

    if args.diagnose:
        _diagnose(fe, cache_dir, args.model, args.diagnose, args.batch, args.seeds)
        return

    out_rows = []
    print(f"SAPLMA OOD, the pools restricted to {list(AVAIL)}, {args.seeds}-seed mean±std "
          f"(batch={args.batch}, L{LAYER}, judge):")
    for E in EVALS:
        # ID reference = cached keystone score (deterministic), gated against 04_eval.
        key = cache.run_key(args.model, E, "ID")
        Xte, yte = _eval_test(fe, E)
        clf_id = cache.load_probe(cache_dir, key, "saplma", LAYER)
        unc_id = probe.uncertainty(clf_id, Xte)
        cached = cache.load_scores(cache_dir, key, "saplma")["unc"]
        if not (len(cached) == len(unc_id) and np.allclose(cached, unc_id, atol=1e-6)):
            sys.exit(f"ID-DIAGONAL GATE FAILED for {E}: recomputed != cached 04_eval scores.")
        idp = results.prr(list(yte), list(unc_id))
        print(f"\n{E}:  ID(keystone) {idp:.3f}  [gate PASS]")
        out_rows.append({"eval": E, "setting": "ID", "pool": E,
                         "prr_mean": round(idp, 4), "prr_std": 0.0, "n_seeds": 1, "omitted": ""})
        for setting in SETTINGS:
            kept, omitted = _restrict(E, setting)
            if not kept:
                continue
            vals = [_prr(fe, E, kept, s, s, args.batch) for s in range(args.seeds)]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            pool = "+".join(f"{s}:{n}" for s, n in kept)
            print(f"  {setting:18s} {m:+.3f} ± {sd:.3f}  pool[{pool}]  omitted{omitted}")
            out_rows.append({"eval": E, "setting": setting, "pool": pool,
                             "prr_mean": round(m, 4), "prr_std": round(sd, 4),
                             "n_seeds": args.seeds, "omitted": ";".join(omitted)})

    out = Path(args.out) if args.out else (ROOT / "results" /
          f"ood_refpools__{cache._slug(args.model)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["eval", "setting", "pool", "prr_mean", "prr_std",
                                           "n_seeds", "omitted"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out}")
    print("NOTE: OOD cells are a SUBSET of the pools (fewer sources) and seed-AVERAGED — NOT a "
          "cell-by-cell reproduction of the reference numbers (impossible from a restricted pool; see docstring).")


def _diagnose(fe, cache_dir, model, E, batch, seeds):
    """Decompose OOD variance (subsample vs training seed) and run the paired xsum test, for eval E.
    Only meaningful for a QA eval whose LOO pool contains xsum among the available sources."""
    kept, _ = _restrict(E, "OOD_LEAVE_ONE_OUT")
    with_xsum = kept
    without_xsum = [(s, n) for s, n in kept if s != "xsum"]
    K = range(seeds)
    total = [_prr(fe, E, with_xsum, k, k, batch) for k in K]
    tronly = [_prr(fe, E, with_xsum, 0, k, batch) for k in K]     # subsample fixed
    subonly = [_prr(fe, E, with_xsum, k, 0, batch) for k in K]    # training fixed

    def s(x):
        return f"mean {np.mean(x):+.3f}  std {np.std(x):.3f}  range[{min(x):+.3f},{max(x):+.3f}]"
    print(f"=== VARIANCE DECOMPOSITION ({E}-LOO, with-xsum, batch={batch}) ===")
    print(f"  both vary                 : {s(total)}")
    print(f"  training-only (sub fixed) : {s(tronly)}   <- init+shuffle noise")
    print(f"  subsample-only (train fix): {s(subonly)}   <- which 200 drawn")
    print(f"\n=== PAIRED xsum test (same sub & train seed per pair) ===")
    wo = [_prr(fe, E, without_xsum, k, k, batch) for k in K]
    diffs = [total[k] - wo[k] for k in K]
    for k in K:
        print(f"  seed {k}: without {wo[k]:+.3f}  with {total[k]:+.3f}  diff {diffs[k]:+.3f}")
    thr = 2 * np.std(diffs) / max(len(diffs) ** 0.5, 1)
    print(f"  PAIRED diff: mean {np.mean(diffs):+.3f}  std {np.std(diffs):.3f}  -> "
          f"{'xsum HELPS' if np.mean(diffs) > thr else 'NO significant xsum effect'}")


if __name__ == "__main__":
    main()
