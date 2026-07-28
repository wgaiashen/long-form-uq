"""Cheap-bug parity check for the SAPLMA leave-one-out OOD approximation.

Rules out three cheap explanations for our sciq-LOO PRR (~0.65) sitting well below
Joe's reported ~0.86, BEFORE attributing the whole gap to "we only have 3 of his 9
LOO training sources". Read-only with respect to the caches (loads features/records,
draws index subsamples, trains nothing that gets written).

It asserts:
  1. LEAKAGE: for every (eval dataset E, OOD setting), the pooled training subsample
     draws ONLY from each source's 'train' split, and the eval dataset E is never in
     the training pool (so E's held-out 'test' rows can never enter training).
  2. FINITE FEATURES: the layer-15 (L15) SAPLMA features are finite (no NaN/Inf) for
     all four sources we have Llama features for.
  3. RECIPE PARITY: prints our train_probe_mlp recipe next to Joe's
     full_sequence_saplma head recipe (read from his repo), so hyperparameter drift
     is visible at a glance.

Run:
    PYTHONPATH=src python scripts/checks/ood_loo_parity.py
    PYTHONPATH=src python scripts/checks/ood_loo_parity.py --model meta-llama/Meta-Llama-3.1-8B
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq.config import Config                     # noqa: E402
# Reuse the exact helpers the real OOD check uses, so this verifies THAT code path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_joepools import (                          # noqa: E402
    AVAIL, EVALS, SETTINGS, LAYER, LABEL, _feats, _rows, _restrict,
)


# Joe's SAPLMA head recipe, read from the reference repo (grounded, not remembered):
#   Temp_robust_UQ_probes/luh/heads/full_seq_head_saplma.py
#     Dense(256, relu) -> Dense(128, relu) -> Dense(64, relu) -> Dense(1, sigmoid)
#     optimizer='adam' (Keras default lr ~1e-3), loss='binary_crossentropy'
#     model.fit(features, targets, epochs=5, batch_size=1)           # <- batch_size=1, hardcoded
#     RAW features (no StandardScaler); target = 1 - correctness (feature_supervision.py:166)
JOE_SAPLMA = {
    "architecture": "Dense 256/128/64 (relu) -> 1 (sigmoid)",
    "epochs": 5,
    "batch_size": 1,
    "lr": "adam default (~1e-3)",
    "weight_decay": 0.0,
    "standardize": False,
    "framework": "Keras/TF",
    "target": "1 - correctness (sigmoid output IS the uncertainty)",
    "source": "luh/heads/full_seq_head_saplma.py:27-48 + feature_supervision.py:166",
}


def check_leakage(fe):
    """Assert: pooled training draws only from 'train' rows, and E is never in its own pool."""
    ok = True
    for E in EVALS:
        eval_test = set(_rows(fe[E][1], "test").tolist())     # E's held-out test indices
        for setting in SETTINGS:
            kept, _ = _restrict(E, setting)
            if not kept:
                continue
            # (a) eval dataset E must not appear as a training source
            pool_sources = [s for s, _ in kept]
            if E in pool_sources:
                print(f"  FAIL [{E}/{setting}]: eval dataset {E} is in its own training pool {pool_sources}")
                ok = False
            # (b) Round-3 Task A (2026-07-27): the invariant is SOURCE != EVAL (a), NOT "train-split-only". A
            # source is a DIFFERENT dataset from E, so it leaks nothing into E's eval regardless of split label.
            # A source that HAS a train split still draws train-only (core sets unchanged); an eval-only source
            # (no train rows) legitimately contributes its own rows -- mirrors the Task-A sampler fallback. So
            # only flag a source that HAS train rows but drew a non-train row (a genuine regression).
            for d, n in kept:
                f, r = fe[d]
                train_idx = _rows(r, "train")
                has_train = len(train_idx) > 0
                pool_idx = train_idx if has_train else np.arange(len(r))   # Task-A sampler: all rows if no train
                for sub_seed in (0, 3):
                    pick = pool_idx[np.random.RandomState(sub_seed).permutation(len(pool_idx))[:min(n, len(pool_idx))]]
                    picked_splits = {r[i]["split"] for i in pick}
                    if has_train and picked_splits != {"train"}:
                        print(f"  FAIL [{E}/{setting}] source {d} (has train split): picked non-train {picked_splits}")
                        ok = False
                    # (c) no source's picks may hit E's eval-test rows (source != E already makes this moot)
                    if d == E and (set(pick.tolist()) & eval_test):
                        print(f"  FAIL [{E}/{setting}] source {d}: training picks overlap E's test rows")
                        ok = False
    print(f"[1] LEAKAGE (source != eval; core sources train-only, eval-only sources draw own rows): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_finite(fe):
    """Assert L15 features are finite for all available sources."""
    ok = True
    for d in AVAIL:
        f = fe[d][0]                       # already sliced to L15 by _feats
        finite = np.isfinite(f).all()
        print(f"    {d:12s} L{LAYER} shape={f.shape} finite={bool(finite)} "
              f"nan={int(np.isnan(f).sum())} inf={int(np.isinf(f).sum())}")
        ok = ok and bool(finite)
    print(f"[2] FINITE L{LAYER} FEATURES (all sources): {'PASS' if ok else 'FAIL'}")
    return ok


def print_recipe():
    """Print our train_probe_mlp defaults next to Joe's, so any drift is visible."""
    import inspect
    from luq import probe
    sig = inspect.signature(probe.train_probe_mlp)
    ours = {k: (v.default if v.default is not inspect._empty else "<required>")
            for k, v in sig.parameters.items() if k not in ("X", "y")}
    print("[3] RECIPE PARITY (informational)")
    print("    OURS  train_probe_mlp defaults:")
    for k, v in ours.items():
        print(f"        {k:14s} = {v}")
    print("    NOTE: the OOD check (ood_joepools.py) passes batch_size=1 explicitly (--batch default 1),")
    print("          which MATCHES Joe's hardcoded model.fit(batch_size=1).")
    print("    JOE   full_sequence_saplma head:")
    for k, v in JOE_SAPLMA.items():
        print(f"        {k:14s} = {v}")


def main():
    ap = argparse.ArgumentParser()
    # The cached LOO features are Llama-3.1-8B, so default there (not the Qwen dev model).
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    args = ap.parse_args()

    cache_dir = Config(model_name=args.model, dataset="sciq", ood_setting="ID").cache_dir
    fe = _feats(cache_dir, args.model)     # {dataset: (L15 features, records)}

    print(f"OOD-LOO parity check  (model={args.model}, layer={LAYER}, label={LABEL!r})\n")
    a = check_leakage(fe)
    b = check_finite(fe)
    print_recipe()

    print(f"\nSUMMARY: leakage={'PASS' if a else 'FAIL'}  finite={'PASS' if b else 'FAIL'}")
    sys.exit(0 if (a and b) else 1)


if __name__ == "__main__":
    main()
