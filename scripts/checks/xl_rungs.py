"""Shared ProbeDrift(XL) rung generation — makes med_quad / samsum / ExpertQA ORGANIC eval targets.

The 5 core ProbeDrift datasets (sciq/trivia/pubmed/xsum/cnn) get their EXACT rungs from `get_training_spec`
(faithful reproduction, numbers unchanged). The XL long-form additions (med_quad, samsum, expertqa) are not in
ProbeDrift's `VALID_EVAL_DATASETS`, so `get_training_spec` raises for them; they instead get their rungs from a
task-family taxonomy (`rung_sources`). `cells()` DISPATCHES per eval target, so every consumer ladder covers all
of them for ID+OOD just by listing them in its evals.

Two per-target properties the ladders must respect:
  * LABEL (`label_of`): the correctness-world datasets use `correctness`; ExpertQA uses `faithfulness`
    (correctness-vs-gold was tested and rejected — its gold differs too much from the generation). So an ExpertQA
    OOD rung (correctness-world sources -> faithfulness eval) is CROSS-LABEL — a documented factuality<->
    faithfulness axis (`cross_label`), not a bug. ExpertQA is therefore EVAL-ONLY (never a training source: its
    faithfulness label must not leak into a correctness training pool).
  * EVAL SPLIT (`eval_split`): the XL sets are split-less (ExpertQA all-`test`; med_quad/samsum all-`train`), so
    the eval target's held-out test set is CARVED here (fixed & deterministic), NOT baked into `load_per_token`.
    This keeps med_quad/samsum's use as TRAINING SOURCES = all their rows (so the core-5 rungs don't move).
"""
import numpy as np

from probe_drift.ood_settings import get_training_spec  # noqa: E402  (ProbeDrift's faithful rung spec)

# Full dataset universe + task-family taxonomy (lifted from xl_eval_ladder, + expertqa as long-form QA).
ALL = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa"]
FINE = {"sciq": "short_qa", "trivia_qa": "short_qa", "pubmed_qa": "long_qa", "med_quad": "long_qa",
        "expertqa": "long_qa", "xsum": "summ", "samsum": "summ", "cnn_dailymail": "summ"}
BROAD = {"short_qa": "qa", "long_qa": "qa", "summ": "summ"}

KEYSTONES = {"sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail"}   # -> get_training_spec (faithful)
XL_EVALS = {"med_quad", "samsum", "expertqa"}                            # -> rung_sources (taxonomy)
SOURCE_POOL = [d for d in ALL if d != "expertqa"]                        # ExpertQA is eval-only (see LABEL note)

SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"), ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("OneDatasetDiffTask", "OOD_ONE_DATASET_DIFF_TASK"), ("DiffTask", "OOD_DIFF_TASK")]
XL_TOTAL = 1800          # matched training budget per XL OOD rung (split across its sources)
XL_TEST_FRAC = 0.30      # held-out test carved from a split-less XL eval target


# ---- per-target label ----------------------------------------------------------------------------
_LABEL_OF = {"expertqa": "faithfulness"}


def label_of(dataset):
    """The correctness signal to score `dataset` on (default `correctness`; ExpertQA -> `faithfulness`)."""
    return _LABEL_OF.get(dataset, "correctness")


def cross_label(eval_dataset):
    """True if the eval target's label differs from the correctness-world training sources (only ExpertQA)."""
    return label_of(eval_dataset) != "correctness"


# ---- eval-target train/test split ----------------------------------------------------------------
def eval_split(split, seed=0, test_frac=XL_TEST_FRAC):
    """(train_idx, test_idx) for an EVAL TARGET. If the dataset has a real baked-in train/test split (the
    core datasets), use it. If it is split-less (an XL set — all one split), carve a FIXED deterministic
    train/test (seed=0) so, exactly like the core datasets, there is ONE stable test set: the per-run seed
    then varies only the training subsample, never the test set."""
    split = np.asarray(split)
    if len(np.unique(split)) >= 2:
        return np.where(split == "train")[0], np.where(split == "test")[0]
    n = len(split)
    perm = np.random.RandomState(seed).permutation(n)
    n_te = int(round(n * test_frac))
    return perm[n_te:], perm[:n_te]


# ---- rung generation -----------------------------------------------------------------------------
def rung_sources(X):
    """Task-family rung composition for an XL eval target X (SameTask=same fine-family, DiffTask=opposite
    broad-family, LOO=all-others), excluding X and the eval-only ExpertQA. Mirrors ProbeDrift's semantics."""
    same = [d for d in SOURCE_POOL if d != X and FINE[d] == FINE[X]]
    diff = [d for d in SOURCE_POOL if d != X and BROAD[FINE[d]] != BROAD[FINE[X]]]
    loo = [d for d in SOURCE_POOL if d != X]
    return {"SameTask": same, "DiffTask": diff, "LOO": loo}


def build_rows(X, spec, PT, seed, sampled_fn):
    """Build (train_rows, test_rows) for one cell, as lists of (dataset, idx). The EVAL TARGET X uses its
    fixed eval_split (baked for core, deterministic carve for XL); OOD sources use `sampled_fn(split, seed,
    cap)` (unchanged for the core). This centralises the XL-aware split logic so every ladder wires the same
    way. `PT[d]` must be `(states, split, y, records)` (split is index [1])."""
    X_tr, X_te = eval_split(PT[X][1])
    test_rows = [(X, int(i)) for i in X_te]
    train_rows = []
    for d, cap in spec:
        if d == X:                                    # ID cell: train on the eval target's OWN train split
            idx = X_tr if cap is None else np.asarray(X_tr)[:cap]
        else:                                         # OOD source: sample from its train rows (core unchanged)
            idx = sampled_fn(PT[d][1], seed, cap)
        train_rows += [(d, int(i)) for i in idx]
    return train_rows, test_rows


def cells(sources, evals):
    """[(rung_tag, X, spec)] for each eval X in `evals` that is cached (`X in sources`). spec = [(source, cap)]
    (cap=None on the ID cell = 'all of X's train rows'). DISPATCH: keystone X -> get_training_spec (faithful,
    unchanged); XL X -> rung_sources (taxonomy). The eval target's test rows come from eval_split(), not here."""
    out = []
    for X in evals:
        if X not in sources:
            continue
        out.append(("ID", X, [(X, None)]))
        if X in KEYSTONES:
            for tag, setting in SETTINGS:
                spec = [(s, n) for s, n in get_training_spec(X, setting) if s in sources and s != X]
                if spec:
                    out.append((tag, X, spec))
        else:                                             # XL taxonomy rungs
            rs = rung_sources(X)
            for tag in ("SameTask", "LOO", "DiffTask"):
                srcs = [d for d in rs[tag] if d in sources]
                if srcs:
                    cap = max(1, XL_TOTAL // len(srcs))
                    out.append((tag, X, [(d, cap) for d in srcs]))
    return out
