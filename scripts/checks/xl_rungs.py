"""Shared ProbeDrift(XL) rung generation — makes med_quad / samsum / ExpertQA ORGANIC eval targets.

The 5 core ProbeDrift datasets (sciq/trivia/pubmed/xsum/cnn) get their EXACT rungs from `get_training_spec`
(faithful reproduction, numbers unchanged). The XL long-form additions (med_quad, samsum, expertqa) are not in
ProbeDrift's `VALID_EVAL_DATASETS`, so `get_training_spec` raises for them; they instead get their rungs from a
task-family taxonomy (`rung_sources`). `cells()` DISPATCHES per eval target, so every consumer ladder covers all
of them for ID+OOD just by listing them in its evals.

Two per-target properties the ladders must respect:
  * LABEL (`label_of`): the correctness-world datasets use `correctness`; ExpertQA uses `factuality`
    (correctness-vs-gold was tested and rejected — its gold differs too much from the generation). So an ExpertQA
    OOD rung (correctness-world sources -> factuality eval) is flagged `different_label_projection`: the
    target is scored on CLAIM PRECISION while the pool was trained on REFERENCE AGREEMENT (see that
    function's docstring -- the old "factuality vs factuality" framing was wrong).
    UPDATE 2026-07-22 (author's decision): ExpertQA and ASQA are ORDINARY TRAINING SOURCES, not eval-only,
    so pools are MIXED-LABEL by default and the flag is how those cells stay identifiable.
  * EVAL SPLIT (`eval_split`): the XL sets are split-less (ExpertQA all-`test`; med_quad/samsum all-`train`), so
    the eval target's held-out test set is CARVED here (fixed & deterministic), NOT baked into `load_per_token`.
    This keeps med_quad/samsum's use as TRAINING SOURCES = all their rows (so the core-5 rungs don't move).
"""
import os

import numpy as np

from probe_drift.ood_settings import get_training_spec  # noqa: E402  (ProbeDrift's faithful rung spec)

# Full dataset universe + task-family taxonomy (lifted from xl_eval_ladder, + expertqa as long-form QA).
ALL = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa",
       "factscore"]   # factscore added 2026-07-27 (Round-3 Task A): eval-only, but a valid TRAINING SOURCE
FINE = {"sciq": "short_qa", "trivia_qa": "short_qa", "pubmed_qa": "long_qa", "med_quad": "long_qa",
        "expertqa": "long_qa", "asqa": "long_qa", "xsum": "summ", "samsum": "summ", "cnn_dailymail": "summ",
        "factscore": "long_qa"}
BROAD = {"short_qa": "qa", "long_qa": "qa", "summ": "summ"}

KEYSTONES = {"sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail"}   # -> get_training_spec (faithful)
XL_EVALS = {"med_quad", "samsum", "expertqa", "asqa"}                    # -> rung_sources (taxonomy)
# TRAINING SOURCE POOL = every dataset (author's decision 2026-07-22): ASQA and ExpertQA are treated the
# SAME as the rest, not eval-only. Consequence to keep visible: ExpertQA carries a FAITHFULNESS label while
# the others carry correctness, so pools that include it are MIXED-LABEL. That is deliberate -- what used to
# be the separate `--include-expertqa` "universal" variant is now the default. Every affected cell is still
# tagged `different_label_projection` in the CSVs, so mixed-label cells remain identifiable during analysis.
# NOTE: this enlarges the training pool for all 10 drivers importing this module, so results computed under
# it are NOT comparable to the pre-2026-07-22 committed numbers -- new runs write to NEW files and the
# earlier baseline is preserved at results/_baseline_preasqa_2026-07-22/.
SOURCE_POOL = list(ALL)

SETTINGS = [("SameTask", "OOD_ONE_DATASET_SAME_TASK"), ("LOO", "OOD_LEAVE_ONE_OUT"),
            ("OneDatasetDiffTask", "OOD_ONE_DATASET_DIFF_TASK"), ("DiffTask", "OOD_DIFF_TASK")]
XL_TOTAL = 1800          # matched training budget per XL OOD rung (split across its sources)
XL_TEST_FRAC = 0.30      # held-out test carved from a split-less XL eval target


# ---- per-target label ----------------------------------------------------------------------------
# ExpertQA ships TWO independently-computed judge labels, and which one we score on is a real research
# choice, so it is switchable from the environment (no per-driver flag needed -- every driver that calls
# label_of() picks it up):
#   factuality (default) -- SUPPORTED/(SUPPORTED+CONTRADICTED) over the claims the expert reference can
#                             adjudicate. Judge-knowledge-independent, but only covers ~44% of the answer's
#                             claims and is NULL for 292/2016 rows (all-claims-uncovered).
#   consistency             -- the WHOLE-ANSWER judge (~0 blind spot): 2016/2016 rows labelled, harsher
#                             (mean 0.529 vs 0.678), correlates r=0.72 / rho=0.73 with factuality, and is
#                             NOT more length-biased (-0.205 vs -0.227), which was the standing objection.
# Running both and comparing is the point: if the method ranking is stable across two independent label
# definitions that is genuine robustness; if it flips, that is itself the finding.
#     LUQ_EXPERTQA_LABEL=consistency python scripts/checks/<driver>.py ...
_EXPERTQA_LABEL = os.environ.get("LUQ_EXPERTQA_LABEL", "factuality")
if _EXPERTQA_LABEL not in ("factuality", "consistency"):
    raise SystemExit(f"LUQ_EXPERTQA_LABEL must be factuality|consistency, got {_EXPERTQA_LABEL!r}")
_LABEL_OF = {"expertqa": _EXPERTQA_LABEL, "factscore": "factuality"}   # factscore = ExpertQA's factuality partner


def label_of(dataset):
    """The correctness signal to score `dataset` on (default `correctness`; ExpertQA -> `factuality`)."""
    return _LABEL_OF.get(dataset, "correctness")


def different_label_projection(eval_dataset):
    """True if this eval target's label measures a DIFFERENT PROJECTION of "good" than the training sources.

    Renamed from `cross_label` (2026-07-22) because that name implied the old factuality-vs-factuality
    story, which reading the three judge prompts showed to be wrong. The real distinction:

      * QA judge (sciq/trivia/pubmed/med_quad) and SUMMARISATION judge (xsum/cnn/samsum) both score
        REFERENCE AGREEMENT -- how much the output matches the gold answer / gold summary. Both penalise
        incompleteness. (Note this means our summarisation label is NOT factuality-to-source: the article
        is in the judge's context but the criterion is match-to-gold-summary.)
      * ExpertQA scores CLAIM PRECISION -- SUPPORTED/(SUPPORTED+CONTRADICTED) -- and is explicitly
        instructed NOT to reward similarity to the reference and NOT to penalise being less complete.

    So the flag marks "the target is scored on a different projection than the pool was trained on", which
    is the thing that actually matters when pooling. Extra caveats for ExpertQA specifically: its label is
    blind to ~56% of the answer's claims (mean `uncovered`) and is absent on 14.5% of rows. See PART X.
    """
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
# OneDatasetDiffTask preference: a SINGLE opposite-broad-family dataset, matching ProbeDrift's
# OOD_ONE_DATASET_DIFF_TASK (a QA eval shifts to samsum; a summarisation eval shifts to med_quad).
_ONE_DIFF_PREF = {"qa": "samsum", "summ": "med_quad"}


def rung_sources(X):
    """Task-family rung composition for an XL eval target X (SameTask=same fine-family, DiffTask=opposite
    broad-family, LOO=all-others, OneDatasetDiffTask=ONE opposite-broad-family dataset), excluding X and the
    eval-only ExpertQA. Mirrors ProbeDrift's semantics."""
    same = [d for d in SOURCE_POOL if d != X and FINE[d] == FINE[X]]
    diff = [d for d in SOURCE_POOL if d != X and BROAD[FINE[d]] != BROAD[FINE[X]]]
    loo = [d for d in SOURCE_POOL if d != X]
    # ordered: the ProbeDrift-canonical single opposite-family dataset first, then the rest as fallbacks
    pref = _ONE_DIFF_PREF.get(BROAD[FINE[X]])
    one_diff = ([pref] if pref in diff else []) + [d for d in diff if d != pref]
    return {"SameTask": same, "DiffTask": diff, "LOO": loo, "OneDatasetDiffTask": one_diff}


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
            idx = list(X_tr if cap is None else np.asarray(X_tr)[:cap])
        else:                                         # OOD source: sampler draws its source rows. Task A
            idx = list(sampled_fn(PT[d][1], seed, cap))   # (2026-07-27): eval-only sets have no split=="train",
        # BUILD-TIME GUARD (Round-3 Task A / V-A0): a NAMED source (d != X) contributing ZERO realised rows is
        # the silent-admission bug -- a pool label that overstates its contents. Fail loud rather than train on
        # a smaller-than-labelled pool. Post-fix every listed source draws its rows (eval-only sets from all
        # rows, source != eval enforced by cells()), so a 0 here is a genuine bug worth crashing on.
        if d != X and len(idx) == 0:
            raise SystemExit(f"build_rows: source '{d}' for eval '{X}' contributed 0 rows (cap={cap}). "
                             f"Eval-only sets must draw from ALL rows via the Task-A sampler fix; see V-A0.")
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
            for tag in ("SameTask", "LOO", "DiffTask"):    # multi-source rungs: split the budget across sources
                srcs = [d for d in rs[tag] if d in sources]
                if srcs:
                    cap = max(1, XL_TOTAL // len(srcs))
                    out.append((tag, X, [(d, cap) for d in srcs]))
            one = [d for d in rs["OneDatasetDiffTask"] if d in sources]   # single opposite-family dataset, full budget
            if one:
                out.append(("OneDatasetDiffTask", X, [(one[0], XL_TOTAL)]))
    return out
