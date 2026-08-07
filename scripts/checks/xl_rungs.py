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

from probe_drift.ood_settings import get_training_spec  # noqa: E402  (ProbeDrift's faithful rung spec)

# ⚠️ SHIM AS OF 2026-08-08. The shared pieces below — `label_of`, `different_label_projection`,
# `eval_split`, `build_rows` — now live in the installed `probe_drift_long` library and are
# re-exported here so the ~43 modules importing this file keep working unchanged. Their
# signatures are preserved EXACTLY (verified: no caller passes a positional second argument to
# `eval_split`, and every `build_rows` call uses the 5-positional PT form).
#
# What stays here: the XL grid (`ALL`, `FINE`, `BROAD`, `KEYSTONES`, `SETTINGS`, `rung_sources`,
# `cells`). That is a DIFFERENT and broader benchmark than ProbeDriftLong — 10 datasets including
# the short-form ones, with its own family names ("long_qa" where the long grid says
# "correctness_qa"). Merging the two taxonomies would be wrong, not tidy; `preflight_cohort.py`
# exists to check their GROUPINGS agree despite the naming, and it still should.
#
# New work should import from `probe_drift_long` directly rather than through this shim.
import probe_drift_long as _pdl  # noqa: E402

# Full dataset universe + task-family taxonomy (lifted from xl_eval_ladder, + expertqa as long-form QA).
ALL = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa",
       "factscore"]   # factscore added 2026-07-27 (Round-3 Task A): eval-only, but a valid TRAINING SOURCE
# ⚠️ FACTUALITY IS ITS OWN FAMILY (author's decision 2026-08-03, resolving the "property tag deferred" /
# "factuality-family split is a pending decision" placeholders left in data.py:32,43,47).
# BEFORE: expertqa, factscore AND asqa were all "long_qa", i.e. one undifferentiated QA family.
# NOW:
#   * expertqa + factscore -> FINE "factuality", and BROAD "factuality" sits BESIDE "qa" and "summ".
#     They are checked against WORLD KNOWLEDGE rather than answering a supplied question, which is the
#     factuality-vs-faithfulness axis in framing.md §1.1 -- a genuinely different task, not a subtype.
#   * asqa STAYS in "long_qa" (confirmed 2026-08-03). data.py:22 calls it "closed-book FACTUALITY QA",
#     which invites the opposite grouping, so this is recorded explicitly: asqa is a QA-family eval.
# CONSEQUENCES, so nobody has to rediscover them: expertqa's SameTask becomes {factscore} alone (was 4
# datasets) and factscore's becomes {expertqa}; pubmed/med_quad/asqa lose both from their SameTask pools;
# and because factuality is BROAD, it becomes a DiffTask source for the QA sets and vice versa.
# ⚠️ Every cell involving these three moves. Results computed before this date are NOT comparable.
FINE = {"sciq": "short_qa", "trivia_qa": "short_qa", "pubmed_qa": "long_qa", "med_quad": "long_qa",
        "expertqa": "factuality", "asqa": "long_qa", "xsum": "summ", "samsum": "summ",
        "cnn_dailymail": "summ", "factscore": "factuality"}
BROAD = {"short_qa": "qa", "long_qa": "qa", "summ": "summ", "factuality": "factuality"}

KEYSTONES = {"sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail"}   # -> get_training_spec (faithful)
# DERIVED, not re-typed (2026-08-04). It used to be the literal {med_quad, samsum, expertqa, asqa},
# which omitted factscore — harmless only because the branch below tests `in KEYSTONES` and takes the
# else. A constant that claims to be "the non-keystones" and isn't is a trap waiting for the first
# `x in XL_EVALS`, and that is precisely the shape of the three defects this file was just fixed for.
XL_EVALS = set(ALL) - KEYSTONES                                          # -> rung_sources (taxonomy)
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

# ---- WHICH CARVE RULE IS IN FORCE (added 2026-08-08) ---------------------------------------------
# "legacy"   : drop unlabelled rows, THEN carve 30% for test. What produced every committed number
#              up to and including the 1,664-cell master table.
# "all-rows" : carve 30% from ALL rows, THEN score whichever carry labels. Model-independent test
#              ROW SET, so two models are compared on the same rows. Changes expertqa + factscore
#              ONLY (verified: med_quad/samsum/asqa are byte-identical either way).
# Default stays LEGACY on purpose: flipping it would silently re-point every existing driver at a
# different population. The switch is deliberate and per-run, and the driver STAMPS it into the
# results CSV so a file can never be ambiguous about which rule produced it.
#     LUQ_CARVE=all-rows python scripts/checks/<driver>.py ...
CARVE = os.environ.get("LUQ_CARVE", "legacy")
if CARVE not in ("legacy", "all-rows"):
    raise SystemExit(f"LUQ_CARVE must be legacy|all-rows, got {CARVE!r}")


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
def eval_split(split, seed=0, test_frac=XL_TEST_FRAC, labelled=None):
    """(train_idx, test_idx) for an EVAL TARGET. Signature preserved; body delegates to the library.

    `labelled` is optional and only meaningful under LUQ_CARVE=all-rows, where `split` must be the
    FULL (unfiltered) array and the returned indices are positions in it. Under the default
    legacy carve this behaves exactly as it always did.
    """
    if CARVE == "all-rows":
        if labelled is None:
            raise SystemExit(
                "LUQ_CARVE=all-rows needs the `labelled` mask and the UNFILTERED split array. "
                "This caller still pre-filters unlabelled rows, so it cannot honour the new carve "
                "— run it under the default LUQ_CARVE=legacy, or update it to pass masks. "
                "(Refusing rather than silently falling back: a quiet fallback here would score "
                "the legacy population while the CSV claimed the new rule.)")
        return _pdl.eval_split(split, labelled, seed=seed, test_frac=test_frac, carve="all-rows")
    return _pdl.eval_split(split, seed=seed, test_frac=test_frac, carve="legacy")


# ---- rung generation -----------------------------------------------------------------------------
# OneDatasetDiffTask preference: a SINGLE opposite-broad-family dataset, matching ProbeDrift's
# OOD_ONE_DATASET_DIFF_TASK (a QA eval shifts to samsum; a summarisation eval shifts to med_quad).
# A third broad family needs its own preferred single-dataset shift, or `.get()` returns None for every
# factuality eval and OneDatasetDiffTask silently falls back to "whatever is first in the pool" — a rung
# whose composition would then depend on dict ordering rather than on a stated convention.
# factuality -> samsum: the largest available shift (summarisation), matching the QA convention.
_ONE_DIFF_PREF = {"qa": "samsum", "summ": "med_quad", "factuality": "samsum"}


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


def build_rows(X, spec, PT, seed, sampled_fn, labelled=None):
    """Build (train_rows, test_rows) for one cell, as lists of (dataset, idx). Signature preserved.

    `PT[d]` must be `(states, split, y, records)` — only index [1], the split array, is ever read.
    `labelled` ({dataset: bool mask}) is required only under LUQ_CARVE=all-rows, where PT must hold
    the UNFILTERED arrays.

    Delegates to `probe_drift_long.build_rows`, which owns the guard that a NAMED source
    contributing zero rows is a crash, not a quietly smaller pool.
    """
    splits = {d: PT[d][1] for d in PT}
    if CARVE == "all-rows":
        if labelled is None:
            raise SystemExit(
                "LUQ_CARVE=all-rows needs `labelled` masks and UNFILTERED PT arrays; this caller "
                "supplies neither. Run under LUQ_CARVE=legacy or update the caller.")
        return _pdl.build_rows(X, spec, splits, seed, sampled_fn, labelled=labelled,
                               carve="all-rows")
    return _pdl.build_rows(X, spec, splits, seed, sampled_fn, carve="legacy")


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
