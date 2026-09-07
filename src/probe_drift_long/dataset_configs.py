"""The long-form dataset universe and its task-family taxonomy.

This is the single definition of the dataset universe and its task families. It was previously
duplicated across several analysis modules, which then needed a separate check that the copies
still agreed. Import from here rather than adding another copy.

The order of `LONG_SRC` is load-bearing. The `1ds-Diff-long` rung takes the first source from a
different task family, so re-ordering this list changes that rung's composition on every
evaluation target, silently and without error. It must not be sorted, de-duplicated or otherwise
tidied.
"""

# ---------------------------------------------------------------------------------------------
# The eight long-form evaluation targets.
# ---------------------------------------------------------------------------------------------
LONG_DATASETS = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum",
                 "expertqa", "asqa", "factscore"]

# Training-source pool for the long grid. Same membership as LONG_DATASETS: ASQA and ExpertQA are
# ordinary training sources rather than evaluation-only targets, so a pool can mix label
# definitions (ExpertQA carries factuality, the rest carry correctness). Cells built from a mixed
# pool remain identifiable through `different_label_projection`. A dataset is always excluded from
# its own evaluation's sources by `cells_long`, so training data never leaks into the test set.
# This is kept as a separate list from LONG_DATASETS, and in this order, because `1ds-Diff-long`
# selects the first different-family entry, as described in the module docstring.
LONG_SRC = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum",
            "expertqa", "asqa", "factscore"]

# Short-form targets, reachable only through the `Long->Short` transfer rung.
SHORT_DATASETS = ["sciq", "trivia_qa"]

# ---------------------------------------------------------------------------------------------
# Fine task families (the 2026-07-27 / 2026-08-03 factuality split).
# ---------------------------------------------------------------------------------------------
# correctness-QA (agreement with a gold answer) is kept SEPARATE from factuality (claim support
# against an external reference), so "SameTask" means same PROPERTY, not merely same surface form.
#   * expertqa + factscore -> "factuality": checked against WORLD KNOWLEDGE rather than answering
#     a supplied question. Checking against world knowledge and checking against a supplied
#     source are different tasks rather than variants of one, so they are separate families.
#   * asqa stays in correctness-QA. It is sometimes described as closed-book factuality question
#     answering, which invites the opposite grouping, so the choice is recorded explicitly here.
# CONSEQUENCES, so nobody rediscovers them: expertqa's SameTask is {factscore} alone and
# factscore's is {expertqa}; pubmed/med_quad/asqa lose both from their SameTask pools.
FINE_FAMILIES = {
    "pubmed_qa": "correctness_qa", "med_quad": "correctness_qa", "asqa": "correctness_qa",
    "expertqa": "factuality", "factscore": "factuality",
    "xsum": "summ", "cnn_dailymail": "summ", "samsum": "summ",
}

# ---------------------------------------------------------------------------------------------
# Per-target label field.
# ---------------------------------------------------------------------------------------------
# ExpertQA ships two independently-computed judge labels and which one we score on is a real
# research choice, so it stays switchable from the environment (every caller of `label_of` picks
# it up without a per-driver flag):
#   factuality (default) -- SUPPORTED/(SUPPORTED+CONTRADICTED) over the claims the expert
#                           reference can adjudicate. Judge-knowledge-independent, but covers only
#                           ~44% of the answer's claims and is NULL on 292/2016 rows.
#   consistency          -- the whole-answer judge (~0 blind spot): 2016/2016 labelled, harsher
#                           (mean 0.529 vs 0.678), r=0.72 with factuality, and NOT more
#                           length-biased (-0.205 vs -0.227), which was the standing objection.
#     LUQ_EXPERTQA_LABEL=consistency python scripts/checks/<driver>.py ...
_LABEL_OVERRIDES = {"expertqa": None, "factscore": "factuality"}   # expertqa filled in at import
DEFAULT_LABEL = "correctness"

# Datasets whose quality label can be absent on some rows, and why. The absence is permanent and
# model-dependent: `factuality` is None when `uncovered == 1.0`, i.e. the judge ran and found no
# claim the reference could adjudicate. Measured on Llama-3.1-8B: expertqa 292/2016 (85.5%
# coverage), factscore 45/500 (91.0%). The judge RAN on 100% of rows in both -- these are not
# "unjudged", they are "judged, no denominator", so they can never be filled in.
# The dropped rows are significantly SHORTER, not longer (expertqa median 177 vs 200.5,
# p=1.2e-4; factscore 47 vs 90, p=3.4e-4): `uncovered == 1.0` needs ZERO adjudicable claims,
# which is easiest when the answer makes few claims. This is why the split is carved over ALL
# rows -- see splits.py.
PARTIALLY_LABELLED = {"expertqa", "factscore"}


def label_of(dataset, expertqa_label="factuality"):
    """The correctness signal `dataset` is scored on (default `correctness`)."""
    if dataset == "expertqa":
        return expertqa_label
    return _LABEL_OVERRIDES.get(dataset) or DEFAULT_LABEL


def different_label_projection(eval_dataset, expertqa_label="factuality"):
    """True if this eval target's label measures a DIFFERENT PROJECTION of "good" than its sources.

    The real distinction (established by reading the three judge prompts, not assumed):
      * The QA judge (sciq/trivia/pubmed/med_quad) and the SUMMARISATION judge (xsum/cnn/samsum)
        both score REFERENCE AGREEMENT -- how well the output matches the gold answer or gold
        summary. Both penalise incompleteness. Note this means our summarisation label is NOT
        factuality-to-source: the article is in the judge's context, but the criterion is
        match-to-gold-summary.
      * ExpertQA scores CLAIM PRECISION and is explicitly instructed NOT to reward similarity to
        the reference and NOT to penalise being less complete.

    So the flag marks "this target is scored on a different projection than its pool was trained
    on", which is what actually matters when pooling.
    """
    return label_of(eval_dataset, expertqa_label) != DEFAULT_LABEL
