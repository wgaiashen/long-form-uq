"""THE canonical dataset cohort. One definition, imported — never re-typed in a driver.

WHY THIS EXISTS (the origin of three separate defects on 2026-08-03/04)
-----------------------------------------------------------------------
Every driver declared its OWN dataset list. `sweep_silent_drops.py` found **27** such lists. Three of
them were quietly too small, and each produced a wrong table rather than an error:

  * `ood_onegrid.AVAIL` = 4 datasets -> its OOD pools differed from contribution_ladder's, so the SAME
    rung name meant two different training populations and the assembler merged them (8 flagged cells).
  * `ood_onegrid.cells()` = 3 rungs -> linear/ptrue/lookback capped at 30/50 XL cells (sat at 9/50).
  * `contribution_ladder.CANDIDATE_SOURCES` = 9 datasets, missing factscore -> expertqa's SameTask rung
    vanished ENTIRELY (its only same-family partner after the split IS factscore), and every LOO and
    DiffTask pool across all ten evals was built without it.

THE ROOT CAUSE IS NOT ANY OF THOSE LISTS. It is that **a script's scope silently outgrew what it was
written for.** `ood_onegrid` was CORRECT as a 3-eval baseline sanity check — that is what it was built
for. It became wrong the moment it was made the XL table's only source of supervised baselines, because
nobody re-derived its assumptions against the new job. A hard-coded cohort is fine until the script is
reused; nothing anywhere connected "this list" to "the study's datasets", so reuse could not fail loudly.

So the fix is not "check the lists again more carefully" — that is what failed. The fix is to make the
study's cohort a SINGLE IMPORTED FACT and to make any divergence a hard error at submit time, before
compute burns. See `preflight_cohort.py`.
"""

# The ten datasets of the study.
CANONICAL_10 = ["sciq", "trivia_qa", "pubmed_qa", "med_quad", "asqa",
                "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]

# The eight long-form ones (the ProbeDriftLong grid); sciq/trivia are short and appear only as the
# Long->Short transfer targets.
LONG_8 = ["pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa", "asqa", "factscore"]
SHORT_2 = ["sciq", "trivia_qa"]

# Task families. factuality is its OWN broad family: those sets are
# checked against WORLD KNOWLEDGE rather than against a supplied question. asqa stays in QA despite
# data.py describing it as "closed-book factuality QA" — recorded explicitly because that phrasing
# invites the opposite grouping.
FINE = {"sciq": "short_qa", "trivia_qa": "short_qa",
        "pubmed_qa": "long_qa", "med_quad": "long_qa", "asqa": "long_qa",
        "xsum": "summ", "cnn_dailymail": "summ", "samsum": "summ",
        "expertqa": "factuality", "factscore": "factuality"}
BROAD = {"short_qa": "qa", "long_qa": "qa", "summ": "summ", "factuality": "factuality"}

# The five XL rungs and the five long rungs, so a driver cannot invent a fourth definition of "a rung".
XL_RUNGS = ["ID", "SameTask", "LOO", "DiffTask", "OneDatasetDiffTask"]
LONG_RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]

# Datasets whose cache lives in a namespaced regime. A `cache/records/*` glob silently returns 7 of 10
# without this map — the reason it is here rather than in three separate drivers.
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}

# Label field per dataset: expertqa/factscore carry FACTUALITY, a different projection of "good".
# Never pool their PRR with the correctness sets unflagged.
LABEL_FIELD = {d: "correctness" for d in CANONICAL_10}
LABEL_FIELD["expertqa"] = "factuality"
LABEL_FIELD["factscore"] = "factuality"
