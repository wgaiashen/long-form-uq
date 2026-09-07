# Pre-registration — retrospective development-set selection of the shrinkage level

> **Recorded outcome:** run. The two missing coefficients were computed and the selection is recorded as retrospective; no clean prospective development-set choice of the shrinkage level exists, which is what this registration was written to establish.

**Date:** 2026-08-13. **Population for selection:** short-form `sciq` and `trivia_qa` only.
**Population for the frozen look-up:** the existing 8-dataset ProbeDriftLong Llama results.
**Written to disk BEFORE the two missing cells (λ = 1, λ = 1.5) were computed.**

## 0. Why this exists, and what it is not

The historical audit (`results/analysis/WMSP_SHRINKAGE_MECHANISM.md` §D, and the reply of
2026-08-13) establishes that **no clean prospective development-set choice of λ exists**. λ = 2 and
λ = 10 were fixed on 2026-07-21, genuinely before the complete 8-dataset long grid existed
(2026-08-04), but the sweep that motivated them was run on **pubmed_qa** — one of the eight
evaluation datasets — on its pre-widened population. `prereg/shrinkage_lambda_and_nll_prior.md` §2.2 already
records this in terms: *"It is NOT 'λ was validated on independent data', and it must never be
written that way."*

This protocol therefore answers the criticism in the only way still available:

> **A retrospective development-set configuration-selection sensitivity.**

It is **not** prospective confirmation, and no wording in the report may imply that it is. Its value
is that the development data (`sciq`, `trivia_qa`) are **not members of the eight-dataset evaluation
set**, so the selection itself never touches an evaluation label.

## 1. Development data — fixed

Only `sciq` and `trivia_qa`. Neither is in the ProbeDriftLong main evaluation.
**No use of ASQA, XSum, FActScore or any of the eight long-form labels at any point in selection.**

## 2. Candidates — fixed, closed

    lambda in {0, 1, 1.5, 2, 10}

No additional value will be created, before or after seeing any result.

## 3. Cells

Exact final wMSP implementation (`weight_mode="normalised"`, `length_normalise=True`,
`loss="pairwise"`, `reg=shrink_to_uniform`), training budget **1800**, **3 seeds** (1, 2, 3).

| role | train | evaluate |
|---|---|---|
| **PRIMARY** | `sciq` (1800) | `trivia_qa` |
| secondary stability check only | `trivia_qa` (1800) | `sciq` |

TriviaQA is primary because SciQ's labels are ~94% correct, leaving too little error mass for a
ranking measure. The reverse direction is reported for stability and **is not part of the selection
rule**.

## 4. Selection rule — fixed before the missing cells are computed

> **Choose the λ with the highest 3-seed mean PRR on `sciq` → `trivia_qa`.**

No significance threshold. No averaging with any long-form dataset. No tie-breaking on long-form
results: if two λ tie exactly at 4 dp, the **smaller** λ is chosen (declared here, not afterwards).

Then **freeze** that λ and simply look up its already-existing ProbeDriftLong values. **The λ is not
changed if its long-form result is disappointing.** That commitment is the whole point of the
exercise and is the only thing that makes it worth running.

## 5. Mandatory disclosure — partial prior observation

Three of the five arms **already exist** and **have already been seen** at the time of writing, as a
by-product of the cross-length transfer experiment (`results/xlen_long2short_othershort_*.csv`,
3 seeds, run 2026-08-12):

| arm | `sciq` → `trivia_qa` | `trivia_qa` → `sciq` |
|---|---:|---:|
| λ = 0 (`wmsp_norm`) | +0.7209 ± 0.0104 | +0.6054 ± 0.0248 |
| λ = 2 (`wmsp_shrink2`) | +0.7293 ± 0.0052 | +0.6073 ± 0.0316 |
| λ = 10 (`wmsp_shrink10`) | +0.7361 ± 0.0095 | +0.5708 ± 0.0091 |

λ = 1 and λ = 1.5 have **not** been computed on either cell. So this is a selection over five arms
of which three were visible in advance. That is weaker than a blind selection and **must be stated
wherever the result is reported.** It is disclosed here, before the remaining two arms exist, for the
same reason the M2 pre-registration disclosed W6's partial observation.

On the visible arms alone the ranking is λ=10 > λ=2 > λ=0; the two missing arms sit between λ=0 and
λ=2 in strength and could change the argmax.

## 6. Compute required

Only the two missing arms, on two cells, at three seeds: **12 wMSP trainings** on short-form data
(sciq/trivia generations are 2–20 tokens, so each is far cheaper than a long-form cell).

Blocker: no committed driver does this combination. `scripts/checks/sharpening_lambda.py` has
λ ∈ {0, 1, 1.5} but no `--train-spec`; the cross-length driver has `--train-spec` (committed
`53668c3`) but a fixed wMSP variant list. **One small driver change plus one qsub.** No canonical
artifact is touched; output goes to `results/analysis/wmsp_dev_selection__<slug>.csv`.

## 7. Reporting requirements

- Report all five arms on both directions, not only the winner.
- Label the whole exercise **retrospective**.
- Report the selected λ's existing long-form values **as a look-up**, never as a fresh evaluation.
- Report the five-arm leave-one-dataset-out result separately as an *upper bound on what
  test-selection optimism would have been* (λ = 1.5 on 7 of 8 folds, +0.2245, against the best fixed
  value's +0.2306 — optimism +0.0061), and do not conflate it with this development selection.
