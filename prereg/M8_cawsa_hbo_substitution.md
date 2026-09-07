# M8 — Substituting a learned token weighting for the probability branch of the hybrid back-off

> **Recorded outcome:** run. Substituting the probability branch raises in-distribution PRR substantially and leaves the shifted settings close to the substituted score, because the gate gives the probe almost no weight there.

Written 2026-08-24, before any prediction-rejection ratio for the modified estimator has been
computed, inspected or interpreted.

## 0. What this registers, and what is already known

The reproduction of the published hybrid back-off is complete on two corrected-span populations at
40 of 40 cells. That reproduction was the pre-condition set by `M7_published_hybrid_baselines.md`
section 2 for modifying any part of the published estimator, and it has been met. This document
registers the modification.

Two things are already known and are not registered as discoveries:

1. The learned token weighting at the frozen shrinkage setting outperforms the published sequence
   probability by a wide margin on long-form generation, on every population measured.
2. The published back-off assigns zero weight to its supervised branch for every evaluation example
   on the most shifted rung, so on that rung the published estimator is exactly its probability
   branch. This is recorded in the diagnostics of the completed grid.

Point 2 has a consequence that must be stated before any number is read, because it would otherwise
look like a finding. **On any rung where the supervised weight is zero everywhere, the modified
estimator is exactly the learned weighting and the difference from the published estimator is exactly
the difference between the two probability scores, which is already known.** That is arithmetic, not
evidence. It will be reported as a confirmation that the substitution behaves as designed, never as a
result about the back-off.

The genuinely open question is confined to the rungs where the supervised weight is non-zero:

> When the back-off still trusts its probe, does a stronger long-form probability branch make the
> combination better than the published one, and does either combination beat the probe alone?

## 1. The estimator

Unchanged from the reference implementation `run_hbo.py` in every respect except one input.

    published:  score = w_unsup * rank(sequence NLL)      + w_sup * rank(probe uncertainty)
    modified:   score = w_unsup * rank(learned weighting) + w_sup * rank(probe uncertainty)

with `w_unsup = min(1, p + 0.5)`, `w_sup = 1 - w_unsup`, and `p` the percentile of the example's mean
token Mahalanobis distance against the held-out half of the training pool. The distance, its
percentile, the rank transformation, the probe, the training pool, the sampled training rows, the
evaluation cohort, the carve seed, the label field and the corrected generation span are all
identical to the published run, taken from the same code path in the same process.

The substituted score is the frozen report-facing learned weighting at shrinkage 2, read from the
per-example sidecar written by the canonical ladder. It is not refitted, reselected or rescaled.

## 2. What is fixed and may not be varied

- No search over the shrinkage parameter. The single frozen setting is used.
- No search over the combination weights. The published percentile rule is used unmodified.
- No search over the distance layer, the metric threshold, or the dev split.
- No selection of any kind on an out-of-distribution rung, on a test cohort, or on any quantity
  computed from evaluation labels.
- If the modified estimator loses, that is the result. There is no fallback variant registered.

## 3. Predictions

Registered before the run.

- **P1.** On the most shifted rung the modified estimator will equal the learned weighting to
  floating-point noise on both populations, because the supervised weight is zero for every example
  there. Verified numerically as a correctness check on the implementation, not reported as a gain.
- **P2.** On the matched rung the modified estimator will not beat the probe alone. The published
  back-off already loses to the probe there, and improving the probability branch narrows that gap
  rather than closing it.
- **P3.** On the intermediate rungs, where the supervised weight is non-zero for some examples, the
  modified estimator will beat the published one. The direction is close to certain given point 1 of
  section 0; what is not known is whether the margin survives a dataset-level test.
- **P4.** The modified estimator will not beat the frozen learned weighting on its own by a margin
  that survives a dataset-level test on any rung. The back-off can only dilute the stronger branch
  with a weaker one wherever the supervised weight is non-zero, and the probe is the weaker branch
  under shift. If this prediction is wrong, that is the interesting outcome of the experiment.

P4 is the prediction the experiment exists to test. A reader should be able to check afterwards that
it was written down first.

## 4. Analysis

The unit of analysis is the **dataset**, n = 8, as everywhere else in this project. Cell-level counts
are pseudo-replication for the unsupervised scores and are not used as the unit.

Reported for each population and each of the five rungs:

- macro prediction-rejection ratio for: published sequence probability, minimum token probability,
  the frozen learned weighting, the probe, the published back-off, the modified back-off;
- per-dataset values for the same six;
- paired differences of the modified back-off against the published back-off, against the learned
  weighting, and against the probe;
- the sign count over the eight datasets;
- the exact Wilcoxon signed-rank p-value, two-sided, computed on the eight paired differences;
- a bootstrap interval over datasets;
- the mean supervised weight and the fraction of evaluation examples in the far-shifted region.

Seeds are averaged as per-seed prediction-rejection ratios, never by scoring a seed-averaged score
vector. A constant score is recorded as blank with an explicit degeneracy count, never converted into
a number by row-order tie breaking.

## 5. Populations

The corrected-span eight-dataset populations, in the state fixed by
`results/analysis/CLEAN_CORE_POPULATION_MANIFEST.json`. Two are runnable from existing artifacts and
are in scope now: `meta-llama/Meta-Llama-3.1-8B` and `google/gemma-2-9b`. A third,
`Qwen/Qwen2.5-14B`, is in scope if and only if its published back-off has been produced first on the
same corrected population; it is not to be run as a modified estimator without its own unmodified
reference on the same rows.

No population is pooled with another. Each is reported as a separate replication.

## 6. Verification required before any number is reported

1. The probability branch actually changed: the substituted vector must differ from the published one
   on every cell, and must equal the sidecar's frozen learned weighting exactly.
2. The supervised branch did not change: the probe vector must be bit-identical between the published
   and modified runs of the same cell and seed.
3. The gate did not change: the distance percentile and the supervised weight must be bit-identical
   between the two runs.
4. Where the supervised weight is exactly zero for every example, the modified score must equal the
   rank transform of the learned weighting exactly, and its prediction-rejection ratio must equal
   that of the learned weighting to storage resolution.
5. Score orientation is confirmed against the scoring function for both probability branches before
   they are combined, since a sign error would silently invert the comparison.

Failure of any of 1 to 5 aborts the run rather than producing a corrected number.
