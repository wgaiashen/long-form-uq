# M10 — Source-calibrated fusion of the learned token weighting and the hidden-state probe

> **Recorded outcome:** run. Complete on two populations at 40 of 40 cells, with all four registered gates passing exactly. The third population was excluded on a pre-specified admissibility check.

Written 2026-08-24, before any fusion score has been computed, inspected or interpreted.

## 0. What motivated this, stated plainly

This experiment did not arise from a hypothesis about fusion in the abstract. It arose from two
completed analyses whose results are already known, and saying so is part of registering it honestly.

1. **A fixed 50:50 rank ensemble of the learned token weighting and the probe improved on the probe**
   on the out-of-distribution means of both corrected-span populations, though the per-dataset
   evidence was weak. That established *complementarity*, but the estimator is **not deployable**: its
   ranks are computed relative to the whole evaluation cohort, so it cannot score a single new
   response. It is a diagnostic, not a method.
2. **The published hard back-off, with its probability branch replaced, showed the opposite limit.**
   Its gate hands essentially all weight to one branch under shift: mean supervised weight falls from
   0.262 at the matched setting to exactly 0.000 on the most shifted rung. The combination stops
   combining, and each estimator collapses to a single branch.

So the open question is neither "do these two signals differ" nor "can a gate choose between them",
both of which are answered. It is:

> Can the complementary signal be used by an estimator that scores one response at a time, with no
> access to the target distribution, no target labels and no evaluation-cohort ranking?

## 1. The estimator

Both component scores are oriented so that **a larger value means greater uncertainty**. Verified
empirically before registering: on both populations the rank correlation between each score and
correctness is negative on every matched-setting cell tested (learned weighting -0.45, probe -0.53
mean). **No reversal is applied**, and none is needed. If a future population disagreed, the reversal
would be applied once, before calibration, and recorded.

For each cell and seed, and each method `m` in `{learned weighting, probe}`, let `s_m^src` be that
method's scores on the **source reference population** defined in section 2. For a response with raw
score `s_m(x)`:

```
q_m(x) = ( #{i : s_i < s_m(x)} + 0.5 * #{i : s_i = s_m(x)} + 0.5 ) / (n + 1)
```

a mid-rank empirical distribution function that is deterministic, handles ties, and never returns
exactly 0 or 1. **This definition is frozen here.** No alternative transform will be tried afterwards.

```
fusion(x) = 0.5 * q_learned(x) + 0.5 * q_probe(x)
```

Computable for one response from: the trained weighting model, the trained probe, and the two frozen
source reference distributions. **Nothing about any other target example enters.**

## 2. The calibration population, audited before implementation

The canonical cell construction was read rather than assumed. `build_rows` carves the evaluation
target with `eval_split(..., seed=CARVE_SEED)` into **train and test only**, and draws
out-of-distribution sources with `sampled_train_idx`. **There is no third partition anywhere in the
pipeline**, so no held-out source calibration cohort exists to be preferred.

The reference distribution is therefore **the source training cohort of that cell and seed** — exactly
the rows the two models were fitted on, and nothing else.

**This is in-sample, and that is a real limitation, recorded here rather than discovered later.**
The probe is a high-dimensional model on roughly 1800 rows and fits its training set almost perfectly,
so its source scores are far better separated than its target scores will be. The consequence is
specific and must be reported: **a fixed 0.5/0.5 weighting on the percentile scale is equal weighting
only if the two methods' realised target percentiles have comparable spread.** If one method's target
scores land in a narrow band of its source distribution, that method contributes less, and the
"50:50" is nominal. The realised spread of each method's target percentiles will be reported for every
cell. **It will not be used to adjust the weights.**

No target test score, target label, pooled multi-dataset score, full evaluation cohort, or any future
example may enter the reference distribution. A run that cannot construct the source reference
aborts rather than substituting another population.

## 3. What is fixed and may not be varied

- The weight is **0.5 / 0.5**. If it does not win, no other weighting is tried.
- No target-specific, rung-specific or dataset-specific calibration.
- No alternative calibration transform after seeing results: no z-scoring, no temperature, no
  isotonic fitting, no min/max/product variants.
- No learned fusion of any kind: no stacking, no gate, no regression on the two scores.
- No target labels, at any point.
- Neither component implementation is modified. The frozen weighting at shrinkage 2 and the frozen
  probe are used as they are.
- The existing cohort-rank ensemble is **not** renamed, moved or overwritten. It keeps its name and
  its meaning as a non-deployable diagnostic. The new estimator gets its own name and namespace.

## 4. Primary comparison, registered

**The primary comparison is `fusion - probe`.** Reported per rung with the dataset as the unit,
n = 8: macro prediction-rejection ratio, per-dataset values, the paired difference, the sign count,
the exact Wilcoxon signed-rank p-value and a bootstrap interval over datasets, following the procedure
already used elsewhere in this project.

Also reported, secondary: `fusion - learned weighting`, and `fusion - cohort-rank ensemble`. The last
answers how much of the complementarity survives being made deployable. From it:

```
retained gain = (fusion - probe) / (cohort-rank ensemble - probe)
```

computed only where the denominator is positive and not near zero, and treated as **descriptive**.
It is not a registered significance test and no claim rests on it alone.

Seeds are averaged as per-seed prediction-rejection ratios, never by scoring a seed-averaged vector.
A constant score is recorded blank with an explicit degeneracy count, never converted into a number by
row-order tie breaking.

## 5. Implementation gates, all before any grid

- **A. No target dependence.** A single evaluation example scored alone must receive exactly the same
  fusion score as when scored inside the full target batch. This is the property the cohort-rank
  ensemble fails, and it is the whole point of the experiment.
- **B. Frozen reference.** Reordering or shuffling the target rows must leave every fusion score
  exactly unchanged.
- **C. Monotonicity.** With one component held fixed, increasing the other component's oriented score
  must never decrease its percentile.
- **D. Orientation.** On a constructed case that is more uncertain under both components, the fused
  score must rank it as more uncertain.
- **E. Component fidelity.** The target-side component scores recomputed here must equal the canonical
  ladder's own per-example vectors for the same cell and seed. This proves the frozen models are being
  used rather than reproductions of them, and it is the gate that would catch a population mismatch.

Failure of any gate aborts rather than producing a corrected number.

## 6. Populations

The corrected-span eight-dataset core. `meta-llama/Meta-Llama-3.1-8B` and `google/gemma-2-9b` first,
both having the required frozen artifacts. `Qwen/Qwen2.5-14B` **only** if its corrected-span
per-example population passes the separate seven-criterion certification already under way; the known
uncorrected directory must not be used. If that certification fails, the result is reported on two
populations, and the absence of the third is stated rather than glossed.

No population is pooled with another. The cross-model summary averages the eight datasets within a
model first, then averages model-level means equally.

## 7. Interpretation, fixed before results

- **A.** Consistent improvement on the probe across models, particularly on the two cross-task rungs,
  without a substantial matched-setting penalty: the complementary signal converts into a deployable
  gain, and may be reported as a main result.
- **B.** Improvement on some models or settings but not consistently: partial complementarity.
  Reported as such. **No weight search to recover a universal win.**
- **C.** Roughly matches the probe while losing most of the cohort-rank gain: the earlier ensemble
  established complementarity, and simple source-only calibration does not convert it into a
  deployable improvement. **This is a useful result and will be reported as one.**
- **D.** Consistently worse than the probe: **stop.** No alternative weights, gates, stacking or
  calibration functions. The cohort-rank result stays what it is, an analysis of complementarity
  rather than a proposed method.

The sentence this experiment is allowed to license, and only if the evidence supports it:

> The learned token weighting carries uncertainty information complementary to a direct hidden-state
> probe, and a fixed source-calibrated combination of the two improves robustness without access to
> the target evaluation distribution.

The experiment decides whether that may be said. The rules above are not to be weakened to make it
true.

---

## AMENDMENT 1 — 2026-08-24, before any fusion score existed

Recorded as an appended amendment rather than an edit to the body above, so the chronology stays
legible. Two wording corrections and one **tightening** of a gate. Nothing here loosens a rule, and no
fusion number had been computed when it was written.

### A1.1 How the prior ensemble result is described

Section 0 point 1 says the earlier rank ensemble "established *complementarity*". That overstates it.
The gains are **directional and replicated across two populations**, not statistically established:
the per-dataset evidence was weak on both. Throughout this registration and every report-facing
sentence derived from it, the correct phrasing is:

> the earlier rank ensemble provides **replicated evidence of complementarity**

It must not be written as though complementarity were an established finding.

### A1.2 What the new estimator may be called

The property this experiment tests, and the only one it can license, is **target independence**: the
score for one response does not depend on any other target example. Gates A and B test exactly that.

- Use **target-independent** or **deployment-compatible**.
- Do **not** use *production-ready*. That would assert latency, robustness and monitoring evidence
  this project does not have and does not collect.
- The earlier estimator is described as **target-cohort-dependent**, not merely "not deployable" —
  naming the actual dependency is more informative than naming its absence.

### A1.3 Gate E is strengthened, and supersedes the version in section 5

Section 5's gate E asked that recomputed target scores "match the sidecar". That is too weak, because
two different score vectors can produce the same prediction-rejection ratio. The gate is now:

> **Row-level raw-score fidelity.** For the same cell, the same seed and the same **row ids**, the
> retrained raw learned-weighting and probe vectors are compared **element by element** against the
> frozen corrected-span sidecar vectors. Reported per cell and seed: **maximum and mean absolute
> difference**, and **ranking agreement** (Spearman correlation and the fraction of discordant pairs).

Agreement of prediction-rejection ratio is **not** accepted as evidence for this gate, in place of it,
or as a fallback when the row-level comparison is inconvenient.

**On failure: stop and report.** Do not widen the tolerance to fit the observed difference, do not
substitute an approximate agreement, and do not argue that the source-side scores are unaffected. A
retrained model that does not reproduce the frozen model row for row **is not the frozen model**, so
the source reference distribution it produces is not the frozen model's reference distribution either,
and the whole estimator would be built on something other than the scores the report quotes.

### A1.4 The percentile-spread diagnostics are descriptive only

Section 2 requires the realised spread of each method's target percentiles to be reported, because a
fixed 0.5/0.5 on the percentile scale is equal weighting only if the two spreads are comparable.

To remove any ambiguity: **those diagnostics explain the result and may not change it.** They are not
grounds for adjusting the weights, the calibration transform, or the reference population, either
before or after seeing the fusion scores.
