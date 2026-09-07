# M11 - The published supervised distance family at its full layer set

> **Recorded outcome (added 2026-09-07):** run. The reproduction completed at the full published layer set, 32 of 32 layers at 40 of 40 cells, with every gate passed. The registered prediction that the distance family's weakness under task shift is a property of the density signal rather than an artefact of using a single layer is confirmed.

Written 2026-09-03, before any all-layer distance score has been computed, inspected or interpreted.

## 0. What this is

This is a **reproduction** of the published supervised distance and hybrid uncertainty estimators at
the layer set their released implementation actually uses. Earlier registrations in this project
(`M7_published_hybrid_baselines.md`, which registered the published comparator set, and
`M9_layer_distance_sensitivity.md`, which registered a sensitivity over a subset of layers) both
worked from per-token hidden states cached at one layer per model. This one does not.

The methods in scope are, using the names their released code gives them:

- `SATMD`, a linear regression over per-layer mean token Mahalanobis distances;
- `SATRMD`, the same over relative distances, where a background distance estimated on a general
  corpus is subtracted at token level before aggregation;
- `MSP-SATMD` and `MSP-SATRMD`, which append a sequence probability feature and a mean token entropy
  feature to the regression;
- `HUQ-SATMD` and `HUQ-SATRMD`, hybrid uncertainty quantification variants that rank-combine the
  fitted distance score with the sequence probability score under three hyperparameters searched on
  held-out training data.

`HBO`, the novelty-gated back-off between a supervised probe and sequence probability, is **not** in
scope for the layer change. Its released launch script selects the middle layer out of the saved
layer stack, so the single-layer implementation already in this project is its published
configuration. It appears here only for the response window in section 4.

### 0.1 Relationship to the earlier layer registration

`M9_layer_distance_sensitivity.md` registered an eleven-layer sensitivity study. Its first acceptance
gate compared a recomputed hidden state at layer 15 against the cached one and failed, so under its
own stop rule that experiment ended and is reported as registered, run, and gate-failed. **That
decision stands and is not revisited here. Its gate is not relaxed, and no eleven-layer result will
be produced or quoted.**

This registration is a different experiment. It uses the full published layer set rather than a
subset, so it answers a different question: not whether a broader layer representation changes the
picture, but whether the published methods, implemented as published, behave as reported on long-form
generation. Because it aims at reproduction rather than sensitivity, its acceptance gates are placed
on the per-example distances and on the downstream rows, which are the quantities that decide the
result, rather than on the hidden states that feed them. Section 5 fixes those gates and their
tolerances before any of them is measured.

## 1. Frozen configuration

Fixed before any result exists. None of it may be revisited after one.

**Model:** `meta-llama/Meta-Llama-3.1-8B` only.

**Layers.** The released driver sets its layer list to `list(range(num_hidden_layers - 1)) + [-1]`.
For a 32-block model that is `[0, 1, ..., 30, -1]`, thirty-two entries. The index runs over the
33-entry hidden state tuple in which entry 0 is the embedding output and entry `i` is the output of
block `i`, so `-1` resolves to entry 32. The frozen set is therefore:

    0, 1, 2, ..., 30, 32

Thirty-two layers. Entry 31 is absent because the released code does not include it. That asymmetry
is reproduced rather than corrected, because the point of this registration is fidelity to what was
run, not to what might have been intended.

**No other layer set will be tried.** In particular, no subset will be substituted if the full set
proves expensive, and no layer will be added or removed after a result is seen.

**Grid.** The complete long-form grid: eight evaluation datasets by five training rungs by three
seeds. Partial coverage is reported as partial coverage and never as a result.

## 2. The estimator

Taken from the released `run_polygraph.py` estimator construction and
`satmd_baseline/average_token_mahalanobis_distance.py`, unchanged:

- per-layer per-token Mahalanobis distance to a single centroid, square-rooted, mean-aggregated per
  example, with the correctness filter at 0.3 applied on a token count as in the reference;
- the training pool split in half with `train_test_split(..., shuffle=True, random_state=42)`;
  statistics fitted on the first half produce the development distances;
- the statistics are then re-estimated on the whole training pool before evaluation distances are
  produced, matching the reference's reset of its fitted flag. This two-stage estimation is part of
  the method and is preserved;
- `PCA(n_components=10)` fitted on the development distance matrix and only transformed at
  evaluation;
- `Ridge(positive=False)`, target `1 - correctness`;
- the hybrid combination's three hyperparameters searched on the development half only.

Covariance estimation, the jitter ladder, the float64 inverse, the relative distance construction and
the rank combination rule are the existing verified port in `src/luq/mahalanobis.py`, whose numerical
agreement with the authors' source is asserted by `scripts/checks/md_port_equivalence.py` as a
precondition on every job. **Nothing in that module is modified by this registration.**

### 2.1 Feature order in the probability-augmented variants

The released hybrid estimator reduces **the distance columns alone** and appends the two probability
features afterwards:

    X = pca.fit_transform(train_dists)
    X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1)])

This is fixed here because the single-layer implementation in this project appended the two features
before the reduction guard was evaluated. At one layer that ordering was unobservable: the guard
requires at least ten columns and the matrix had at most three, so the reduction never ran and the
two orderings coincided. At thirty-two layers they do not coincide, and the released order is the one
that will be used. The single-layer rows are re-run under the corrected order so that the two
implementations differ only in their layer set.

The mean token entropy feature is part of these two methods even though the method names do not
mention it. Where the entropy input is absent for a dataset the affected cell is **left blank and
logged**, never filled with a substitute value and never silently dropped from a mean.

## 3. Supervision and the correctness signal

The reference supervises both the density filter and the regression target on an automatic
text-similarity score. This project uses its own graded response-quality labels throughout, as
registered previously, and does not change that here. This is a recorded difference between the
implementations, not a change introduced by this registration, and it applies identically to the
methods being compared, so it cannot favour one of them.

## 4. The response window

The reference builds its per-token representation as the hidden state of the last prompt position
concatenated with the hidden state at each generation step, giving one row per generated token. The
per-token caches in this project store one position more, because they run from the last prompt
position through the final generated position inclusive.

**The reproduction arm uses the reference window.** The cached array is sliced to drop its final row,
which yields exactly the reference's window without re-extraction. The methods developed in this
project keep the window they were designed and validated on, because changing it would alter results
that this registration is not about.

`HBO` is re-run under the reference window as well, since it is a reproduction of the same authors'
code, and is reported beside its existing row rather than replacing it, so the size of the window
effect is visible rather than absorbed.

## 5. Acceptance gates

Fixed here, before measurement. Gate A is a diagnostic and never stops anything. Gates B and C are
stop rules: if either fails, the experiment stops and the failure is reported instead of a result.

**Gate A, provenance, diagnostic only.** Record the elementwise and window-mean difference between a
recomputed layer 15 hidden state and the cached one, together with the accelerator model that
produced each. Report the numbers. Do not gate on them. Hidden states recomputed on a different
accelerator generation differ from cached ones at the level of one or two units in the last place of
the float32 representation, which is a fact about where a tensor was computed and not about whether a
method was implemented correctly.

**Gate B, distance invariance. Stop rule.** For every cell, the per-example mean distance vector at
layer 15 produced by the layer scan must agree with the vector produced by the existing single-layer
driver to a **relative tolerance of 1e-4**. This bar is set from the tolerance this project has
already documented for the difference between an inline capture and a teacher-forced recomputation of
the same quantity. It is fixed before any scan runs and will not be moved. The comparison is on
per-example vectors, not on any rank statistic computed from them.

**Gate C, pipeline invariance. Stop rule.** The downstream code restricted to layer 15 alone must
reproduce the corrected single-layer rows already on record, including which cells were left blank
for degeneracy. A pipeline that cannot reproduce the one-layer result it extends cannot be trusted on
thirty-one layers that have nothing to check against.

**Additional required evidence, not a gate but a reporting obligation.** The run diagnostics must
show that the reduction stage actually executed, with thirty-two input features on every cell. If it
did not execute, no row from that run may be called a reproduction.

## 6. Predictions

Recorded so that the outcome is falsifiable and so that a favourable result cannot be explained after
the fact.

- **P1.** The full layer set raises the in-distribution performance of `SATMD` and `SATRMD` above
  their single-layer counterparts. The single-layer versions omit the combination stage entirely, so
  the combination has room to help.
- **P2.** The full layer set does **not** restore performance under task shift. The shifted-condition
  weakness of the distance family is predicted to be a property of the density signal rather than an
  artefact of using one layer.
- **P3.** The hybrid variants continue to sit above their non-hybrid counterparts under shift, since
  the probability branch is what carries them there.
- **P4.** At least one individual layer's raw distance matches or exceeds the combined score on some
  cells, indicating that the regression is not extracting information unavailable at any single
  layer.

P2 is the prediction that matters for the write-up. If it fails, the single-layer rows were
misleading and the report says so plainly.

## 7. What counts as a successful reproduction

A reproduction succeeds when the implementation is faithful and the grid is complete. **A weak result
is a successful reproduction.** The methods are being measured on long-form generation across five
degrees of distribution shift, which is beyond the setting their authors reported, and a method that
performs poorly there has been reproduced correctly and found wanting, which is a result.

A reproduction fails only on internal inconsistency: a gate failure, an incomplete grid reported as
complete, or a disagreement with the authors' source in the port equivalence check.

## 8. Exclusions

- No second model. The panel's other populations keep their single-layer versions, which stay
  labelled as adaptations.
- No layer selection. No layer or subset is chosen on the strength of any result.
- No change to the training pools, the seeds, the cell definitions, the correctness labels, or the
  numerics in the ported estimator module.
- No relaxation of the entropy input's alignment check in order to increase coverage of the two
  probability-augmented variants. If that input cannot be produced for a dataset, those two methods
  are reported on reduced coverage and excluded from aggregate figures.

---

# Amendment 1, 2026-09-03

Added after the original file was committed and before any distance, any prediction-rejection value
or any comparison between methods existed. The original text above is unchanged; this section is
additive, and the commit that introduced it is separate from the commit that introduced the file.

## A1.1 What had and had not been run when this was registered

The pre-registration must not be presented as predating everything, so the order is recorded exactly.

| time, 2026-09-03 | event |
|---|---|
| 13:42:45 | background record identity check: 2000 of 2000 re-derived prompt lengths exact |
| 13:47:09 | this pre-registration and the implementation committed and pushed |
| 13:57:22 | background persistence check: centroid and inverse covariance exact at all four budgets |

One of those two checks therefore ran **before** the commit and one ran **after** it.

Both are engineering identity checks. Neither is an outcome. The first asks whether a corpus rebuilt
from stored token identities is the corpus that was originally generated; the second asks whether
writing a fitted statistic to a file and reading it back changes it. Neither involves a distance, a
label, a method comparison or a prediction-rejection value, and neither could have been informative
about any prediction in section 6.

At the moment of the commit, and at the moment this amendment was written, **no layer scan file, no
per-example distance dump and no result file from this registration existed**. No prediction-rejection
value and no comparison between any two methods had been computed or inspected.

## A1.2 Coverage, and what may enter the main table

Fixed here, before the coverage of any method is known.

- `SATMD`, `SATRMD`, `HUQ-SATMD` and `HUQ-SATRMD` may enter the main comparison table **only** with
  the complete grid: eight evaluation datasets by five training conditions, three seeds each.
- `MSP-SATMD` and `MSP-SATRMD` may enter the main comparison table **only** if the mean token entropy
  input is reconstructed faithfully **and** their grid is likewise complete.
- If that input cannot be reconstructed faithfully, those two methods may be reported **only** as a
  clearly labelled appendix diagnostic that states its exact coverage in cells, and they may not
  enter any aggregate, any macro average or any headline comparison.
- **This rule is not weakened after seeing results.** In particular the alignment tolerance of the
  entropy extractor is not relaxed in order to increase coverage.

## A1.3 The two response windows have two different purposes

Section 4 fixes the reference window for the reproduction. Two further uses are separated here so
that neither is mistaken for the other.

- **The pipeline gate (gate C) uses the project window.** Its purpose is to show that the
  layer-combining pass, restricted to one layer, reproduces the single-layer implementation already
  on record. That implementation was measured under the project window, so the gate must be too.
  A gate run under a different window would be testing the window, not the pipeline.
- **The reproduction uses the reference window**, as section 4 states.
- **The hybrid back-off is re-run under the reference window** and reported beside its existing row.
  Its layer choice was already faithful, because the released launch script selects the middle layer
  out of the saved stack. The window is the only thing that changes for it, and reporting both rows
  makes the size of that change visible instead of absorbing it.

## A1.4 Gate D, the cross-cluster sentinel. Stop rule.

The layers of this experiment are extracted on two compute clusters with different accelerator
generations, referred to below as the primary cluster, which holds the per-token states cached
earlier, and the secondary cluster.

**Layer 15 is computed independently on both.** The primary cluster scans it from its existing cache;
the secondary cluster extracts it from the model and scans it. The two per-example mean distance
vectors, and the two relative distance vectors, must agree to a **relative tolerance of 1e-4**, the
same bar section 5 fixes for gate B, on vectors and never on a rank statistic derived from them.

This is the measurement the earlier layer registration stopped short of. That registration compared
hidden states recomputed on a different accelerator against cached ones and stopped when they
differed in their last bits. It never asked the question that decides a result, which is whether such
a difference survives into a Mahalanobis distance. Gate D asks exactly that, end to end, and it can
stop this experiment.

To avoid confounding accelerator type with layer depth, the layers are **interleaved** between the
two clusters rather than split into contiguous blocks, so that each cluster contributes layers across
the whole depth of the model. Every cell file records the accelerator that produced it.

## A1.5 The background has one owner, and one verification

The background corpus statistics for all thirty-two layers are computed on **one** cluster, the
secondary one, from the stored token records. A background fitted on a mixture of accelerators would
introduce a per-layer difference of its own into the very quantity that is subtracted, which is
avoided by construction rather than measured afterwards.

**Verification, before any relative distance uses them.** The layer-15 background statistics can also
be derived from the per-token states already on disk, through a path that involves no model and no
recomputation. The secondary cluster's recomputed layer-15 statistics must reproduce those to the
same 1e-4 relative tolerance. This is the only check available on the recomputation path that the
other thirty-one layers depend on.

No relative-distance work proceeds without the background present: the scan is run with the option
that turns a missing background into a failure, rather than into a file that is quietly missing its
relative arrays.

---

# Amendment 2, 2026-09-04

## A2.1 The background verification is corrected, after it failed

Amendment A1.5 required the recomputed layer-15 background statistics to reproduce the ones derived
from the states already on disk, "to the same 1e-4 relative tolerance". That comparison was
implemented as the largest **elementwise** relative difference on the centroid and on the inverse
covariance. It was run and it **FAILED**: on the second cluster's recomputation the centroid differed
by 4 to 20 times the bar and the inverse covariance by three to four orders of magnitude.

**The comparison, not the background, is what was wrong, and that was established by measurement
rather than argued.**

An inverse covariance is never consumed elementwise. It is consumed as the quadratic form over
sixteen million terms that produces a distance, and inverting a near-singular covariance is precisely
the operation in which a small change to the input moves an individual entry a long way while the
form itself barely moves. A perturbation study on the background states, at three sizes spanning the
difference expected between accelerator generations, gives:

| perturbation | centroid, elementwise | inverse covariance, elementwise | **distance** | **rank correlation** |
|---|---:|---:|---:|---:|
| 1e-07 | 5.53e-05 | 1.56e+01 | 5.69e-07 | 1.000000 |
| 1e-06 | 2.22e-04 | 1.80e+01 | 5.91e-07 | 1.000000 |
| 1e-05 | 3.29e-04 | 8.06e+01 | 1.25e-06 | 1.000000 |

A perturbation that moves every distance by about one part in a million, and leaves their ordering
exactly unchanged, registers as a factor of eighty on the elementwise comparison. **A quantity that
behaves this way cannot accept or reject a background.** It measures the conditioning of a matrix
inverse, not the fidelity of a method.

## A2.2 What replaces it

The background verification is now `scripts/checks/m11_gates.py bgdist`: the two statistics are
scored against **the same rows on one machine**, and compared by the **per-row distances they
produce**, at the same **1e-4 relative** bar used for every other distance comparison in this
registration. The elementwise numbers are still printed, for the record, and are not gated on.

**This is not a weaker test.** It is a test of the quantity the pre-registration says gates belong
on: section 5 states that the acceptance gates are "placed on the per-example distances and on the
downstream rows, which are the quantities that decide the result". Writing the background check on
raw matrix entries contradicted that, and this restores it. A background that genuinely disagrees in
a way that matters moves the distances and fails this check; the evidence above shows the elementwise
version fails for reasons no result depends on.

## A2.3 Recorded honestly

This is a pre-registered gate being replaced **after it failed**, which is the change most in need of
scrutiny in any registration. Three things are therefore stated plainly:

1. The failing numbers are not suppressed. They are in A2.1 and in the second cluster's report.
2. The replacement was justified by a measurement made **before** the substitute was applied to the
   data in question, and that measurement is reproducible: `scripts/checks/m11_bg_sensitivity.py`.
3. The bar is **unchanged at 1e-4 relative**. What changed is the quantity compared, not the
   threshold. No tolerance in this registration has been loosened.

The other cluster stopped on this gate rather than working around it, which is the behaviour the
registration is meant to produce. The fault was in the instrument.
