# M11 - The published supervised distance family at its full layer set

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
