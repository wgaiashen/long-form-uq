# M9 — Layer sensitivity for the supervised distance family

> **Recorded outcome (added 2026-09-07):** run and stopped. Extraction completed at 88 of 88 layer files and the first acceptance gate failed, so under this registration no result was computed. This is registered, extracted and gate-stopped, which is not the same as not run.

Written 2026-08-24, before any multi-layer distance score has been computed, inspected or interpreted.

## 0. What this is, and what it is not

This is a **sensitivity study**, not a reproduction of the published distance methods. It is labelled
that way in every table it produces and in every sentence that quotes it.

The published supervised distance methods fit a Mahalanobis distance at every hidden layer and learn
a regression over the resulting per-layer distances. This project caches per-token hidden states at
one layer per model, so the versions measured so far combine nothing: with a single distance feature
the layer-combination stage has no layers to combine, and the published ten-component reduction is not
even defined. Those versions are middle-layer adaptations and are reported as such.

The single question here is:

> Does a broader layer representation rescue the supervised distance family enough that the
> single-layer result was misleading?

It is explicitly **not** a question about whether the distance family beats the learned token
weighting. A negative answer is as useful as a positive one, and neither changes what the
single-layer rows are called.

## 1. Frozen configuration

Fixed before any result exists. None of it may be revisited after one.

**Layers: `range(0, 32, 3)` = 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30.** Eleven layers, evenly spaced
by construction, for `meta-llama/Meta-Llama-3.1-8B` only.

Two properties motivated the choice, both fixed in advance:

1. **Eleven exceeds ten**, so the published `PCA(n_components=10)` is mathematically valid. At eight
   or fewer features it raises, which would have forced a second deviation on top of the layer
   reduction.
2. **The set contains layer 15**, this project's canonical middle layer, so the eleven-layer set is a
   strict superset of the single-layer adaptation and the two are nested rather than merely adjacent.

The same eleven layers are used for every cell and every seed. **No other subset will be tried.** If
this configuration gives an uninteresting answer, that is the answer.

## 2. The estimator

The published downstream recipe, unchanged, taken from `run_polygraph.py:360-380` and
`satmd_baseline/average_token_mahalanobis_distance.py`:

- per-layer per-token Mahalanobis distance, mean-aggregated per example, correctness filter at 0.3
  applied on a token count as in the reference;
- the training pool split 50/50 with `train_test_split(..., shuffle=True, random_state=42)`;
  statistics fitted on the first half produce the development distances;
- **the statistics are then re-estimated on the whole training pool before evaluation distances are
  produced**, matching the reference's reset of its fitted flag. This two-stage estimation is part of
  the method and is preserved;
- `PCA(n_components=10)` **fitted on the development matrix and only transformed at evaluation**;
- `Ridge(positive=False)`, target `1 - correctness`;
- the hybrid combination's three hyperparameters searched on the development half only.

Not used, because the reference does not use them: dimension reduction of the embeddings themselves,
reduced precision, approximated covariance, token subsampling, or any change to the training pools.

## 3. Predictions

Registered before the run.

- **P1.** The single-layer degeneracy will not reappear. With eleven features the regression has
  something to combine and a flat prediction should be rare or absent.
- **P2.** The eleven-layer version will beat the corrected single-layer version on the macro
  out-of-distribution mean. The direction is expected; the magnitude is not.
- **P3.** It will still not beat the probe under cross-task shift. If it does, the conclusion about
  the distance family on long-form generation changes and must be rewritten rather than defended.
- **P4.** The raw per-layer distances will be individually weak at every layer, so any gain comes
  from combination rather than from having picked a poor layer. If instead one layer is strong on its
  own, the middle-layer result was a layer-choice artefact and that must be reported plainly.

## 4. Acceptance gates

Two gates run before any eleven-layer number is interpreted. **If either fails, the experiment stops
and the failure is reported instead of a result.**

1. **Extraction gate.** The layer-15 output of the multi-layer extractor must reproduce the existing
   layer-15 cache exactly, or within the tolerance already established for these caches (about 1e-6
   on the window mean). A multi-layer extractor that does not reproduce the single-layer one cannot be
   trusted on the ten layers that have nothing to check against.
2. **Pipeline gate.** The new downstream code, restricted to layer 15 alone, must reproduce the
   corrected single-layer outputs, including their degeneracy decisions. This proves the difference
   between the two experiments is the layer set and nothing else.

## 5. Analysis

Unit of analysis is the dataset, n = 8, as everywhere in this project. Reported against the corrected
middle-layer adaptation, the published sequence probability, minimum token probability, the frozen
learned weighting and the probe:

macro prediction-rejection ratio by rung, the out-of-distribution mean, per-dataset values, degenerate
seed counts, the frequency of zero coefficients, the per-layer raw distance performance, and an
explicit statement of whether the qualitative conclusion changes.

Every row is labelled a layer sensitivity. No row is labelled a reproduction.
