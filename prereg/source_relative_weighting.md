# Pre-registration — source-relative rank supervision for activation-weighted surprisal

> **Recorded outcome:** run, under the condition its registration set.

**Date:** 2026-08-12. **Population:** `meta-llama/Meta-Llama-3.1-8B`, canonical ProbeDriftLong,
carve `legacy`, layer 15, seeds 1/2/3, 1800-row supervised pools.
**Committed BEFORE any source-relative PRR existed.**

**Status: CONDITIONAL.** This experiment runs only if PR1 (prompt-residual) is neutral or negative
against its gate. If PR1 passes, the remaining time goes to replicating PR1 on Qwen at DoC instead.

## 0. Honesty preamble

- Every wMSP result on this benchmark was inspected before this method was designed. Prospective
  test, **reused development population**, not independent confirmation.
- **The prior is poor and is stated up front.** Roughly twenty weighted-MSP variants have already
  closed negative on this population (adaptive Lehmer, SAR relevance weighting, Orgad masks,
  hard top-k, prior-guided attention, segment/sentence weighting, per-instance blends and routers,
  softmax sharpening τ, length-conditioned τ, rank weighting, power mean, fixed Lehmer β,
  pool-aware shrinkage, ensembles). A positive here would be the exception, not the expectation.
- The comparator is `wmsp_shrink2`, the registry's main regularised configuration. Two standing
  corrections are carried, not re-imported: the honest leave-one-dataset-out incumbent has moved to
  **λ = 1.5**, and *"wMSP-shrink@2 beats SAPLMA at the hardest OOD rung"* is **do-not-claim** — it
  was a tie in every population.

## 1. What the audit found (verified in code, `src/luq/weighted_msp.py`)

`train_weighted_msp` shuffles the whole mixed training pool (`:418`, `torch.randperm(n_seq)`), slices
batches of 32 (`:419-420`), and computes a soft rank over the batch with the full pairwise difference
matrix masked **only on the diagonal** (`_soft_rank`, `:121-123`, `mask = 1.0 - torch.eye(n)`). The
target ranks come from `_true_rank(incorrect[batch])` (`:444`), ranked globally within the batch.

**So every mixed-source rung trains on cross-dataset comparisons** — a 6-source `LOO-long` batch
averages ≈5.3 examples per source — while PRR is scored **within one dataset**. The training and
evaluation objectives are misaligned. No source label reaches the method; the driver has one
(`train_rows` is `list[(dataset, idx)]`, `splits.py:180`) and discards it.

**Two rungs are structurally inert.** `ID` (`[(X, None)]`) and `1ds-Diff-long` (one source at the
full budget) are single-source pools, where within-source ranking **is** global ranking. They are run
anyway, at seed 1, as an **exact-invariance control**: both arms must reproduce canonical to <1e-6.

## 2. The two arms

Everything not named below is byte-identical to canonical `wmsp_shrink2`: `TokenWeightMLP`
(d→256→128→64→1), generated-token states only (`answer_states = state[1:]`), `content_keep` special
exclusion, `softmax(raw) · n_kept` so weights average 1, `q = Σ w·nll / n_kept`, AdamW lr 1e-3,
5 epochs, batch 32, `torch.manual_seed(seed)`, and λ = 2 shrink-to-uniform.

**PRIMARY — `wmsp_srcrel_masked_shrink2` (clean isolation).** Batch membership, permutation, step
count, seed, optimiser and λ all unchanged. Only the rank term changes: computed separately within
each source subgroup of the batch, subgroups of fewer than 2 examples skipped, averaged over the
valid subgroups. **The shrink penalty stays averaged over the full batch**, as in canonical.

*Acknowledged power limitation:* ≈5.3 examples per source per batch, so the within-subgroup rank
targets are noisy. This is the price of changing exactly one thing.

**SECONDARY — `wmsp_srcrel_pure_shrink2` (better powered).** Source-pure batches of 32, drawn
round-robin across sources so no source dominates by size. (The XL rungs already cap each source
equally — `cap = XL_TOTAL // len(srcs)` — so the pools are near-balanced by construction; the
round-robin makes it exact rather than incidental.)

**λ is not tuned. Nor is lr, batch size, epochs, or any sampling temperature.**

## 3. Estimand and gate (fixed now)

**Unit of analysis: the dataset, n = 8.**

Scope is the **multi-source rungs only** — `SameTask-long`, `DiffTask-long`, `LOO-long` — because the
other two cannot differ. Cells where a rung is single-source for a given target (`SameTask` for
`xsum`, `factscore`, `expertqa`) are excluded from the scope and reported as inert, not as zero.

```
delta_e     = mean over in-scope rungs of [ srcrel(e) - wmsp_shrink2(e) ]
macro_delta = mean_e(delta_e)
```

**Promising if:** `macro_delta > +0.010` **and** ≥6/8 datasets positive **and** every
leave-one-dataset-out macro positive. Decision rule, not a significance claim.

**Reported regardless:** per-dataset deltas, per-rung breakdown, seed mean and sd per cell, exact
paired two-sided Wilcoxon p, dataset bootstrap CI, all LODO macros, and both arms side by side.

## 4. Interpretation, fixed before results

| outcome | conclusion |
|---|---|
| both arms improve | strong evidence for source-relative supervision |
| masked weak, pure improves | promising, but improved within-source rank resolution and changed batching may contribute — **say so** |
| masked negative, pure positive | **do NOT** claim that removing cross-source comparisons is responsible |
| both neutral or negative | close the direction. No tuning of lr, batch size, sampling temperature or λ to rescue it |

## 5. Mechanism — and an honest caveat about the motivating story

The stated motivation is that mixed-source ranking lets the weighter exploit **source identity**.
That story is **structurally weakened** by the method's own normalisation: weights are `softmax(raw)·n`
and so average 1 **within each sequence**, meaning the network can reweight tokens inside a response
but can barely shift that response's overall NLL level. A per-source calibration offset is therefore
largely *unlearnable*, which makes the shortcut second-order at best.

The honest alternative is that cross-source pairs contribute **gradient noise** rather than a learned
shortcut — they push the weighter toward something it structurally cannot do.

These make **different predictions**, and the diagnostics distinguish them:

| mechanism | prediction |
|---|---|
| shortcut removal | weight dispersion `Ω(w)` **falls**; weights transfer better across datasets |
| noise removal | training rank loss converges **lower/faster**, with `Ω(w)` little changed |

Recorded: training rank loss per epoch, `Ω(w) = mean((w−1)²)`, effective weight concentration, and
the correlation of learned weights with token NLL. These are diagnostics for interpreting the
primary result, **not** a new research programme.

## 6. Scope limit

If PR2 passes, the only follow-up considered is **worst-source aggregation** (`τ·logsumexp(L_d/τ)`
in place of `mean_d L_d`), at one fixed τ justified before results, λ unchanged, on the same cells.
Given the 13/14 Aug freeze it is expected to be dropped. If PR2 fails, the direction closes and no
variant is tried.
