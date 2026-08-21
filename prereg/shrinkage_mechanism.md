# Pre-registration — shrinkage mechanism diagnostic and configuration-selection audit

**Date:** 2026-08-13. **Population:** `meta-llama/Meta-Llama-3.1-8B`, canonical ProbeDriftLong,
layer 15, carve `legacy`. **Written and committed BEFORE any arm-vs-arm mechanism output was
inspected.**

## 0. Honesty preamble — what this is and is not

- This is **not a new method search**. No new λ is created. No configuration is proposed after this.
- The complete Llama master, the Qwen master and every closed wMSP variant were inspected long
  before this document. This is a **mechanism diagnostic on an already-observed population**, plus a
  **retrospective** configuration-selection sensitivity. It is not prospective confirmation.
- **All performance numbers remain the existing 3-seed ladder results.** Nothing here re-scores a
  method. Any diagnostic computed at one seed is labelled **mechanism-only** and may not be quoted
  as a performance figure.
- Part A (the algebraic decomposition and its numerical check) was completed before this file was
  written. That is deliberate and harmless: it is an identity that holds for *any* weights and
  contains no arm-vs-arm comparison. The quantities in §2 below had been *computed* but **not
  inspected** when this was written.

## 1. The hypothesis being tested

Not "regularisation reduces overfitting". Specifically:

> wMSP is a **fixed content-token mean-NLL anchor plus a hidden-state-derived correction**. Shrinkage
> penalises the magnitude of the weight deviation that generates that correction. If this explains
> the out-of-distribution improvement, moderate shrinkage should **reduce the realised correction
> and weight dispersion while retaining useful ranking signal**.

## 2. Pre-specified per-response measures

Over the content-token set `C` actually used by wMSP, `n = |C|`, `delta_t = w_t - 1`:

| quantity | definition |
|---|---|
| `mu_C` | `(1/n) sum_{t in C} nll_t` — the anchor |
| `U_wmsp` | the production score `(1/n) sum_t w_t nll_t` |
| `C_learned` | `U_wmsp - mu_C` = `(1/n) sum_{t in C} delta_t (nll_t - mu_C)` |
| `abs(C_learned)` | correction magnitude |
| `Omega_C` | `(1/n) sum_{t in C} delta_t^2` — dispersion over content tokens |
| `Omega_impl` | `((w-1)^2).mean()` over **all G** positions — what `shrink_to_uniform` actually returns (see §5) |
| `sigma_nll` | `sqrt((1/n) sum_{t in C} (nll_t - mu_C)^2)` |
| `bound` | `sqrt(Omega_C) * sigma_nll` (Cauchy–Schwarz upper bound on `abs(C_learned)`) |
| `alignment` | `C_learned / bound`, in [-1, 1]; reported only where `bound > 0` |

## 3. Pre-specified reporting

For every dataset and rung available: mean and median `abs(C_learned)`; mean and median `Omega_C`;
mean `sigma_nll`; mean `bound`; mean `alignment`; and the existing 3-seed PRR. Then the change from
the unregularised arm to λ=1.5 and λ=2.

**Arms:** λ ∈ {0, 1, 1.5, 2, 10}, existing configurations only. **The principal report-facing
comparison is unregularised vs moderate shrinkage**, not "which of five λ wins".

## 4. Pre-specified dataset-level tests — fixed before reading any output

Unit of analysis: **the dataset, n = 8**. Spearman, two-sided, descriptive at n = 8.

1. Spearman(unregularised mean `Omega_C`, OOD PRR gain from shrinkage).
2. Spearman(unregularised mean `abs(C_learned)`, OOD PRR gain from shrinkage).
3. Spearman(PRR of the content-token mean-NLL anchor, OOD PRR gain from shrinkage).
4. Reproduce the existing descriptive relation between anchor quality and preferred shrinkage
   regime. Any per-dataset best-λ quantity is labelled **TEST-SELECTED / MECHANISM-ONLY**.

**No other correlation will be computed after seeing these.** If a further relation is examined it
will be reported as post-hoc and excluded from any claim.

## 5. Declared implementation discrepancy (found in Part A, before any arm comparison)

`shrink_to_uniform(w) = ((w - 1.0)**2).mean()` is taken over **all G generated positions**, not over
the `n` content tokens. Excluded special tokens carry `w_t = 0` exactly, each contributing
`(0-1)^2 = 1`. Therefore

    Omega_impl = (1/G) [ sum_{t in C} (w_t - 1)^2 + (G - n) ]
               = (n/G) * Omega_C + (G - n)/G

The `(G-n)/G` term is **constant in the model parameters** (excluded weights are identically zero
regardless of the MLP), so it contributes no gradient and does not move the optimum. The consequence
that does bite is the scale: the penalty actually optimised is `(n/G) * Omega_C`, so the **effective
shrinkage strength is `lambda * n/G`**, which varies slightly across responses with the
special-token fraction. Both `Omega_C` (the interpretable quantity) and `Omega_impl` (the optimised
quantity) are reported. This is a description of what the code does, not a defect claim.

## 6. Falsification conditions — stated before the outputs

The proposed mechanism is **NOT claimed** unless all four hold:

1. λ=1.5 / λ=2 **reduce** mean `Omega_C` relative to λ=0;
2. they **reduce** mean `abs(C_learned)`;
3. the reduction holds **both ID and OOD**;
4. the reduction **coexists with** the existing improved OOD PRR.

If any fails, the write-up says the mechanism is not supported and reports the failure. Directional
consistency is required; no significance threshold is set at n = 8, and none will be introduced
afterwards.

## 7. Compute discipline

Persisted weights exist for λ=0 and λ=10 at the **ID rung, seed 1 only** (`cache/viz/*__wmsp_weights.npz`).
Any arm or rung beyond that requires retraining. If retraining happens: **one seed, the same seed for
every arm**, labelled mechanism-only, and the cost is reported before launch. Canonical masters are
never modified and no result CSV is overwritten.

## 8. Population caveat

The persisted-weight population is the dump's `eval_split` over the **full** record list. On
`expertqa` and `factscore` this is a slightly larger set than the ladder's scored population, which
drops unlabelled rows before carving (605 vs 517, and 150 vs 136). Mechanism statistics on those two
datasets therefore describe a marginally different population from their PRR, and are labelled so.
