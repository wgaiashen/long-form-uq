# PRE-REGISTRATION — R4b: implement the HBO and VALIDATE it before using it as a baseline

**Written 2026-08-03, BEFORE the implementation is scored.** Registered as a standalone step on the
author's instruction (2026-08-02): *"if HBO is implemented AS R5's baseline, there is a pull toward
specifying it in whatever form makes our comparison favourable."* So it is built and validated on its
own, and only then used anywhere.

---

## 1. Why this exists

Our proposed system — a supervised probe backed off to an unsupervised estimate depending on how far a
test example sits from training — is **structurally the published HBO**. Discovering that after
building would be the bad outcome. HBO must therefore be a **reported baseline**, which means it must
first be implemented **faithfully** and shown to behave as published. If it does not reproduce, we have
implemented something else and every comparison against it is meaningless.

## 2. The method, quoted from the paper (§5.2), not from memory

Per-test-example OOD score via Mahalanobis distance, following Vazhentsev et al. (2025b):

1. Split the training data in half. Use the **first half** to compute a mean and covariance for the MD.
2. Compute the average MD for each of the **remaining** training examples.
3. Compare those to the MD of each **test** example, using a mean and covariance computed from the
   **entire** training set.
4. For each test example, `r` = the number of training examples with a smaller MD (rank of the test MD
   within the combined training MDs + that test MD).
5. `R = r / (N + 1)`, where `N` is the size of the training data.

Weights and combination (paper equations 2–4):

```
W_usv = R + 0.5   if R <= 0.5
        1         otherwise
W_sv  = 1 - W_usv
UQ_hyb = W_sv * UQ_sv + W_usv * UQ_usv
```

with `UQ_sv` = **SAPLMA (middle layer)** and `UQ_usv` = **MSP**, both **rank-normalised** before
combining.

Two consequences of the formula, recorded now so they are not mistaken for implementation bugs later:
- **W_sv never exceeds 0.5.** At `R = 0` (most in-distribution) the weights are 0.5/0.5, which is the
  "even weighting for in-distribution examples" the paper describes. HBO never leans supervised.
- **W_sv is exactly 0 for any test example with `R > 0.5`**, i.e. HBO collapses to pure MSP for the
  more-OOD half of the test set. This is why the paper's HBO row equals its MSP row on the far rungs.

## 3. REGISTERED VALIDATION CRITERIA — fixed before the numbers are seen

The paper's Table 4, Llama-3.1-8B, **short-form**, is the target:

| method | ID | LOO | 1D-SameTask | DiffTask | 1D-DiffTask |
|---|---|---|---|---|---|
| MSP | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 |
| SAPLMA (mid) | 0.63 | 0.46 | 0.30 | 0.35 | 0.32 |
| **HBO** | **0.66** | **0.61** | **0.57** | **0.57** | **0.57** |

**We must NOT expect the absolute numbers to match.** Their short-form population, splits, label
functions and training-set composition are not identical to our ProbeDrift-XL short-form grid, and PRR is
population-dependent. Requiring 0.66 would be requiring the wrong thing, and hitting it would be luck.

**What is registered instead is the QUALITATIVE BEHAVIOUR, which is what "faithful" means here.** All
four must hold on our short-form grid for the implementation to be declared valid:

- **V1 — HBO ≥ MSP at every rung** (within −0.01 tolerance for numerical noise). The paper's central
  claim is that HBO improves robustness *without* costing ID performance.
- **V2 — HBO ≥ SAPLMA on every OOD rung.** This is the entire point of backing off.
- **V3 — HBO converges to MSP as shift increases:** on the far rungs (DiffTask, 1D-DiffTask) HBO is
  within 0.02 of MSP. This is a **direct consequence of the formula** (W_sv = 0 whenever R > 0.5), so it
  is close to a unit test of the implementation rather than an empirical claim.
- **V4 — HBO ≥ both at ID**, matching the paper's "no loss of in-distribution performance".

**If V1–V4 hold:** HBO is validated and may be used as a named baseline.
**If any fails:** we have implemented something other than HBO. It is reported as a **failure to
reproduce**, and it is NOT quietly re-tuned until it passes. Re-tuning a baseline until it behaves is
how a baseline gets weakened, and this project has a standing rule against comparing to the most
convenient baseline.

**A diagnostic that must be reported whichever way it goes:** the distribution of `R` on each rung. If
`R > 0.5` for essentially every test example even at the **ID** rung, then HBO is pure MSP everywhere,
V1–V3 pass trivially, and the "validation" is vacuous. **Report `mean(R)` and `frac(R > 0.5)` per rung.**

## 4. Population

v1 generations, ProbeDrift-XL **short-form** evals, layer 15, SAPLMA features from `cache/features`
(teacher-forced), MSP from the cached token logprobs. Seed 1, the project's standard.
Output: `results/regime_R4b_hbo_validation__<slug>.csv`.

Long-form HBO is **explicitly out of scope here** — the paper says HBO does not generalise to long-form
and leaves it to future work, which is the gap our own system would occupy. Validating on short-form is
what tells us our implementation is theirs.
