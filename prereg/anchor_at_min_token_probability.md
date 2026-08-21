# Pre-registration — regularise activation-weighted surprisal toward the minimum token probability rather than mean token NLL

> **Status (2026-08-10):** the linear penalty stalled grid-wide (an optimisation failure, §16.1); the registered −log p[k] fallback ran 8/8 (F5b): mechanism confirmed both directions with the random-anchor control, method short of the floor, penalty bites on 3/8 evals only. Continued in `anchor_warm_start_combination.md`; F5 closed as mechanism-only.

**Written 2026-08-09, BEFORE the driver was implemented and before any cell was run.**
Population: the complete ProbeDriftLong grid, 8 long evals × 5 rungs, `meta-llama/Llama-3.1-8B`,
legacy carve, 3 seeds. **Headline numbers are the mean over the 4 OOD rungs, n = 8 datasets.**


---

## 1. The observation this rests on, measured not assumed

Every measured "how do we combine per-token signals" move on the OOD grid is worth about +0.02:

| move | Δ (OOD mean, 32 cells) |
|---|---|
| attention pooling vs mean-pool | +0.019 |
| NLL-as-pooler vs mean-pool | +0.011 |
| C2, NLL as an input feature, vs armA | +0.017 |
| NLL prior tilt vs plain attention | **−0.0005** |

**But regularisation is worth four to five times more:**

| move | Δ |
|---|---|
| **wMSP shrink@2 vs unregularised wMSP-norm** | **+0.090** |
| MLP vs linear head on mean-pool | +0.038 |
| SAPLMA vs `msp_min` (family gap) | +0.056 |

**And the existing regulariser is anchored on the WEAKER floor.**
`weighting.py:62`, `shrink_to_uniform(w) = ((w − 1)²).mean()`, pulls the weights to 1; and
`weighted_msp.py:522` states what `w = 1` *is*: `q = mean(nll) == msp 'perplexity'`. So wMSP is
shrunk toward **`perplexity` (+0.1169)** when **`msp_min` (+0.1855)** is the stronger floor and is
the project's pre-registered bar.

The λ trajectory is consistent with a bad anchor: **λ=0 → +0.139, λ=2 → +0.229, λ=10 → +0.208,
λ=∞ → +0.117.** It peaks near λ = 2 and then declines toward `perplexity`.

**F5 changes the anchor, not the pooling.** Every trained method in this project is structurally a
weighted MEAN (armA/armB/armC/armD/C2 pool hidden states; wMSP averages NLLs; all weights sum to a
constant). `msp_min` is a **MAX**. A mean-family method cannot inherit a max, which is exactly why
C2 "helps where the floor is strong but never closes the gap". Re-anchoring is the one move that
changes the limit.

## 2. THE PENALTY IS DEFINED IN PROBABILITY SPACE, AND WHY THE OBVIOUS FORM IS WRONG

The naive analogue of the existing penalty is squared error toward `w* = n · onehot(argmax nll)`.
**That is rejected before use.** At uniform weights it evaluates to `n − 1`, so the penalty scales
as **O(n)** — one λ would mean something ~7× different at `pubmed_qa` (median 32 tokens) and
`expertqa` (214). That is **exactly the length confound that killed the τ family** (round 1 §3.6:
`max z ≤ √(n−1)`), and it must not be reintroduced.

**REGISTERED PENALTY, in probability space:**

```
p        = softmax(raw) over the KEPT tokens          (so sum p = 1, length-independent)
k        = argmax(nll) among the KEPT tokens          (the anchor)
penalty  = 1 - p[k]
```

- **Bounded [0, 1]** regardless of length, so one λ means the same thing on every dataset.
- **Zero exactly at the anchor.**
- The anchor is `argmax(nll)` **among kept tokens only**. `_weights_from_raw` masks special tokens
  to `-inf` before the softmax, so an anchor on an excluded token could never be reached and the
  penalty could never reach 0.
- Since `_weights_from_raw` returns `w = softmax(masked_raw) · n_kept`, we have `p[k] = w[k]/n_kept`.

## 3. THREE CONTROLS, RUN BEFORE ANY λ SWEEP IS READ

**C1 — ANCHOR ENDPOINT IDENTITY (the claim).** Forcing `w = n_kept · onehot(k)` must give a PRR
**exactly equal to `msp_min`** on every cell. This is a scoring-side identity, independent of any
training, and it is what makes the λ → ∞ limit `msp_min` rather than `perplexity`. **Confirm
numerically before any sweep. If it fails, F5's entire rationale is false and the run stops.**

**C2 — NO-OP FIDELITY OF THE COPIED TRAINING LOOP.** `reg(w)` in `weighted_msp.train_weighted_msp`
receives only the weight vector, so it cannot see which token is the anchor, and `weighted_msp.py` is
on the Qwen port's edit list and is not to be modified. The loop is therefore **copied verbatim into
a standalone driver** with only the penalty call changed. **Run with `shrink_to_uniform` at the same
seed it must reproduce `train_weighted_msp` exactly.** A copied loop that has silently diverged would
make every F5 number incomparable with the master ladder.

**C3 — DOES THE PENALTY ACTUALLY BITE?** Registered because the bounded form has a known weakness:
`d(1 − p[k])/d raw ∝ p[k]`, and at initialisation `p[k] ≈ 1/n`, which is ~0.005 on `expertqa`. The
penalty may barely act at small λ. **Report mean `p[k]` per λ per cell.** If `p[k]` does not increase
monotonically with λ, the result is a **failure of the optimisation, not of the idea**, and must be
reported as such rather than as a null. In that case the fallback (`−log p[k]`, unbounded but with
O(1) gradients) is named here so that trying it later is not a post-hoc invention.

## 4. THE REGISTERED QUESTION AND BAR

**Q: does anchoring at `msp_min` beat anchoring at `perplexity` on the OOD mean?**

Primary comparison: **best LODO-selected λ under the new anchor vs `wMSP-shrink@2` (+0.2287)**, the
incumbent, on the per-dataset OOD mean, n = 8, with margin, sign count and Wilcoxon — the same
three-part bar as W1.

**λ has NO pre-committed value.** Unlike W5, no prior prediction exists for a new penalty, so this
arm is **exploratory** and λ is chosen by **leave-one-dataset-out**, reported as what the procedure
achieves and never as the best. The per-cell best is an ORACLE and is never a result.
Registered grid: `λ ∈ {0, 0.5, 1, 2, 5, 10, 20}` (the penalty is bounded, so the scale differs from
the unbounded uniform penalty's {2, 10}).

**Secondary, reported alongside:** the per-dataset breakdown against `msp_min` itself — the mechanism
predicts the gain should concentrate on the datasets where the floor wins far-OOD (`pubmed_qa`,
`factscore`), which is where a `perplexity` anchor is most wrong.

**Registered failure readings.** (a) If LODO-λ under the new anchor does not beat `shrink@2`, the
re-anchoring did not help and that is the result — the anchor is not the binding constraint. (b) If it
beats `shrink@2` **only** on `pubmed_qa`/`factscore`, report it as a regime-specific gain, not a
general one. (c) **G3 applies:** the per-cell λ oracle over the existing three λ has net headroom
**−0.0105, p = 0.986**, so per-cell λ selection is already known to be artefact. F5 must be judged on
a **fixed or LODO-selected** λ, never per cell.

## 5. PROVENANCE

Driver: `scripts/checks/anchor_msp_min.py` (new, standalone). Imports `weighted_msp` for
`per_token_nll`, `answer_states`, `content_keep`, `_seq_q`, `TokenWeightMLP` and reuses
`probedriftlong.cells_long` / `build_rows` / `xl_rungs.eval_split`, so the population matches the
master ladder cell for cell. **Edits no shared file.** `LUQ_CARVE=legacy`, stamped into every row.
Output `results/anchor_msp_min_<eval>__<slug>.csv`, one job per eval. RCS, CPU only, ~36 GB
(measured: every job in this workstream peaks at ~23 GB).
