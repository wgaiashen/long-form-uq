# PRE-REGISTRATION — W4: honest selection across the sharpening families, and a rank-weighted arm

> **Status (2026-08-09):** run. Lehmer under raw-argmax LODO is a consistent small positive (+0.027, 6/8) that misses its registered bar (p = 0.250); the rank arm removes the length confound and still fails on PRR. Per-dataset record and the corrected mechanism statistic: the project's working notes

**Written 2026-08-09, BEFORE either analysis was implemented or run.**
Population: `meta-llama/Llama-3.1-8B`, the 8 long ProbeDriftLong evals, legacy carve, judge label.
Free methods are rung-invariant, so **the unit is the DATASET, n = 8**, and every number is
equivalently the ID and the OOD value.

Predecessor: `W1_sharpening_axis.md` (round 1).

---

## 0. THE HONESTY POSITION OF THIS DOCUMENT, STATED FIRST

**Round 1 already read this test data.** `W1` swept three families over 24 grid points and reported
their per-dataset curves. So W4 is a **SECOND LOOK AT DATA ALREADY SEEN**, and that has two
consequences which are registered here rather than argued afterwards:

1. **A pass here is weaker evidence than a pass in round 1 would have been**, and it is **not** a
   claim until it replicates on a population this workstream has never touched. `Qwen2.5-14B` is the
   venue; it has not begun generating, so there is ample runway and no need to rush a decision.
2. **Nothing in W4 may be reported as "pre-registered" in the same sense W1 was.** W1 committed a
   parameter before seeing anything. W4 commits a *procedure* and a *bar* before running, on data
   whose neighbourhood is known. That is weaker and is to be described as such.

**Why it is still worth running.** Q-A asks about the output of a *procedure* (leave-one-dataset-out),
which is determined by the data and cannot be steered by what the author has already seen. Its answer
is genuinely undetermined: round 1 never computed it for two of the three families. Q-D introduces a
*new estimator* whose defining property (length-invariance) was chosen to fix a confound round 1
measured, not to fit a result.

---

## 1. Q-A — does an honest procedure select a useful sharpening parameter, in ANY family?

Round 1's A1 arm ran leave-one-dataset-out for **softmax-τ only**, and it returned `τ = ∞` on all 8
folds, i.e. it reproduced `msp_min` exactly. It was never run for **power-p** or **Lehmer-β**.

### 1.1 ALL THREE FAMILIES, NOT JUST LEHMER

Lehmer looks like the strongest of the three *because round 1 looked at the test means*. Running LODO
on Lehmer alone would therefore smuggle in a **family-selection step** chosen on test. All three
families are run. This costs nothing (the curves are already on disk) and it makes the family
comparison itself honest.

### 1.2 TWO SELECTION RULES, BOTH REPORTED

Round 1's one-standard-error rule proved **near-vacuous** at this sample size: between-dataset PRR
variance is so large that the 1-SE band covered most of the grid, and the registered tie-break
("nearest an endpoint") then decided the answer. That is a property of the rule, not of the data.

Both rules are therefore reported for every family:

- **raw-argmax LODO** — select the parameter with the best mean on the other 7, apply to the held-out one.
- **1-SE LODO** — round 1's rule, unchanged, so the two are comparable.

Reporting only one would let the choice of rule do the work.

### 1.3 REGISTERED BAR (identical to round 1, so it cannot be softened)

For each family, the LODO arm must beat `msp_min` (+0.1855) on the per-dataset mean with **all three**:

| part | threshold |
|---|---|
| margin | > **+0.010** |
| sign count | ≥ **6 of 8** |
| test | two-sided Wilcoxon signed-rank, **p < 0.05** |

**Also reported for every family, whatever the verdict:** the 8 selected parameters and their spread,
and the **transfer rate** — on how many of the 8 did the selected parameter beat **both** endpoints.

### 1.4 REGISTERED FAILURE READINGS

- If a family's LODO returns the endpoint on every fold (as softmax-τ did), that is recorded as
  **"the procedure declines to leave the endpoint"** — a null, not a small positive.
- If the raw-argmax and 1-SE rules disagree in verdict, **neither is reported as the answer**; the
  disagreement is the finding, and it says the selection is rule-dependent at n = 8.
- **Three families are tested, so a single p < 0.05 among them is roughly what chance produces.**
  A pass is only interesting if it is the family round 1's curves already favoured (Lehmer) **and**
  the two selection rules agree. This is stated now so a lone hit cannot be promoted later.

### 1.5 REGRESSION CHECK (must pass before any W4 number is read)

The new code path must reproduce round 1's softmax-τ A1 result **exactly**: `τ = ∞` on all 8 folds,
mean +0.1855, 0/8 against `msp_min`. If it does not, the refactor changed the procedure and nothing
below is interpretable.

---

## 2. Q-D — a rank-weighted family, length-invariant by construction

### 2.1 The problem it fixes, measured in round 1

Round 1's diagnostics showed the softmax-τ family behaves substantially as a **length rule**: `ESS`
correlates with answer length at ρ ≥ 0.85 on three of eight datasets. The cause is structural — the
largest attainable standardised value in an answer of `n` tokens is bounded by about `√(n−1)`, so at
a fixed τ a long answer can concentrate far more than a short one.

### 2.2 The registered estimator

```
r_t   = (rank of nll_t within this answer − 1) / (n − 1)      in [0, 1], largest NLL -> 1
w     = softmax(tau * r)
score = sum_t w_t * nll_t                                      scored with the RAW nll
```

- `tau = 0` → uniform → `perplexity` exactly.
- `tau -> inf` → one-hot on the largest NLL → `msp_min`'s ranking exactly.
- **`ESS/n` is constant in `n`** (for `r` equally spaced, `ESS/n → 2/tau` at large tau), so this is a
  soft **top-fraction** rule rather than a soft **top-count** rule. That is the confound removed by
  construction rather than by fitting.

### 2.3 THE GRID MUST BE MUCH LARGER, AND WHY

Because `ESS/n ≈ 2/tau`, reaching roughly one token in a hundred needs **tau ≈ 200**. Round 1's grid
(max 32) would barely move this family off uniform, and reusing it would manufacture a null.

**Registered grid:** `tau ∈ {0, 1, 2, 5, 10, 20, 50, 100, 200, inf}`.

### 2.4 THE A-PRIORI VALUE, COMMITTED NOW

**`tau = 2`**, by the direct analogue of round 1's justification: round 1 chose `tau = 1` on
standardised units so that a token one standard deviation above its answer's mean gets `e` times the
weight of an average token. On rank units the same statement is *"the top-ranked token gets `e` times
the weight of the median-ranked token"*, which gives `tau · (1 − 0.5) = 1`, so **`tau = 2`**. Chosen
from the stated principle, not from any result.

### 2.5 REGISTERED SUCCESS CONDITION — TWO PARTS, BOTH REQUIRED

**PRR alone is not sufficient here, and that is the point of the arm.**

1. **PRR:** `tau = 2` clears the same three-part bar as §1.3 against `msp_min`.
2. **MECHANISM:** `ρ(ESS, length)` **collapses toward zero** on the datasets where round 1 measured
   ρ ≥ 0.85 (`asqa` +0.946, `expertqa` +0.882, `factscore` +0.847). Registered threshold: **|ρ| < 0.3
   on all three**.

If (2) fails, the estimator is **not** length-invariant in practice whatever its algebra says, and
any PRR gain is not attributable to the fix. If (2) passes and (1) fails, that is the more
informative outcome and is reported as such: **the confound was real, removing it did not help, and
therefore the confound was not what was holding the family back.**

### 2.6 REGISTERED PRIOR — THE HONEST EXPECTATION IS NEGATIVE

- The **hard** top-fraction version of exactly this idea is already measured. The `frac` sweep in
  `results/topk_floor_sweep__…csv` peaks at the endpoint on the cross-dataset mean (`k = 0.01`,
  i.e. `msp_min`), declining monotonically thereafter.
- This is the **fourth family** examined on this test data.

**A negative here is the expected outcome and is a perfectly good result**: combined with §2.5(2) it
would say the length confound was real but not the binding constraint.

---

## 3. PROVENANCE

Driver: `scripts/checks/sharpening_family.py` (extended). Inputs: `record["token_logprobs"]` from
`cache/records/*.jsonl` plus the three `cache/*_rp12/records/*.jsonl` — no hidden states, no GPU, no
training. Selection uses only the 8 long evals; `sciq`/`trivia_qa` are carried for the external arm
and are **never pooled** into the n = 8. `LUQ_CARVE=legacy`, stamped into every output row. Outputs:
`results/sharpening_family__<slug>__round2.csv`, with the `sharpening_` prefix keeping it outside
every master-assembler glob. Machine: RCS, CPU only. Independent of the Qwen workstream.
