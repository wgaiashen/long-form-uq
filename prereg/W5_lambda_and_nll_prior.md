# PRE-REGISTRATION — W5: λ = 1.5 from a July prediction, and NLL inside the weight logits

> **Status (2026-08-09):** run 8/8. The registered λ = 1.5 claim FAILS (tie with shrink@2); the anchor-quality mechanism holds (Spearman +0.835, LOO-stable). Record: the project results log

**Written 2026-08-09, BEFORE the driver was implemented and before any cell was run.**
Population: the COMPLETE ProbeDriftLong grid, 8 long evals × 5 rungs, `meta-llama/Llama-3.1-8B`,
legacy carve, 3 seeds. **Headline numbers are the mean over the 4 OOD rungs, n = 8 datasets.**
Plan: `../the project plan. Results: `../the project results log

---

## 1. Why this exists

Rounds 1 and 2 hunted a *new* free estimator and produced negatives. This registers the opposite
move: **improve the method that already wins where it matters.** Verified from
`results/pdl_master__meta-llama_Meta-Llama-3.1-8B.csv` (3-seed, per-rung means over the 8 long evals):

| rung | `msp_min` | wMSP@2 | Δ | best method at this rung |
|---|---|---|---|---|
| SameTask-long | +0.1855 | +0.2381 | +0.0526 | SAPLMA (+0.3157) |
| DiffTask-long | +0.1855 | **+0.2051** | **+0.0196** | **wMSP@2** (SAPLMA +0.2029) |
| LOO-long | +0.1855 | +0.2398 | +0.0542 | **wMSP@2** (SAPLMA +0.2369) |
| 1ds-Diff-long | +0.1855 | **+0.2320** | **+0.0465** | **wMSP@2** (SAPLMA +0.2092) |

**wMSP@2 is the best method at 3 of the 4 OOD rungs, including both hardest.** (It is *not* "the only
method `msp_min` never overtakes" — SAPLMA and wMSP@10 also never lose on the rung mean. The
best-at-3-of-4 statement is the accurate one and is the stronger claim.)

---

## 2. Q-B — λ = 1.5, PRE-COMMITTED, justified only by a prediction made on earlier data

### 2.1 THE REGISTERED VALUE AND ITS SOLE JUSTIFICATION

**λ = 1.5**, the midpoint of the interval predicted on **2026-07-12**, a month before the widened long
grid existed. `reports/OVERNIGHT_12jul.md:79,90`, quoted verbatim because the **date** is the entire
evidential value of the choice:

> **pubmed OneDatasetDiffTask / DiffTask:** shrink@1–2 +0.06 to +0.08 SIG (0.31–0.36, **above the
> 0.202 floor**) — a NEW long-form OOD win the coarse sweep missed.

> Sweet spot ≈ **shrink@1–2**. λ up to 3 still never significantly hurts.

and, from the same section: *"there is **not one significant hurt** across the entire fine sweep"*
for λ ∈ [0.5, 3].

That prediction names **the same two rungs** (DiffTask, 1ds-Diff) where wMSP@2 is currently the best
method. **λ = 1 and λ = 1.5 have never been run on the long grid.**

### 2.2 TWO CAVEATS, WRITTEN IN BEFORE THE RUN, NOT DISCOVERED AFTER

**(a) It is a prior prediction, not fully independent data.** The July sweep was on **pubmed**, i.e.
the same eval dataset, on the **pre-widened** population. So the honest claim is *"λ chosen by a
prediction made on earlier data"*. It is **NOT** *"λ was validated on independent data"*, and it must
never be written that way.

**(b) July's "clears the floor" used a floor the project has since rejected.** The 0.202 figure is
**`msp_sum`**, which `src/luq/msp.py:47-53` records as the rejected most-convenient baseline.
Against today's pre-registered bar, `msp_min` on pubmed is **0.371**, and July's 0.31–0.36 **does not
clear it**. So July established that λ ≈ 1–2 beats *unmoderated wMSP*; it did **not** establish that
it beats the floor.

### 2.3 REGISTERED BAR — the same three parts as W1, so it cannot be softened

λ = 1.5 must beat `msp_min` (+0.1855) on the **OOD mean, n = 8 datasets**, with all three of:

| part | threshold |
|---|---|
| margin | > **+0.010** |
| sign count | ≥ **6 of 8** |
| test | two-sided Wilcoxon signed-rank, **p < 0.05** |

**Secondary, reported alongside and never instead:** the per-rung breakdown (does the gain land on
DiffTask and 1ds-Diff, as July predicted?), and **λ selected by leave-one-dataset-out over
{norm, 1, 1.5, 2, 10}**, in the three-column old-beside-new style `selection_audit.py` uses.

### 2.4 REGISTERED FAILURE READING

**If λ = 1.5 misses the bar, that is a failure of the pre-committed value — even if some other λ in
the sweep clears it.** Promoting a different λ afterwards would be exactly the test-set selection
this whole workstream exists to avoid, and it is banned here in advance.

If λ = 1.5 passes but λ = 2 (already on the grid) passes by more, the registered reading is that
**the prediction was directionally right and the pre-committed value is what gets reported**, with
λ = 2's number shown beside it and labelled as the previously test-selected one.

---

## 3. Q-C — NLL inside the weight logits, so the family genuinely reaches `msp_min`

### 3.1 The defect this fixes, measured

Track 2 sharpened the *learned* logits, `w = softmax(raw / T)`. Its `T → 0` limit concentrates on
`argmax(raw)`, which is **not** `argmax(nll)`: measured agreement is **6–13%** against chance rates of
1–5% (the project results log). So Track 2 never spanned wMSP ↔ `msp_min` at all, and the
claim that it did was withdrawn.

**The existing `armD` does not fix this either, and it is important not to conflate them.** armD
adds `β·log(prior)` to the **attention pooler's** logits (`attn_pool.py:463`) and has already been run
at β ∈ {0, 0.25, 0.5, 1, 2, 4}. But armD pools **hidden states** — its tilt-→-∞ limit is a probe on the
worst token's hidden state, *not* `msp_min`. Only a tilt inside **weighted MSP**, which sums NLLs,
reaches `msp_min`. Q-C is therefore new, and is not a re-run of S8.

### 3.2 The registered estimators — BOTH tilts, because they are round 1's two families

```
exponential :  w = softmax( raw + tau * z(nll) )       z standardised within the answer
power       :  w = softmax( raw + beta * log(clip(nll, 1e-9)) )
```

Both reach `msp_min` as their parameter → ∞, regardless of what the query learned. **Both are run**,
because with `raw = 0` they are exactly round 1's softmax-τ and Lehmer-β families, and round 1 found
Lehmer the stronger of the two. Running only the exponential would rest the arm on the weaker
functional form by accident. The power tilt is additionally **scale-invariant by construction**
(rescaling NLL leaves the renormalised weights unchanged), so β transfers without standardisation;
`clip(..., 1e-9)` reuses `attn_pool.py:463`'s existing convention rather than inventing a second one.

Grids: `tau ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, inf}`, `beta ∈ {0, 0.5, 1, 2, 4, 8, 16, inf}`.
`T = 1` and `gamma = 0` throughout — **this is a different axis from Track 2's temperature and the two
must not be swept together**, or a win cannot be attributed.

### 3.3 THE THREE CONTROLS, RUN BEFORE ANY CURVE IS READ

1. **`tau = beta = 0` reproduces plain wMSP to `< 1e-6`** on the same trained model — the identity
   Track 2 uses, which has now passed **75/75** (cell, seed) checks.
2. **`tau, beta → ∞` reproduces `msp_min`'s PRR exactly.** **THIS IS THE CLAIM.** If it fails, "the
   family spans wMSP ↔ `msp_min`" is false and the arm is dead on arrival. This is precisely the check
   that would have caught the Track 2 error had it existed then.
3. **`argmax(w)` vs `argmax(nll)` agreement → 100%** in the limit, against the 6–13% Track 2 measured.
   This is the quantitative version of control 2 and is reported per dataset.

### 3.4 REGISTERED BAR AND ITS HONESTY POSITION

Q-C has **no pre-committed parameter** — unlike Q-B, there is no prior prediction to draw one from.
It is therefore registered as **exploratory**: the LODO-selected tilt is reported against `msp_min`
with the same three-part bar, and **the per-cell grid best is an ORACLE and is never a result**.

Given round 1's experience, the registered expectation is stated now: **honest selection is likely
to keep only a small fraction of any oracle gain**, and the number to report is that fraction, not
the oracle. Round 1 and Track 2 both showed the same pattern (14–41–22% as datasets were added).

**Stability is part of the report, not an afterthought.** Track 2's factor attribution flipped sign
when a single dataset was added. For any positive here, the **leave-one-out stability of the
conclusion** is reported alongside the number.

---

## 4. PROVENANCE

Driver: `scripts/checks/sharpening_lambda.py` (new, standalone). It imports
`weighted_msp.weighted_msp_unc` / `train_weighted_msp` / `_seq_q` and `weighting.shrink_to_uniform`,
and runs its own scoring loop, so it **edits no shared file** — `probedriftlong.py` and
`weighted_msp.py` are both on the Qwen port's list and are left untouched. Reuses
`probedriftlong.cells_long` / `build_rows` / `xl_rungs.eval_split` so the population matches the
master ladder cell-for-cell; the λ = norm/2/10 rows it produces must reproduce `pdl_master` to 4 dp,
the same outside check Track 2 passed on 25/25 cells. `LUQ_CARVE=legacy`, stamped into every row.
Outputs `results/sharpening_lambda_<eval>__<slug>.csv`, one job per eval. RCS, CPU only, no GPU.
