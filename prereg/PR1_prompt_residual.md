# PR1 — Prompt-residual hidden-state probing (pre-registration)

**Date:** 2026-08-12. **Population:** `meta-llama/Meta-Llama-3.1-8B`, the canonical ProbeDriftLong
grid (8 long evals × 5 rungs, seeds 1/2/3, layer 15, carve `legacy`, 1800-row supervised pools).
**Committed BEFORE any `R1` or `R2` PRR existed.**

> Requested filename was `results/analysis/PROMPT_RESIDUAL_PREREG.md`. It lives here instead
> because `results/` is gitignored (`.gitignore:11`) and a pre-registration that is not in git is
> not a pre-registration. `prereg/` is where every other pre-registration in this project is
> tracked. The analysis output still goes to `results/method_dev/prompt_residual/`.

## 0. Honesty preamble — what this is and is not

- The complete Llama master ladder, the Qwen master, the sharpening-axis results and every closed
  wMSP variant were **inspected before this method was designed**. This is a prospective test of a
  new method on a **reused development population**, not independent confirmation.
- **One of the three arms is not new and its numbers are already known.** `R0` (anchor-inclusive
  mean pool) is the existing mean-pool control, at 40/40 in the master with mean OOD +0.2031. It is
  included here as a **reproduction gate**, not as a result. The **primary comparison `R2 − R1`
  involves only quantities that do not yet exist.**
- No dataset was selected for this experiment. The population is the entire 8-eval grid, so there
  is no target-selection step that could be cherry-picked.
- A genuinely confirmatory result would need a second model. Qwen replication is conditional on
  this gate passing and would run on DoC, where the L23 per-token cache already lives.

## 1. Hypothesis

Response-level hidden-state probes pool **absolute** states. Those states carry prompt, domain and
task identity, which is exactly the component that does not transfer when the eval target changes
task family. Representing the response as its **change from the prompt anchor** should strip a
shared task-specific component and improve cross-task ranking.

The per-token cache window is `[last_prompt_token] + gen_tokens` (`scripts/01h_pertoken.py:71`,
`lo, hi = P - 1, P + G`), so row 0 **is** the anchor `h0` and rows 1..G are the generated states.
Both quantities are already on disk; this experiment adds no generation, judging, GPU work or API
cost.

## 2. Method — three representations, one classifier

Per example, from the layer-15 per-token array `s` of shape `(G+1, d)`:

| name | vector | role |
|---|---|---|
| `resid_R0_anchormean` | `s.mean(0)` | anchor-inclusive mean = the existing mean-pool control. **Reproduction gate.** |
| `resid_R1_genmean` | `s[1:].mean(0)` | generated-only mean. The matched partner. |
| `resid_R2_promptresid` | `s[1:].mean(0) - s[0]` | prompt-residual mean. **The method under test.** |

`R2` is `R1` with the *same* weight vector applied negatively to `h0`: a linear probe on `R2` is
the tied-weight (`+w`, `−w`) special case of a probe on `[R1, h0]`. It is therefore a **restriction**
of the representation, not extra capacity — the same shape as the one intervention that has worked
on this benchmark (shrinking learned weights toward uniform buys OOD robustness at an ID cost).

**Classifier and recipe, identical across all three arms:** the canonical mean-pool control's, i.e.
`attn_pool.train_attn(..., freeze_query=True)` — single `Linear(d,1)` head, Adam, lr 1e-3, 60
epochs, batch 32, weight decay 1e-2 on the head, `BCEWithLogitsLoss` against the graded judge label,
seeds 1/2/3. Uncertainty = `1 − sigmoid(logit)`.

**How the recipe is reused without editing `attn_pool.py`** (which is on the Qwen port's no-edit
list): each pre-pooled vector `z` is passed as a length-1 pseudo-sequence of shape `(1, d)`. With
`freeze_query=True` the query stays at zeros, so scores are 0 and the softmax over a single real
position is exactly 1.0, giving `pooled == z` into the same head with the same RNG draw order.

**This equivalence was verified before the prereg was committed, on synthetic data:** attention
weight exactly 1.0 at length 1; frozen-query attention exactly uniform over real tokens at full
length; and trained-model agreement between `train_attn(full states)` and
`train_attn(pseudo-sequence of s.mean(0))` of **1.19e-07** on held-out logits, 1.49e-08 on head
weights. Verification script retained in the session scratchpad; the same identity is re-asserted
per cell at runtime (§5).

Nothing else changes. Same layer, same cells, same sampled training rows, same seeds, same eval
split, same PRR harness.

## 3. Population and cells

The **complete** canonical grid, no subset: 8 long evals (`pubmed_qa`, `xsum`, `cnn_dailymail`,
`med_quad`, `samsum`, `expertqa`, `asqa`, `factscore`) × 5 rungs (`ID`, `SameTask-long`,
`DiffTask-long`, `LOO-long`, `1ds-Diff-long`) × 3 seeds × 3 arms = 360 fits.

The full grid is run rather than a diagnostic subset because the cost is dominated by loading the
per-token caches (~13 GB), which is paid whether 6 cells or 40 are scored. There is no
target-selection step and therefore no target-selection bias.

`SAPLMA` and `msp_min` are carried as external references, read from the existing master — not
refitted.

## 4. Primary estimand, and the decision rule (fixed now)

**Unit of analysis: the dataset, n = 8.** Not "32 OOD cells", which is 8 values counted four times
and yields an interval roughly twice too narrow.

For dataset `e`, `OOD_mean(m, e)` = mean PRR over the 4 OOD rungs (3-seed mean per cell).

```
delta_e     = OOD_mean(R2, e) - OOD_mean(R1, e)
macro_delta = mean_e(delta_e)
```

**Primary comparison is `R2 − R1`, never `R2 − R0` alone.** Comparing the residual only against the
anchor-inclusive mean would confound "subtracting `h0`" with "dropping `h0`", which is the whole
reason `R1` exists.

**Promising if all three hold:**
1. `macro_delta > +0.010`
2. `delta_e > 0` on at least **6 of 8** datasets
3. every leave-one-dataset-out macro stays positive

This is a decision rule for whether to spend the remaining time, **not a significance claim**. It
will not be altered after results are seen.

**Reported regardless of outcome:** all 8 per-dataset deltas; the **per-rung breakdown reported
separately for `ID`, `SameTask`, `LOO`, `DiffTask`, `1ds-Diff`**; per-cell seed mean and sd; exact
paired two-sided Wilcoxon p; dataset-level bootstrap CI; all 8 LODO macros; the `R2 − R0` and
`R1 − R0` decompositions; and `SAPLMA` alongside as the external bar.

## 5. Secondary coherence axis — a directional prediction, stated before results

If residualisation works by removing task-specific prompt information, then "better everywhere" is
**not** the expected pattern. The predicted ordering is

```
gain(DiffTask, 1ds-Diff)  >  gain(SameTask, LOO)  >  gain(ID)
```

with `gain(ID)` permitted to be **negative**. An ID cost bought against a hard-OOD gain is recorded
in advance as a **pass** on this axis, because it is what the mechanism predicts. A method that
improved ID and hard-OOD equally would be evidence of a *general* capacity gain rather than the
claimed task-relative one, and would be reported as such.

The converse is also fixed now: if `R2` improves ID but not the hard OOD rungs, the mechanism claim
is **refuted** even if the macro mean is positive.

## 6. Correctness gates — the run stops if any fails

1. **Reproduction gate (blocking).** `R0` must reproduce the canonical mean-pool control to <1e-6
   on matched cells and seeds. `R0` is not a new measurement; if it drifts, the harness is wrong and
   no arm is reportable.
2. **Empty-generation gate.** `G == 0` makes `s[1:]` an empty slice, which numpy silently returns as
   `NaN` with only a `RuntimeWarning` — a plausible-looking number in place of an absence. The
   driver **asserts `s.shape[0] >= 2` and aborts**. It never falls back to `h0`.
3. **Absence is blank, never zero.** A missing per-token cache leaves the cell empty and says so.
4. **Coverage.** Completeness of 8 × 5 × 3 × 3 is stated before any table; missing cells are named.
5. **Population in every caption.** Llama and Qwen are never pooled.

## 7. Promotion rule

Headline status requires the project's standing rule (`prereg/W8_adaptive_lehmer.md` §2): the effect
is meaningfully positive across datasets, not driven by a single dataset (LODO positive under every
omission), and directionally supported by the predeclared secondary axis (§5). Otherwise the result
is reported as mechanism evidence or as a pre-registered negative. No post-hoc favourable story.

**Scope limit if it passes.** Exactly one follow-up: the same residual representation through the
SAPLMA MLP head. Explicitly **not**: residual attention pooling, residual weighted MSP, per-token
residual routers, multi-anchor variants, layer sweeps, or learned prompt subtraction.

**If it fails.** The direction closes. No tuning of the head, learning rate, epochs, weight decay or
layer to rescue it.

## 8. AMENDMENT 2026-08-12 (same day, before the full run) — the smoke, and the R3 control

**Recorded rather than hidden.** After §§1-7 were written, a labelled SMOKE run
(`pubmed_qa`, `ID` + `DiffTask-long`, seed 1, pools hard-subsampled to 120 train / 60 test) was
executed to verify the harness. It passed the reproduction gate at exactly `0.00e+00`. It also
produced PRR numbers, so the claim "committed before any R1/R2 PRR existed" is now **qualified**:
it holds for every reportable number, not for the smoke. The smoke values were:

| rung | canonical | R0 | R1 | R2 |
|---|---|---|---|---|
| `ID` | +0.0713 | +0.0713 | +0.0612 | **+0.5783** |
| `DiffTask-long` | −0.2604 | −0.2604 | −0.3037 | **−0.3467** |

**What this changed, and what it did not.** The primary comparison, the scope, the gate thresholds
in §4 and the directional axis in §5 are **UNCHANGED**. One **control** is added:

`resid_R3_constanchor` = `s[1:].mean(0) − mean_over_TRAIN_rows(h0)`.

**Why.** Subtracting the example's own anchor does two things at once. It makes the representation
task-relative (the claim), and it removes a large offset that hidden states share — which alone
could be better *conditioning* for a linear head trained without input standardisation for 60
epochs. Those two mechanisms have the same sign, so R2 on its own cannot separate them. R3 delivers
the conditioning and nothing else:

- `R2 ≈ R3` → the gain is centring, **not** prompt-relativity. The mechanism claim fails.
- `R2 ≫ R3` → the per-example anchor carries real signal.

The constant is computed on **training rows only**; using all rows would leak the test set into the
representation.

**This is a control, not a variant.** No arm was added to improve the method's chances, and no
threshold moved. The smoke is labelled `smoke=True` in its CSV, is written to a `__SMOKE` filename,
is rejected by the verdict script, and never enters a results table.

**Note the direction of the smoke.** R2's large gain is on **ID**, and on the hard OOD rung it is
the **worst** arm. That is the pattern §5 pre-registered as **refuting** the mechanism. Stated here
so that if the full grid reproduces it, the conclusion was fixed in advance and not fitted after.

## 9. What this cannot show

`h0` is a single token's state. If `R2` fails, that is evidence against *this* anchor, not against
task-relative representation in general. If `R2` succeeds, the mechanism claim ("removes
task-specific components") is supported by §5's ordering but not proven — a representation-level
probe of what `h0` actually encodes would be needed, and is out of scope before the freeze.
