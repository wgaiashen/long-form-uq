# Pre-registration — pooling heads with different fixed recipes

> **Recorded outcome (added 2026-09-07):** run. The fixed-recipe heads do not beat the single-head pooler.

**Written 2026-08-06, BEFORE any cell of this arm has been run.** Committed with its numbers, because a
pre-registration without thresholds is barely one.

## What is being tested

Idea 3 was several attention heads with a choice made at test time. Idea 4 was a diversity-forcing
loss to stop them converging. B.2 ran K heads sharing ONE target and measured them collapsing to pairwise
attention correlation **1.0000**, so K heads only ever expressed two behaviours, "has a target" and "has
none". DoC's re-run (`DOC_AUX_RERUN_STATUS.md` §8) confirms this and states that what ran was not the
idea at all.

This tests the version that **cannot collapse**: each frozen head's attention *is* a fixed recipe, so what
makes the heads differ is not learned and no repulsion term is needed. That is **our extrapolation of
the idea, not his proposal** — his was multiple *learned* heads plus a diversity loss.

## Arms, declared in advance

| arm | heads | role |
|---|---|---|
| `floor_min` | — | unsupervised floor, `msp_min` |
| `armA` | 1 learned | the bar being improved; also the reproduction gate |
| `single_<recipe>` | 1 frozen | each recipe alone |
| `mh_diverse` | `nll` + `topk:25` + `content_mass` + 1 free | **the method** |
| `mh_same` | 3 × `nll` + 1 free | control: "more classifiers" vs "different recipes" |
| `mh_diverse_nofree` | 3 different recipes, no free head | is the free head carrying the ensemble? |

**The three recipes and the control's recipe are fixed here and not chosen from the results.** `topk:25`
is named because it was the strongest top-k arm on the OOD rungs of S4; selecting a per-dataset recipe from
test PRR would be an oracle, which is the trap `prereg/S4` already named.

## Population

**4 evals × 5 rungs = 20 cells**, 3 seeds: `pubmed_qa` (concentrated), `xsum` (no signal in the
probabilities), `cnn_dailymail` (spread), `factscore` (factuality).

This is an **indicative 4-dataset run, NOT the full 8-dataset ProbeDriftLong grid**, and every table
must say so in its caption. It does not satisfy the standing evaluation rule on its own.

## Decision rule, fixed in advance

- **PRIMARY — carry forward** if `mh_diverse` beats `armA` on the pooled OOD rungs by **> +0.02** (the
  measured aggregation-axis noise floor). Learned attention is the thing being improved and the
  condition is that the attention probe must actually improve.
- **SECONDARY — required** for the primary to mean anything: `mh_diverse` beats `mh_same` by **> +0.02**.
  If it does not, any gain is "more classifiers", not "different recipes", and **the line closes**.
- **DIAGNOSTIC, reported whatever happens**: head attention correlation. `mh_diverse` must be **low** by
  construction. If it comes back near 1.0 the wiring is wrong and no PRR from this run may be reported.

  **CORRECTION, 2026-08-06, after the first smoke cell — this clause was written wrong.** It originally
  said `mh_same` must be **≈ 1.0**. That is false for the arm as specified, and the smoke run returned
  **+0.5097**, which looked like a fault and is not one. `mh_same` is *three identical frozen recipes plus
  one free head*, so its 12 off-diagonal entries are 6 frozen-frozen pairs (exactly 1.0) and 6 frozen-free
  pairs (near 0); the mean is therefore ≈ 0.5 by construction, not ≈ 1.0. The error was mine in writing the
  expectation, not the code's: the self-test confirms that K identical recipes with **no** free head give
  correlation 1.000000000.

  **Restated expectation:** report the **full K×K matrix**, not just its mean, because the mean mixes pair
  types and is uninterpretable on its own. The checkable properties are (i) the frozen-frozen block of
  `mh_same` is exactly 1.0, (ii) the frozen-frozen block of `mh_diverse` is well below 0.9, (iii) the free
  head is near-zero with the frozen heads in both.

## Handicap carried, not netted out

B.2 measured every multi-head arm sitting **0.06–0.10 PRR below** the single-head pooler. That handicap is
stated alongside any result here. A `mh_diverse` that merely closes the gap to `armA` has not won.

## Predictions, so a miss is visible

1. **I do not expect this to beat SAPLMA.** Every fixed recipe individually loses to learned attention OOD
   (+0.179 vs +0.195 in S4), so the honest target is beating `armA`, not the strongest existing probe.
2. **`mh_diverse` > `mh_diverse_nofree`.** The free head contains the current best single method; removing
   it should cost. If removing it *helps*, the recipes are doing the work and that is more interesting
   than the headline.
3. **The heads will genuinely differ** (correlation well below 0.9). This is by construction, so it is a
   wiring check rather than a finding.
4. **If it works anywhere it will be on `xsum` and `cnn_dailymail`.** Those are where S4 showed different
   recipes winning by different amounts, i.e. where there is complementarity to exploit. A uniform gain
   across all four datasets is a **warning sign**, not a success.

## What would make me drop it

Any of: `mh_diverse` ≤ `mh_same` + 0.02; `mh_diverse` ≤ `armA` + 0.02 pooled OOD; or head correlation
coming back ≈ 1.0 for `mh_diverse` (wiring fault, results void).

## Constraints inherited from supervisors

- **Selection must be per dataset, never per instance**, a constraint agreed in supervision. The primary method here is the **ensemble**, which needs no selection at
  all. Any best-head-per-dataset number is a **declared oracle ceiling**, never a deployable result.
- **Idea 3's proposed selector is dead.** "Pick the head with the smallest ID→OOD entropy drop" rests on
  entropy-delta, which was rejected by its own pre-registered rule (`prereg/0.1`, `prereg/0.2`). No
  label-free per-dataset selector currently exists.
