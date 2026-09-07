# Pre-registration — does attention flattening predict performance drop?

> **Recorded outcome:** run. A clean negative under the registered decision rule: the two correlation statistics disagree in sign and the rank statistic is wrong-signed, so this signal was not adopted.

**Written 2026-07-31, BEFORE running the analysis.** Committed ahead of the run so the predictions are
timestamped and cannot be retrofitted. Code repo HEAD at writing: see the commit that adds this file.

---

## Why this analysis exists

the explicit request at the 31 July meeting:

> *"Is there a correlation between how much it flattens — that delta — and how much performance drops?
> If so, that's a really useful signal."*

And the reason it matters:

> *"If the entropy doesn't change too much it's generalising successfully; if there's a big difference
> between ID and OOD then we think it's not generalising well, so we can switch."*

So this is not a curiosity. **It is the candidate label-free switching signal for the proposed system**
(weighted-MSP backing off to an attention probe). If it is weak, the back-off should be gated on
**output length** instead, and the system must not be rebuilt around entropy.

---

## The quantities

- **Δentropy (the "flattening")** — mean per-example normalised attention entropy `H / log T` on the OOD
  rung minus the same quantity at ID. Normalised because raw `H` is a length artefact across datasets
  spanning ~2 tokens (sciq) to ~384 (expertqa). Convention matches `pool_attention_ood_diag.panel`.
  **Positive = flatter OOD = the learned query dissolved toward uniform.**
- **PRR drop** — the attention pooler's own PRR at the OOD rung minus its PRR at ID.
  **Negative = performance got worse.**
- **The hypothesis predicts a NEGATIVE correlation:** more flattening (larger positive Δentropy) should
  go with a larger drop (more negative). Sign discipline matters here — a positive correlation is not
  "weak support", it is evidence *against*.

---

## Prior evidence already visible — the hypothesis is ALREADY broken at dataset level

From the project's working notes (E1 dump, job `3468720`, 4 datasets) joined to `pdl_master`
(`armA(attention)`, ID vs the mean of the four OOD rungs). **All four points verified at the time of writing:**

| dataset | Δentropy | pooler ID | pooler OOD mean | PRR drop |
|---|---|---|---|---|
| pubmed_qa | **+0.354** | 0.731 | 0.193 | **−0.538** |
| xsum | +0.080 | 0.577 | 0.240 | −0.337 |
| expertqa | +0.011 | 0.645 | 0.214 | −0.431 |
| cnn_dailymail | **−0.005** | 0.594 | 0.088 | **−0.506** |

**cnn is a direct counterexample.** Its attention barely moves (−0.005, i.e. very slightly *sharper*)
yet it drops almost as much as pubmed (−0.506 vs −0.538), whose attention flattens by +0.354. Two
datasets with near-identical drops sit at opposite ends of the Δentropy range.

Ranking the four by Δentropy gives pubmed > xsum > expertqa > cnn; ranking by drop severity gives
pubmed > cnn > expertqa > xsum. Only pubmed is in the same place in both.

---

## PREDICTIONS (registered before the run)

1. **Dataset-level (n = 8): the correlation will be WEAK**, and I would not be surprised by a
   near-zero or wrong-signed coefficient. Reason: cnn is an explicit counterexample and it is not a
   marginal point — it is one of the two largest drops.
2. **Per-cell (n = 32, all 8 evals × 4 OOD rungs): possibly more signal than at dataset level**, because
   rung severity varies within a dataset and the dissolution finding (§D.0) reported dissolution scaling
   with shift severity. This is the version worth running; it is the one that is genuinely unrun.
3. **Even if per-cell correlation is real, it may be driven by within-dataset rung ordering rather than
   by across-dataset differences** — which would make it useless as a switching signal, since the switch
   has to fire across datasets. **Report the within-dataset and across-dataset components separately.**

## If the dataset-level correlation comes back STRONG, treat it as a suspect

A strong result would contradict the four points tabled above, which are already on record. Before
believing it, check:
- that Δentropy is computed on the **same rungs** as the PRR drop (a rung mismatch is the exact class of
  bug that produced the retracted "router 0.400" cross-population number);
- that the entropy convention is normalised `H/log T`, not raw `H` (raw H correlates with length, and
  length correlates with the drop — that would be a **confound, not a signal**);
- that the pooler PRRs come from `pdl_master`'s `cells_long` population and not the broad-LOO population
  in `cache/router_ood/` (verified 2026-07-31 to be a *different* population);
- that the n=4 subset above reproduces inside the n=32 result.

## Decision rule, fixed in advance

- **Weak or wrong-signed** → entropy is **not** the switching signal. Phase 3 gates on **length**, and
  this is reported as a clean negative answering the question. That is a result, not a failure.
- **Strong and it survives the checks above** → entropy becomes a candidate gate, and it must then be
  compared head-to-head against the length gate leave-one-dataset-out before either is adopted.

## Outputs

Correlation (Pearson + Spearman) with a bootstrap CI at both dataset level and per-cell level, the
per-dataset scatter plot requested in review, and the within/across decomposition from prediction 3. State
n explicitly on every figure; n = 8 at dataset level is small and the limitation is stated, not hidden.
