# PRE-REGISTRATION — M3: the split rule, label coverage, and the scope of R1–R4

**Written 2026-08-08, BEFORE any Qwen2.5-14B generation exists.** Committed ahead of the run, so
every threshold and decision rule below is fixed while only the Llama numbers are known.

Companion to `prereg/M2_qwen14b_replication.md` (the four replication claims) and
the project's planning notes (the implementation).

---

## 1. The defect this fixes

The ladder dropped unlabelled rows and **then** carved the 30% test split. On two datasets the judge
label is absent for a reason that depends on **what that model generated**:

`factuality` is `None` exactly when `uncovered == 1.0` — the judge ran and found **no claim the
reference could adjudicate**, so `SUPPORTED/(SUPPORTED+CONTRADICTED)` is 0/0.

Verified against the caches: the judge ran on **100%** of rows in both files (factscore 500/500,
expertqa 2016/2016) — every row carries a `factuality_model` stamp, including the unlabelled ones.
These are **judged, no denominator**, not "not yet judged", so they can never be filled in and
**re-scoring costs £0 — compute only, zero new API calls.**

Consequence: a different model produces a different labelled subset, hence a different test row set,
so R1–R4 would aggregate across **different populations** on 2 of the 8 datasets.

## 2. The registered rule

**Carve 30% of ALL rows (fixed `RandomState(0)` permutation, positional), then score whichever of
those rows carry labels.** Recorded in the manifest: `seed=0`, `test_frac=0.30`, order
carve-then-filter.

Measured consequence on Llama-3.1-8B — **only the two that drift move**:

| dataset | rows | labelled | legacy test | new scored / carved | verdict |
|---|---|---|---|---|---|
| med_quad | 1800 | 1800 | 540 | 540 / 540 | identical |
| samsum | 1800 | 1800 | 540 | 540 / 540 | identical |
| asqa | 948 | 948 | 284 | 284 / 284 | identical |
| **expertqa** | 2016 | 1724 | 517 | **516 / 605** | **MOVES** |
| **factscore** | 500 | 455 | 136 | **133 / 150** | **MOVES** |

**Positional, not hashed** — a content hash would also reassign med_quad/samsum/asqa, which have no
drift problem: five datasets re-scored instead of two, for no gain. Positional is reproducible here
because row order is a deterministic function of the data (verified: `load_records` preserves file
order; all five carve-relevant datasets are single-split, contiguous and idx-ordered), and an **order
guard** now enforces that rather than assuming it.

**HONESTY CLAUSE — carried verbatim into every write-up of this change.**
*Freezing the row set does NOT fully equalise the populations. Each model is still scored on the
subset its own judge could label, so the compared row sets still differ. The gain is that the
difference is now fixed, visible and measurable rather than silent. Do not claim the problem is
solved.*

## 3. Label coverage is a first-class number

Reported per model per dataset in **every table**, never a footnote. Llama-3.1-8B:

| dataset | labelled / rows | coverage |
|---|---|---|
| pubmed_qa, xsum, cnn_dailymail, med_quad, samsum, asqa | all | **100%** |
| expertqa | 1724 / 2016 | **85.5%** |
| factscore | 455 / 500 | **91.0%** |

### Registered coverage-divergence thresholds: **>5 pp record, >10 pp flag**

Measured, not chosen: on Llama, x% of each test set was dropped at random (400 draws) and `msp_min`
PRR re-scored.

| coverage drop | expertqa PRR sd | factscore PRR sd | vs the R3 effect (+0.054) |
|---|---|---|---|
| 2 pp | 0.010 | 0.014 | ~0.2× |
| 5 pp | 0.018 | 0.021 | ~0.4× |
| **10 pp** | **0.024** | **0.030** | **~0.5×** |
| 20 pp | 0.037 | 0.042 | ~0.7× |
| 30 pp | 0.045 | 0.060 | ~1.0× |

**Justification.** At 10 pp the PRR noise induced by coverage loss alone reaches **half the effect
size R3 is testing**, and the worst single draw shifted PRR by ~0.09 — larger than R3's entire
effect. Beyond that, a per-dataset cross-model difference of the size we care about could be
manufactured by coverage divergence rather than by the model.

**These are LOWER bounds.** The simulation drops rows *at random*; the real drop is **not**
random. Measured (§5): dropped rows are significantly shorter and score-correlated —
`perplexity` differs by **+0.058 (expertqa)** and **+0.310 (factscore)** between dropped and kept
rows. So 10 pp is if anything generous.

## 4. Registered scope rule for R1–R4

- **Within-model** (Llama alone, Qwen alone): **all eight datasets primary.**
- **Reported immediately alongside, at equal status** (not an appendix): the **six
  correctness-labelled** datasets only, excluding expertqa + factscore.
- **Cross-model replication verdict:** report both. **If coverage divergence on expertqa or factscore
  exceeds the 10 pp flag, the six-dataset version becomes primary for that claim.** Mechanical, fixed
  now, so it is never a judgement made after seeing Qwen.

**Why all eight stay primary** — measured on Llama, excluding the two changes **no conclusion**:

| claim | all 8 | six only |
|---|---|---|
| R1a msp_min − msp_sum | +0.1310, 8/8, p=0.0078 | +0.1400, **6/6**, p=0.0312 |
| R2 slope (H0: b=1) | b=+0.242, p=0.0041 | b=+0.072, **p=0.0097** |
| R3 DiD | +0.0540, 7/8, p=0.0156 | +0.0664, **6/6**, p=0.0312 |

R1a and R3 get *stronger* and become unanimous on six; R2's slope moves further below 1. And within
the factuality pair alone R2's slope is **0.918** — SAPLMA tracks msp_min nearly one-for-one there,
the **opposite** of the claim — so those two **weaken** R2 rather than manufacturing it. Nothing is
rescued by exclusion, so there is no gain to offset narrowing the thesis claim to reference-agreement
or the appearance of post-hoc selection.

**Intersection sensitivity.** R1–R4 additionally recomputed on the rows **both** models' judges could
label, in its own table, marked as a sensitivity analysis. **Never written back into the frozen
master** — that would let Qwen retroactively change the Llama numbers.

## 5. Registered finding: the dropped rows are SHORTER, not longer

Tested because derailed rambling is plausibly both long and unadjudicable, and expertqa is the high
anchor of the length analysis. **The hypothesis is refuted; the direction is the opposite.**

| dataset | group | n | median | % at budget |
|---|---|---|---|---|
| expertqa | dropped | 292 | **177.0** | 12.0% |
| | kept | 1724 | 200.5 | 18.2% |
| factscore | dropped | 45 | **47.0** | 2.2% |
| | kept | 455 | 90.0 | 5.3% |

Mann-Whitney **p = 1.2e-4** / **3.4e-4**. Mechanism: `uncovered == 1.0` needs **zero** adjudicable
claims, easiest when the answer makes **few** claims. Effect on the length profile is small and safe
(expertqa mean 213.9 → 217.7, +1.7%), so the high anchor holds.

**Registered directional expectation for Qwen:** dropped rows will again be shorter than kept rows on
both datasets. A reversal would mean the absence mechanism differs by model and would invalidate
treating coverage as a simple scalar.

## 6. R1b — retired to descriptive, and why it still matters

The published short-form result (SelfCheckGPT, Manakul et al. 2023) is that **Max(−log p) beats
Avg(−log p)** on short sentences. Our long-form data does **not** reproduce that ordering: `msp_min`
beats `perplexity` on only **5/8** datasets (p=0.46), and on the cleaner six-dataset subset **3/6** —
a literal coin flip.

The split is by **where the signal sits**, not by noise:

| eval | msp_min | perplexity | min − ppl | winner |
|---|---|---|---|---|
| pubmed_qa | +0.3710 | −0.1736 | **+0.5446** | min |
| expertqa | +0.2054 | +0.0393 | +0.1661 | min |
| xsum | −0.0149 | −0.1719 | +0.1570 | min |
| factscore | +0.4283 | +0.3260 | +0.1023 | min |
| med_quad | +0.1492 | +0.0771 | +0.0721 | min |
| asqa | +0.2498 | +0.3161 | −0.0663 | perplexity |
| samsum | −0.0243 | +0.1128 | −0.1371 | perplexity |
| cnn_dailymail | +0.1198 | +0.4096 | **−0.2898** | perplexity |

**Registered as descriptive, never as a replication target**, but framed as a contribution rather
than a null: *the published short-form ordering of Max vs Avg token probability does not transfer to
long-form generation.* **Registered directional expectation for Qwen:** neither aggregate dominates
across datasets; the per-dataset winner tracks where the signal is concentrated rather than a
universal ordering.

## 7. R4 — corrected to "wins one rung, ties the other"

Re-checked against the 3-seed master table. The earlier "wins the two hardest rungs" is **wrong on
the DiffTask rung**:

| rung | wMSP@2 | SAPLMA | margin | wins | Wilcoxon p | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| DiffTask-long | +0.2051 | +0.2029 | **+0.0022** | 4/8 | **1.0000** | [−0.093, +0.107] |
| 1ds-Diff-long | +0.2320 | +0.2092 | +0.0228 | 6/8 | 0.6406 | [−0.068, +0.106] |

**Registered as: wMSP@2 is directionally ahead at `1ds-Diff-long` (6/8, +0.023) and TIES at
`DiffTask-long` (4/8, +0.002).** Both CIs span zero, so **neither rung is a significant win** —
`1ds-Diff-long` is directional only. R4 remains descriptive with no inferential claim, and a Qwen
non-replication of it carries no information.

## 8. Order of operations

1. Gate 1 (index equivalence) — passed 2026-08-08, 126/126 byte-identical.
2. Commit the modified tracked files (the ladder refuses to run on a dirty tree).
3. Gate 2 — re-run samsum + factscore under `LUQ_CARVE=legacy`; PRR must match exactly.
4. Gate 3 — `LUQ_CARVE=all-rows`, re-score expertqa + factscore only; the other six must not move.
5. Record old beside new in the project's working notes as a dated revision, never a silent edit.

Re-runs use the **paper method set only**: floors, SAPLMA, attention pooler + mean-pool control,
`wmsp_norm` + `wmsp_shrink2` — matching the Qwen Tier 1 set so both models stay method-matched.
