# PRE-REGISTRATION — M6b: does the CAWSA + SAPLMA complementarity result replicate on Qwen2.5-14B?

## 0. Provenance and scope

Written and committed **before any Qwen2.5-14B ensemble PRR has been computed or read by anyone**. No
Qwen per-example sidecar for the ProbeDriftLong grid exists at the time of writing (verified: zero
per-token and zero feature caches for this population on RCS; the ladder that produced
`results/pdl_master__Qwen_Qwen2.5-14B.csv` ran without `--perex-dir`).

**This is a replication, not a method search.** It inherits `M6_hapes_saplma_ensemble.md` §2–§6 and §10
**unchanged**: the same component pair, the same single combiner, the same λ = 2, the same dataset-level
unit of analysis (n = 8), the same two estimands, and the same interpretation rule. Nothing about the
method is re-selected on Qwen. This document fixes **only what is new**: the replication verdict, the
role of the ExpertQA sensitivity, and one directional prediction.

**What is NOT claimed.** The Qwen *component* results are fully observed and public
(`STOCKTAKE_qwen.md`, the master table). They are the motivation for this registration, not evidence for
it. What has never been formed on this population is any of the four **ensembles**.

**Population caption for every table produced under this registration:**

> *`Qwen/Qwen2.5-14B` (base), fp32, layer 23, judge label gpt-5-mini, `carve=legacy`; complete
> ProbeDriftLong long grid — 8 evals × 5 rungs × 3 seeds, 40/40 cells.*

⚠️ **Never pooled with Llama.** Cross-model reporting is one explicitly-labelled row per population,
the shape `QWEN_REPLICATION_VERDICT.md` already uses.

---

## 1. Inherited, unchanged from M6

| item | value |
|---|---|
| primary | `rankavg{CAWSA λ=2, SAPLMA}`, equal weight, one combiner |
| control | `rankavg{SAPLMA, attention-pool}` |
| references | `rankavg{msp_min, SAPLMA}`, `rankavg{wmsp_norm, SAPLMA}` |
| unit of analysis | **the dataset, n = 8** |
| estimand A | per dataset, mean PRR over its 4 OOD rungs, ensemble − SAPLMA |
| estimand B | the same on the 8 ID cells |
| seed convention | combine **within** each seed, score, then average PRRs. **Never** average uncertainty vectors across seeds before scoring (M6 deviation D2: that error inflated SAPLMA at LOO by +0.050) |
| banned | λ sweep, combiner sweep, pair selection, dataset removal on ensemble performance, new trained combiner, HBO |

`zavg` is a robustness footnote only, never an alternative from which the better is chosen.

## 2. Validity gate, strengthened

Before any ensemble PRR is read, every component must reproduce
`results/pdl_master__Qwen_Qwen2.5-14B.csv` **per cell** — up to 8 components × 8 evals × 5 rungs = **320
comparisons** — at tolerance 1e-3, plus sidecar integrity (equal rows, equal lengths, 3 seeds for
trained methods, floors seed-identical) and 40/40 coverage.

⚠️ M6's original gate compared a *macro* against seven hardcoded values at 0.02 tolerance. That is
weaker than it looks, since a macro can match while cells are wrong in cancelling directions. The gate
was strengthened for both models on 2026-08-20, and the Llama result re-verified under it unchanged.

## 3. The replication verdict, fixed in advance

- **A — Strong qualitative replication.** CAWSA+SAPLMA improves OOD broadly without ID loss **and**
  gains clearly more than SAPLMA+attention.
- **B — Partial replication.** Some but not all of the Llama complementarity pattern transfers.
- **C — Non-replication.** The combination adds no useful signal, or behaves like the hidden-state
  control.

The Llama outcome this is measured against: OOD **+0.0343**, 7/8 datasets, CI [−0.0092, +0.0765],
p = 0.148 (**verdict NULL** under M6 §6); ID **+0.0079**, no material loss; control OOD +0.0086;
disattenuated correlation **0.519** (CAWSA↔SAPLMA) against **0.801** (SAPLMA↔attention).

⚠️ Llama's own primary was a **null**, so "replication" here means *reproducing the pattern*, not
reproducing a significant effect. A Qwen null with the same shape is category **A or B**, not **C**.

## 4. ExpertQA: mandatory sensitivity, and a directional prediction

`results/analysis/QWEN_GENERATION_VALIDITY_AUDIT.md` established that Qwen's ExpertQA has a
**text-identifiable label-zero cluster**: **26.5%** of labelled rows are quarantined at exactly **0.000**,
and a bare severe-degeneration indicator reaches **PRR +0.7297** there. That is a measurement confound,
not a UQ result.

**Therefore every aggregate is reported twice**: the **8-dataset grid is PRIMARY**, and a **7-dataset
ExpertQA-excluded** arm is reported beside it. The excluded arm is a **sensitivity and never replaces
the primary**. The write-up must state explicitly whether any replication conclusion depends materially
on ExpertQA.

### ⭐ 4.1 Registered directional prediction

Degenerate text is repetitive, so its token-NLL profile is distinctive. The confound should therefore
inflate the **probability-grounded** components (**CAWSA**, **`msp_min`**) **more than SAPLMA**, whose
signal is the hidden state.

> **Prediction: excluding ExpertQA shrinks the primary and the `msp_min` reference deltas MORE than it
> shrinks the SAPLMA+attention control delta.**

Registered before looking so that it is a test rather than a rationalisation available in either
direction. There is no strong Llama prior: ExpertQA's Llama ensemble delta was only +0.010.

⚠️ If the prediction **fails**, that is recorded as a failed prediction, not reinterpreted.

## 5. Exploratory, explicitly not confirmatory

The complementarity **bonus** (`ensemble PRR − mean(component PRRs)`, which is 0 for an ensemble that
merely interpolates) and the observation that the gain tracks CAWSA's own per-dataset strength
(Llama ρ = +0.952) were both found **post-hoc on Llama, after its null**. They are reported here as
**exploratory replication checks**. Their p-values are **not** the primary test, and at n = 8 any 8/8
result yields exactly p = 0.0078, the floor of the exact test — "maximally consistent", not a small
p-value.

## 6. Deviations

*(none at registration)*
