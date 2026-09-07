# Pre-registration — does the CAWSA + SAPLMA complementarity result replicate on Qwen2.5-14B?

> **Recorded outcome:** run. Replicates, and is the clearest dataset-level effect of the three populations.

## 0. Provenance

Written and committed before any Qwen2.5-14B ensemble PRR has been computed or read. At the time of
writing no per-example sidecar exists for this population on the ProbeDriftLong grid: the ladder that
produced `results/pdl_master__Qwen_Qwen2.5-14B.csv` ran without `--perex-dir`, and the population had
no per-token or feature caches on RCS until they were transferred from DoC on 2026-08-20.

This is a replication, not a method search. It inherits sections 2 to 6 and section 10 of
`cawsa_saplma_ensemble.md` unchanged: the same component pair, the same single combiner, λ fixed at
2, the same dataset-level unit of analysis, and the same two estimands. Nothing about the method is
re-selected on Qwen. This document fixes only what is new: the replication verdict, the role of the
ExpertQA sensitivity, and one directional prediction.

The Qwen component results are already observed and public (the project's working notes, the master table).
They motivate this registration and are not evidence for it. What has never been formed on this
population is any of the four ensembles.

Population caption for every table produced under this registration:

> Qwen/Qwen2.5-14B (base), fp32, layer 23, judge label gpt-5-mini, carve=legacy; complete
> ProbeDriftLong long grid, 8 evals x 5 rungs x 3 seeds, 40/40 cells.

Results are never pooled with Llama. Cross-model reporting is one labelled row per population, the
form `QWEN_REPLICATION_VERDICT.md` already uses.

## 1. Inherited from M6, unchanged

| item | value |
|---|---|
| primary | `rankavg{CAWSA λ=2, SAPLMA}`, equal weight, one combiner |
| control | `rankavg{SAPLMA, attention-pool}` |
| references | `rankavg{msp_min, SAPLMA}`, `rankavg{wmsp_norm, SAPLMA}` |
| unit of analysis | the dataset, n = 8 |
| estimand A | per dataset, mean PRR over its four OOD rungs, ensemble minus SAPLMA |
| estimand B | the same on the eight ID cells |
| seed convention | combine within each seed, score, then average PRRs. Never average uncertainty vectors across seeds before scoring (M6 deviation 2: that error inflated SAPLMA at LOO by +0.050) |

Not permitted: sweeping λ, sweeping combiners, selecting component pairs, removing datasets on the
basis of ensemble performance, training a new combiner, or running HBO. `zavg` is a robustness
footnote only, never an alternative from which the better is chosen.

## 2. Validity gate

Before any ensemble PRR is read, every component must reproduce
`results/pdl_master__Qwen_Qwen2.5-14B.csv` per cell, up to 8 components x 8 evals x 5 rungs = 320
comparisons, at tolerance 1e-3. Sidecar integrity (equal rows, equal vector lengths, three seeds for
trained methods, floors seed-identical) and 40/40 coverage are checked alongside.

M6's original gate compared a macro against seven hardcoded values at 0.02 tolerance, which a macro can
pass while individual cells are wrong in cancelling directions. The gate was strengthened for both
models on 2026-08-20 and the Llama result re-verified under it with every published number unchanged.

## 3. Replication verdict, fixed in advance

- A, strong qualitative replication: CAWSA+SAPLMA improves OOD broadly without ID loss, and gains
  clearly more than SAPLMA+attention.
- B, partial replication: some but not all of the Llama complementarity pattern transfers.
- C, non-replication: the combination adds no useful signal, or behaves like the hidden-state control.

The Llama outcome this is measured against: OOD +0.0343, 7/8 datasets, CI [-0.0092, +0.0765], p = 0.148,
which did not clear the bar in the ensemble registration section 6; ID +0.0079, no material loss; control OOD +0.0086; disattenuated
correlation 0.519 for CAWSA against SAPLMA, and 0.801 for SAPLMA against attention.

Llama's own primary did not clear its bar, so replication here means reproducing the pattern, not
reproducing a significant effect. A Qwen result of the same shape is category A or B, not C.

## 4. ExpertQA sensitivity, and a directional prediction

`results/analysis/QWEN_GENERATION_VALIDITY_AUDIT.md` established that Qwen's ExpertQA contains a
text-identifiable label-zero cluster: 26.5% of labelled rows are quarantined at exactly 0.000, and a
bare severe-degeneration indicator reaches PRR +0.7297 there. That is a measurement confound, not a
UQ result. Qwen's ExpertQA also carries 2016 labelled rows against Llama's 1724, so it weighs more
heavily in this grid than in Llama's.

Every aggregate is therefore reported twice. The 8-dataset grid is primary; a 7-dataset
ExpertQA-excluded arm is reported beside it as a sensitivity and never replaces the primary. The
write-up must state explicitly whether any replication conclusion depends materially on ExpertQA.

Registered prediction: degenerate text is repetitive, so its token-NLL profile is distinctive. The
confound should therefore inflate the probability-grounded components, CAWSA and `msp_min`, more than
SAPLMA, whose signal is the hidden state. Excluding ExpertQA should shrink the primary and `msp_min`
deltas more than it shrinks the SAPLMA+attention control delta. This is registered before looking so
that it is a test rather than a rationalisation available in either direction. There is no strong Llama
prior: ExpertQA's Llama ensemble delta was only +0.010. If the prediction fails it is recorded as a
failed prediction, not reinterpreted.

## 5. Exploratory, not confirmatory

The complementarity bonus, defined as ensemble PRR minus the mean of the component PRRs and equal to
zero for an ensemble that merely interpolates, and the observation that the gain tracks CAWSA's own
per-dataset strength (Llama Spearman +0.952), were both found post-hoc on Llama after that result. They
are reported here as exploratory replication checks. Their p-values are not the primary test, and at
n = 8 any 8/8 result yields exactly p = 0.0078, the floor of the exact test.

## 6. Deviations

None at registration.
