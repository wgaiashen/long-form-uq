# Pre-registration: does the CAWSA and SAPLMA complementarity result replicate on Llama-3.1-8B-Instruct?

> **Recorded outcome:** run and resolved. The result was not carried into the report, which does not evaluate this population.

## 0. Provenance

Written and committed **before any Llama-3.1-8B-Instruct ensemble PRR has been computed or read.**
Verified at the time of writing: no file under `results/` containing this model slug holds an
ensemble, rank-average or complementarity result. The per-example sidecars this analysis reads
(`results/perex_wmodels_chat/meta-llama_Llama-3.1-8B-Instruct/`, 18 files) were produced by the
chat-template regeneration for a different purpose, the reduced multi-model panel, and no ensemble
has been formed from them.

This is a replication, not a method search. It inherits the component pair, the single combiner, the
shrinkage coefficient fixed at 2, the dataset-level unit of analysis and the two estimands from
`cawsa_saplma_ensemble.md` (referred to below as M6, the original registration of this experiment on
the development population). Nothing about the method is re-selected here.

**What is already known, stated plainly because it motivates this registration and is not evidence
for it.** The same estimator has been evaluated on three populations. It improves out-of-distribution
on all three, never costs in-distribution performance, and reaches significance on one:

| population | out-of-distribution gain over the probe | datasets improving | exact Wilcoxon p |
|---|---:|---:|---:|
| `meta-llama/Meta-Llama-3.1-8B`, original span | +0.034 | 7 of 8 | 0.148 |
| `meta-llama/Meta-Llama-3.1-8B`, corrected span | +0.0299 | 6 of 8 | 0.250 |
| `Qwen/Qwen2.5-14B` | +0.0411 | 7 of 8 | 0.0156 |
| `google/gemma-2-9b`, corrected span | +0.0207 | 6 of 8 | 0.383 |

**This registration exists because adding a fourth population after seeing a mixed result is
exactly the situation that requires the analysis to be fixed in advance.** The outcome is reported
whatever it is, including a null, and including an outcome that weakens the overall picture.

Population caption for every table produced under this registration:

> meta-llama/Llama-3.1-8B-Instruct, chat-template prompt regime `llama31i_chat`, judge label
> gpt-5-mini; reduced multi-model panel, 6 evals x 3 settings x 3 seeds, 18 cells.

Results are never pooled with any other population. Cross-model reporting is one labelled row per
population.

## 1. Inherited from M6, unchanged

| item | value |
|---|---|
| primary | `rankavg{CAWSA shrinkage 2, SAPLMA}`, equal weight, one combiner |
| control | `rankavg{SAPLMA, attention-pool}` |
| references | `rankavg{msp_min, SAPLMA}`, `rankavg{wmsp_norm, SAPLMA}` |
| unit of analysis | the dataset |
| estimand A | per dataset, mean PRR over its out-of-distribution settings, ensemble minus SAPLMA |
| estimand B | the same on the in-distribution cells |
| seed convention | combine within each seed, score, then average PRRs. Never average uncertainty vectors across seeds before scoring |

**Not permitted:** sweeping the shrinkage coefficient, sweeping combiners, selecting component pairs,
removing datasets on the basis of ensemble performance, training a new combiner, or running the
back-off variant. The rank-average with z-scoring is a robustness footnote only, never an alternative
from which the better is chosen.

## 2. Deviations from M6, and the consequence for power

This population is on the **reduced panel**, not the full grid. Two deviations follow, and both are
recorded here rather than discovered later:

- **D1. Six datasets, not eight.** `asqa`, `cnn_dailymail`, `factscore`, `pubmed_qa`, `samsum`,
  `xsum`. So n = 6, not n = 8.
- **D2. Three settings, not five.** In-distribution, different-task, and one-source-different-task.
  Estimand A is therefore the mean over **two** out-of-distribution settings rather than four.

**The power consequence, fixed in advance so that a null is read correctly.** For an exact
two-sided Wilcoxon signed-rank test at n = 6, the smallest attainable p-value is 2 / 2^6 = **0.031**.
Significance at the 0.05 level therefore **requires all six datasets to move in the same direction**.
At n = 8 the smallest attainable p is 0.0078 and 7 of 8 suffices.

**Consequently: a "not established" outcome on this population is substantially weaker evidence of
absence than the same outcome on the eight-dataset populations, and must not be reported as though it
carried equal weight.** This sentence is registered before the result is seen.

## 3. Validity gate

Before any ensemble PRR is read, every component must reproduce the population's own scored ladder,
`results/wmodels_master_chat__meta-llama_Llama-3.1-8B-Instruct.csv`, per cell at tolerance 1e-3.
Sidecar integrity is checked alongside: equal row counts, equal vector lengths, three seeds for the
trained methods, floors seed-identical, and 18 of 18 cell coverage.

**If the gate fails, no ensemble number is reported from this population.**

## 4. Replication verdict, fixed in advance

- **A, strong qualitative replication.** The combination improves out-of-distribution without
  in-distribution loss, and gains clearly more than the hidden-state control.
- **B, partial replication.** Some but not all of the pattern transfers, for example a positive
  direction that does not separate from the control.
- **C, non-replication.** The combination adds no useful signal, or behaves like the control.
- **D, reversal.** The combination is worse than the probe on the majority of datasets.

## 5. Directional prediction

Registered before the result: the out-of-distribution difference against the probe will be
**positive**, with a point estimate in the range already seen on the other three populations, roughly
+0.02 to +0.04, and **will not reach significance**, because six of six is required and no population
so far has been unanimous. **A significant positive here would be a stronger result than any obtained
so far and should be treated with corresponding suspicion**: the first check would be whether the
validity gate genuinely passed on all 18 cells.

## 6. What is reported regardless of outcome

The verdict category, estimands A and B with signs, confidence interval and exact p, the control's
values on the same cells, and the per-dataset table. The population caption states the reduced panel
and n = 6 every time. The power statement in section 2 accompanies any null.

## 7. Exploratory, not confirmatory

Any comparison not listed above is exploratory and labelled as such: per-setting breakdowns, the
reference ensembles, and any comparison against the other populations beyond a one-row-per-population
panel.
