# Published results

The result files the reported tables and figures are computed from. `scripts/report_numbers.py`
reads this directory and recomputes the reported quantities from it.

There are two kinds. The three **masters**, one per evaluated population, hold the main comparison:
every method scored on every target under every training setting. The remaining files hold the
supplementary comparisons, which are computed by their own drivers and are not rows of a master.

| file | what it holds |
|---|---|
| `master_llama-3.1-8b.csv`, `master_qwen2.5-14b.csv`, `master_gemma-2-9b.csv` | the main comparison, one row per method, target and setting |
| `combination_equal_model.csv` | the combination of the constrained weighting with the hidden-state probe, averaged equally over the two populations with certified corrected-span per-response scores |
| `combination_per_model.csv` | the same, per population |
| `density_all_layer_llama-3.1-8b.csv` | the published density and hybrid estimators at their full layer set, per cell, with the layer count and response window recorded per row |
| `hybrid_backoff_llama-3.1-8b.csv`, `hybrid_backoff_gemma-2-9b.csv` | the back-off gate: the mean weight it gives its supervised branch per setting, and the paired comparisons for the substitution experiment |
| `token_probability_methods_llama-3.1-8b.csv`, `token_probability_methods_qwen2.5-14b.csv`, `token_probability_methods_gemma-2-9b.csv` | the token-probability comparison: the fixed aggregation rules, the relevance-weighted and answer-span variants, and the learned weighting scored on the same responses, with the rate at which an answer span was located |
| `response_lengths.csv` | mean and median response length per population and dataset, over all labelled rows and over the held-out evaluation partition separately |

The sections below describe the masters, which have the most involved schema.

## Scope

Three populations are published, and they are the three the report evaluates:
`meta-llama/Meta-Llama-3.1-8B`, `Qwen/Qwen2.5-14B` and `google/gemma-2-9b`. Other populations were
generated during the project and are not published here, because the report does not evaluate them
and a result table for a model the report never mentions would raise a question the report does not
answer. Their pre-registrations remain in `prereg/`, so the record that the work was done is public
even though the numbers are not.

## Columns of a master

| column | meaning |
|---|---|
| `model` | the population, as a Hugging Face model identifier |
| `eval` | the target dataset being scored |
| `rung` | the training setting: `ID`, `SameTask-long`, `LOO-long`, `DiffTask-long` or `1ds-Diff-long`, written in the report as ID, Same Task, LOO, DiffTask and 1ds-Diff |
| `train` | the labelled sources the method was trained on for this cell |
| `method` | the implementation key; `src/luq/method_names.py` maps it to the name used in the report |
| `family` | the presentation grouping |
| `prr_mean` | the Prediction-Rejection Ratio, averaged over the training seeds |
| `prr_std` | the standard deviation across those seeds |
| `n_seeds` | how many seeds contributed, normally 3, and 1 for a method that fits nothing (see below) |
| `degenerate_seeds` | seeds excluded by the generation-validity gate, normally 0 |
| `source` | which driver produced the row |
| `method_cells`, `complete_grid` | coverage for that method across the grid |

A blank `prr_mean` means the cell was not measured. It never means zero. That distinction is
enforced throughout the pipeline: a missing input raises or leaves the field blank rather than
being replaced by a neutral value.

## How these were produced

Each row is one method scored on one target dataset under one training setting, averaged over three
training seeds. Responses were generated once per population and are held fixed across all
settings; only the labelled training data change. The uncertainty score is compared with the graded
response-quality label by the Prediction-Rejection Ratio over the complete rejection curve,
implemented in `src/luq/results.py`. The long-form ladder that computes the cells is
`scripts/checks/probedriftlong.py`; the published distance and hybrid comparators come from
`scripts/checks/md_hybrids.py` and `scripts/checks/hbo.py`. `scripts/report_numbers.py` recomputes
the reported tables from the files in this directory.

Two cautions that apply to the supplementary files. `response_lengths.csv` carries two populations:
the reported length table is the held-out evaluation partition, in the `eval_test_*` columns, and
reading the wider `mean` column instead gives values that differ by several words. And
`density_all_layer_llama-3.1-8b.csv` records the response window per row, because a middle-layer row
under one window and an all-layer row under another differ in two ways at once; compare rows that
share a window.

## Two things to expect when reading these files

The masters carry more methods than the report reports. The report presents a selected comparison;
the master is the full grid that comparison was drawn from, including variants that were computed
and not carried forward.

`ptrue_unsup`, the training-free verification score, carries `n_seeds` of 1 and the literal `train`
value `(floor: eval set only)`. Neither is a gap. The score is read off the stored response rather
than fitted, so there is nothing for a training seed to vary and no labelled source it depends on,
and its value is therefore identical across all five settings for a given target. Read its five rows
per target as one measurement repeated, not as five observations: averaging them as though they were
independent would count the same number five times. The same caution applies to the fixed
probability aggregation rules, which do vary their `train` field but ignore it.

Four methods in the Llama-3.1-8B file carry `complete_grid` set to `NO`: `msp_satmd_mid` and
`msp_satrmd_mid` at 6 cells of 40, and `msp_satmd_alllayer` and `msp_satrmd_alllayer` at 15. These
are the entropy-augmented density variants, which the report states were not evaluated across the
benchmark because generation-time entropy was unavailable for two of the eight datasets. They are
left in the file rather than stripped, so the gap is visible rather than silent. Read the
`complete_grid` column before aggregating anything: a mean taken over a partial method is not
comparable with one taken over a complete method, and these four must not be read as results.
