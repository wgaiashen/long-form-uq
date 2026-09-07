# Published result masters

One file per evaluated population, added at submission. Each is the corrected-span master from
which the reported tables and figures were computed.

## Scope

Three populations are published, and they are the three the report evaluates:
`meta-llama/Meta-Llama-3.1-8B`, `Qwen/Qwen2.5-14B` and `google/gemma-2-9b`. Other populations were
generated during the project and are not published here, because the report does not evaluate them
and a result table for a model the report never mentions would raise a question the report does not
answer. Their pre-registrations remain in `prereg/`, so the record that the work was done is public
even though the numbers are not.

## Naming

`master_<model>.csv`, with the model written as it appears in the report: `master_llama-3.1-8b.csv`,
`master_qwen2.5-14b.csv`, `master_gemma-2-9b.csv`.

## Columns

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
| `n_seeds` | how many seeds contributed, normally 3 |
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

## Two things to expect when reading these files

The masters carry more methods than the report reports. The report presents a selected comparison;
the master is the full grid that comparison was drawn from, including variants that were computed
and not carried forward.

Two methods, `msp_satmd_mid` and `msp_satrmd_mid`, appear with only six cells filled and
`complete_grid` set to `NO`. Those are the entropy-augmented density variants that the report states
were not evaluated across the benchmark, because aligned token-entropy features were not available
for every required dataset. They are left in the file rather than stripped, so the gap is visible
rather than silent, and they must not be read as a result.
