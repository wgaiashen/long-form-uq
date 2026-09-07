# CAWSA and ProbeDriftLong: uncertainty estimation for long-form generation

MSc Computing project, Imperial College London.
Author: Gaia Shen. Supervisors: Joe Stacey and Lihu Chen.

## What this is

This project studies white-box uncertainty estimation for long-form LLM generation: producing one
uncertainty score for a whole generated response, and asking whether that score stays useful when
the task changes.

Two things come out of it.

**ProbeDriftLong** is the evaluation framework. It takes eight long-form evaluation sets across
three task families (question answering, summarisation and factual writing) and varies the labelled
data a supervised estimator is trained on, from in-distribution through to a different task
entirely. That makes cross-task transfer, rather than matched-task accuracy, the thing being
measured.

**CAWSA** (Constrained Activation-Weighted Surprisal Aggregation) is the method. It keeps token
surprisal as the quantity being aggregated, uses middle-layer activations to learn which generated
positions matter more, and shrinks those learned weights back towards uniform aggregation. The
constraint is the point: supervision is used to learn *where to read* the probability signal rather
than to learn an unrestricted mapping from activations to correctness, which is what loses its
advantage under task shift.

## ProbeDriftLong

ProbeDriftLong is the controlled long-form out-of-distribution evaluation framework introduced in
this work. Its benchmark definition is implemented in `src/probe_drift_long/`, while dataset loading
and experiment execution are provided by the surrounding project code.

The definition is deliberately small and separable: `dataset_configs.py` fixes the eight evaluation
targets, their task families and the quality label each carries; `ood_settings.py` fixes the five
training settings, the sources feeding each and the 1,800-example budget with its per-source caps;
and `splits.py` fixes the deterministic train and test carve, its seed and the row-order guard that
keeps it reproducible. Together they answer only the question of which examples go where. They do not
generate, judge, extract features, train probes or compute PRR, all of which live in `src/luq/` and
`scripts/`.

Five of the eight evaluation targets are supplied by the upstream ProbeDrift library
(`github.com/joestacey/ProbeDrift`), which covers short-form and mixed tasks, and are inherited
unchanged so that the reproduction stays faithful. The remaining three have their own loaders in
`src/luq/`. The benchmark datasets themselves are obtained from their original public sources and are
not redistributed here.

## Approach

1. Generate long-form outputs from a frozen base model.
2. Label each output for response quality (an LLM judge on a graded 0 to 1 scale, and a claim-level
   factuality pipeline for ExpertQA and FActScore; string match is used only for the short-form sets).
3. Fit the uncertainty estimator on cached internal features and token probabilities.
4. Evaluate with the prediction-rejection ratio (PRR), in distribution and under task shift.

PRR ranks responses by uncertainty and progressively rejects the most uncertain ones. If the score
is informative, the average quality of what remains climbs. 1 is oracle ordering, 0 is no better
than random, and below 0 is worse than random. It is used rather than AUROC because long-form
response quality here is graded rather than binary.

A method is just a feature extractor plus an aggregation choice. The rest of the pipeline is
shared, so every method is scored by the identical harness on the identical population.

**Models.** Three base models are evaluated on the full eight-dataset benchmark:
`meta-llama/Meta-Llama-3.1-8B` at layer 15 of 32, `Qwen/Qwen2.5-14B` at layer 23 of 48, and
`google/gemma-2-9b` at layer 20 of 42. The layer is fixed per model by a single rule and is never
selected per target, because a target-specific layer would not be available under transfer. The
three are never pooled into one table: each result is reported per population, and cross-model
agreement is stated per claim. Llama-3.1-8B carries the widest method coverage and is the model
used for the aggregation analysis.

**Datasets.** Eight long-form evaluation sets (PubMedQA, XSum, CNN/DailyMail, SAMSum, MedQuAD,
ASQA, ExpertQA, FActScore) across three task families. Two short-form sets, SciQ and TriviaQA, are
reachable through a transfer setting but are not part of the benchmark and are not evaluated in the
write-up.

**Methods.** CAWSA is implemented as `wmsp_shrink2`, with its unconstrained control at lambda = 0
as `wmsp_norm`. It is compared against three training-free probability aggregates (Minimum token
probability `msp_min`, Mean token NLL `perplexity`, Sum NLL `msp_sum`), three hidden-state methods
sharing the same features (SAPLMA, a mean-pooling control and learned attention pooling), and the
combination of CAWSA with SAPLMA. Those form the panel evaluated on all three models.

Further comparators are evaluated on fewer models: P(True) in a training-free and a probe form, and
Lookback Lens, on two; and on Llama-3.1-8B, TokenSAR, the answer-span probability baselines, the
Mahalanobis and relative-Mahalanobis density estimators with their supervised and hybrid forms, and
Hybrid Back-Off. `src/luq/method_names.py` maps every implementation key to the name used in the
write-up.

Alongside these the repository holds the method development that did not carry: hard token masks,
sentence-level aggregation, probability-guided and multi-head pooling, and several weighting schemes
that were registered and then failed their own bar. These are development evidence rather than
benchmark results, were mostly run on one model, and are summarised as such in the write-up. They
are kept because the argument that no fixed aggregation rule works depends on knowing which nearby
designs were tried and did not work; `prereg/README.md` groups them with their outcomes.

The repository also contains work that the write-up does not report at all: an earlier ten-dataset
benchmark that preceded this one, a long-to-short transfer experiment, and two model populations
that were generated and not carried forward. They are left in place rather than pruned, so that what
was run is visible, but nothing in the write-up rests on them.

## Pipeline (`scripts/`)

Numbered stages run in order. Stage 01 is the only GPU-heavy step; everything downstream reads
its cache, so probes can be retrained without touching a GPU.

| stage | what it does |
|---|---|
| `01_extract.py` | generate, cache the Tier-1 records (token ids, logprobs) and pooled hidden states |
| `01b_ptrue.py`, `01c_lookback.py`, `01s_sar_relevance.py`, `01o_orgad_llm_extract.py` | extra per-method features, all from the cached records |
| `01e_repool.py` | re-pool the feature cache teacher-forced, so it matches the per-token cache |
| `01h_pertoken.py` | per-token hidden states for the aggregation work |
| `01f_alignscore.py` | the secondary AlignScore label |
| `02_label*.py` | correctness labels (string match, LLM judge, ExpertQA and FActScore variants) |
| `03_probe.py` | train the probe on cached features |
| `04_eval.py` | the PRR table |
| `05_transfer.py`, `06_pool.py` | cross-task transfer matrix and pooled training |

## Repository layout

- `src/probe_drift_long/` — the ProbeDriftLong benchmark definition: the evaluation targets and
  their task families, the five training settings with their budgets and caps, and the
  deterministic train and test carve. See the section above.
- `src/luq/` — the pipeline library: data, generation, cache, probe, results, plus
  `features/` (saplma, ptrue, lookback, sar, orgad) and `labels/` (string match,
  LLM judge, AlignScore, FActScore).
- `scripts/` — the numbered stages above, plus `checks/` (309 analysis and verification
  drivers, see `scripts/checks/README.md`) and `tools/` (22 helpers, mostly visualisation).
- `prereg/` — 41 pre-registrations, written and committed **before** the runs they describe, so
  the commit timestamp shows a prediction pre-dates its result. Grouped by what each supports, with
  its outcome, in `prereg/README.md`. Most record a negative, and are kept for that reason: the
  argument that no fixed aggregation rule works across tasks depends on knowing which nearby designs
  were tried and failed.
- `tests/` — 14 files, 93 CPU unit tests over the uncertainty maths, the aggregation code and
  the benchmark definition.
- `pbs/`, `slurm/` — cluster job scripts (see below).
- FActScore's entity list and Wikipedia database are not redistributed here; `src/luq/factscore.py`
  documents what the labeller expects and where it looks for them.

### `results/` is deliberately not in this repository

The drivers read and write a `results/` directory at the repo root, and **a fresh clone will not
have one**. The result tables contain unpublished numbers and are version-controlled separately.
In the development environment `results` is a symlink to a directory outside this repo, which is why
every path in the code still resolves unchanged.

To run anything that writes results, create the directory first:

```bash
mkdir -p results
```

Model weights, caches (`cache/`), generated outputs and scheduler logs are likewise not tracked.

## Setup

Python 3.11. Heavy and GPU steps run on a real compute node, never on a login node.

```bash
conda activate luq
export HF_HOME=/path/to/big/volume/hf_cache   # NOT ~/.cache, home is usually over quota
export OPENAI_API_KEY=...                     # only for the long-form judge (02_label)
```

Main dependencies are pinned in `requirements.txt`: PyTorch, transformers, ProbeDrift,
scikit-learn, scipy, and the OpenAI client for the long-form judge. The mean-NLL aggregate was
verified against lm-polygraph, but that package is not required at run time. The judge-comparison panel
(`scripts/checks/judge_agreement.py`) additionally uses `krippendorff` and `irrCAC`
(`pip install krippendorff irrCAC`) and degrades gracefully to `n/a` without them.

Training and evaluation data are loaded through the ProbeDrift library, kept outside this
repository.

## Two clusters: Slurm and PBSPro

Jobs run on either of two independent clusters. The Python pipeline is identical, only the
submission wrapper and a few paths differ.

- **DoC GPU cluster** — Slurm, A100 80GB. Scripts in `slurm/` (119 files).
- **RCS HPC (CX3)** — PBSPro, a larger pool (default L40S 48GB, and its A100 is a 40GB card).
  Scripts in `pbs/` (304 files).

The two directories are **not** a one-for-one mirror. They accumulated per experiment and per
cluster, so most jobs exist on one side only. Treat them as a record of what was actually
submitted rather than as a curated interface.

A single command submits to whichever cluster `LUQ_CLUSTER` names (it also auto-detects from the
hostname):

```bash
export LUQ_CLUSTER=doc   # or rcs
./scripts/submit.sh extract pubmed_qa ID     # -> sbatch on doc, qsub on rcs
```

Cluster-specific paths (repo root, `HF_HOME`, the python env) are resolved in one place:
`pbs/_env.sh` for shell, `src/luq/cluster.py` for Python (`python -m luq.cluster hf_home`).
First-time RCS setup is in **`pbs/SETUP_RCS.md`**.

| Action | Slurm | PBSPro |
|---|---|---|
| submit | `sbatch slurm/x.sbatch` | `qsub pbs/x.pbs` |
| queue | `squeue -u $USER` | `qstat -u $USER` |
| cancel | `scancel <id>` | `qdel <id>` |
| interactive | `salloc --gres=gpu:1 ...` | `qsub -I -l select=1:ngpus=1:...` |

## Reproduce the results tables

From cached features and labels, with no GPU:

```bash
mkdir -p results
python scripts/reproduce.py                 # all ID datasets, all methods
python scripts/reproduce.py --dataset xsum  # one dataset
```

Note that `scripts/reproduce.py` does **not** reproduce the full set of results. It covers the early method set only
(SAPLMA, the mean-pool probe, P(True) and Lookback on SciQ, PubMedQA and XSum) and is kept as the
original end-to-end check. The results in the write-up come from the drivers in `scripts/checks/`,
principally `probedriftlong.py` (the long-form ladder) and `assemble_pdl_table.py` (which rolls the
per-eval outputs into the master table).

Both require the cached records, which are not distributed with this repository.

## Tests and verification

```bash
pytest -q    # 93 CPU unit tests: surprisal and PRR maths, weighting, pooling, score parsing,
             # and the benchmark grid, carve and row-order guard
```

The suite pins the uncertainty maths, so a refactor that changes it fails loudly. The mean-NLL
aggregate is the same quantity lm-polygraph calls `Perplexity`, and the two were checked against
each other by hand; the tests assert the formula rather than importing that package.

Method-fidelity checks against the original authors' released code load the model, so they are
GPU scripts run by hand: `scripts/checks/check_lookback_vs_authors.py` (Lookback Lens vs Chuang
et al., to ~1e-6), `check_ptrue_verdict.py`, `check_msp_sign.py`, `check_determinism.py`, and
`judge_agreement.py` (a cheap open judge against the GPT-5 labels). These expect the authors'
reference repositories to be checked out alongside this one, and they are not vendored here.

## Status

The full long-form grid, eight evaluation sets by five in-distribution and out-of-distribution
settings at three seeds, is complete on all three base models, together with the cross-task
transfer matrix, the aggregation analysis, the published probability, distance and hybrid
comparators on Llama-3.1-8B, and the combination of CAWSA with a hidden-state probe. The result
tables the reported numbers come from are in `published_results/`, with
`scripts/report_numbers.py` to recompute the reported quantities from them.

Several results are pre-registered negatives. `prereg/` records what was predicted before each run
and `prereg/README.md` indexes them.
