# Long-Form Uncertainty Estimation for LLMs

MSc Computing project, Imperial College London.
Author: Gaia Shen. Supervisors: Joe Stacey and Lihu Chen.

## What this is

This project investigates white-box uncertainty quantification for long-form LLM
generation. The aim is to train a probe on a model's internal states that produces one
uncertainty score for a whole long-form output, and to test whether a single probe
generalises across long-form task types (factuality and faithfulness) rather than being
tied to one.

## Approach

1. Generate long-form outputs from a frozen base model.
2. Label each output for correctness (string match for short QA; an LLM judge for long-form).
3. Train a probe on internal features to predict correctness.
4. Evaluate with the prediction-rejection ratio (PRR).

Methods compared: MSP (unsupervised token probability), SAPLMA (hidden-state probe),
P(True) (an appended self-verification probe), and Lookback Lens (an attention-faithfulness
probe). A method is just a feature extractor; the rest of the pipeline is shared.

## Pipeline (`scripts/`)

1. `01_extract.py` - generate, then cache the records and pooled hidden-state features.
2. `01b_ptrue.py`, `01c_lookback.py` - extra per-method features from the cached records.
3. `02_label.py` - correctness labels.
4. `03_probe.py` - train the probe on cached features.
5. `04_eval.py` - PRR table.

## Repository layout

- `src/luq/` - the pipeline library (data, generation, features, labels, probe, eval).
- `scripts/` - the pipeline stages above, plus `checks/` (verification scripts) and
  `tools/` (helpers, e.g. a parallel labeller).
- `slurm/` - batch scripts for the GPU cluster.

Training and evaluation data are loaded through the ProbeDrift library, kept outside this
repository. Model weights, caches, and generated outputs are not tracked in Git.

## Setup

Python 3.11, conda env `luq`. Heavy/GPU steps run on a real compute node via Slurm (not the
cloud VM, which is an 8 GB CPU jump box):

```bash
conda activate luq
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache    # NOT ~/.cache (home is over quota)
export OPENAI_API_KEY=...                                 # only for the long-form judge (02_label)
salloc --partition=a30 --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=01:00:00   # for GPU stages
```

Main dependencies: PyTorch, transformers, lm-polygraph, the uhead library (`luh`), ProbeDrift,
scikit-learn, and the OpenAI client (for the long-form judge).

## Reproduce the results tables

From the cached features and labels (no GPU), regenerate every supervised score and PRR table:

```bash
python scripts/reproduce.py                 # all ID datasets, all methods
python scripts/reproduce.py --dataset xsum  # one dataset
```

The printed PRRs match the tables in `../worklog.md`. Each method uses its default middle layer
(SAPLMA/P(True) → 14, Lookback → 0).

## Tests and verification

```bash
pytest    # CPU unit tests: the MSP and PRR maths, msp_nll == lm-polygraph Perplexity, score parsing
```

Method-fidelity checks against the original authors' code load the model, so they are GPU scripts
run by hand: `check_lookback_vs_authors.py` (lookback vs `lookback-src/`, Chuang et al., to ~1e-6),
`check_ptrue_verdict.py`, `check_msp_sign.py`, `check_determinism.py`, and `judge_agreement.py`
(a cheap/open judge vs the GPT-5 labels).

## Status

The four methods run end-to-end on short QA (SciQ), long QA (PubMedQA), and summarisation (XSum),
in-distribution. (CNN/DailyMail was dropped — only partially labelled.) Cross-task generalisation
is the next step.
