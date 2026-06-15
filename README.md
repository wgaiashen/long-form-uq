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

Python 3.11 and a GPU, conda env `luq`. Main dependencies: PyTorch, transformers,
lm-polygraph, the uhead library (`luh`), ProbeDrift, scikit-learn, and the OpenAI client
(for the long-form judge).

## Status

The four methods run end-to-end on short QA (SciQ), long QA (PubMedQA), and summarisation
(XSum, CNN/DailyMail), in-distribution. Cross-task generalisation is the next step.
