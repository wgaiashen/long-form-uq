# Long-Form Uncertainty Estimation for LLMs

MSc Computing project, Imperial College London.
Author: Gaia Shen. Supervisors: Joe Stacey and Lihu Chen.

## What this is

This project investigates white-box uncertainty quantification for long-form LLM
generation. The aim is to train a probe on a model's internal states that produces one
uncertainty score for a whole long-form output, and to test whether a single probe
generalises across long-form task types (factuality and faithfulness) rather than being
tied to one.

## Approach (short version)

1. Generate long-form outputs from a frozen base model.
2. Label each output for correctness (string match for short QA; an LLM judge for long-form).
3. Train a probe on hidden-state features to predict correctness.
4. Evaluate with the prediction-rejection ratio (PRR), in-distribution and out-of-distribution.

Baselines: MSP (unsupervised) and SAPLMA (supervised), with task-specific baselines added later.

## Repository layout

- `test.py` - minimal end-to-end check (load data, generate, reach hidden states).
- (more code is added as the project develops.)

Training and evaluation data are loaded through the ProbeDrift library, which is kept outside
this repository. Model weights, caches, and generated outputs are not tracked in Git.

## Setup

Requires Python 3.11 and a GPU. Uses the conda env `luq`. Main dependencies: PyTorch,
transformers, lm-polygraph, the uhead repository (`luh`), ProbeDrift, and scikit-learn. The PRR
scorer `robust-uq-eval` is not installed yet. See the project notes for the full environment
setup on the Imperial Computing GPU cluster.

## Status

Early stage. Current milestone: a SAPLMA baseline running end-to-end on a short-form dataset.
