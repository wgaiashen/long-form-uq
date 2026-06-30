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
scikit-learn, scipy, and the OpenAI client (for the long-form judge). The judge-comparison panel
(`scripts/checks/judge_agreement.py`) additionally uses `krippendorff` and `irrCAC`
(`pip install krippendorff irrCAC`); it degrades gracefully (those metrics print `n/a`) if they
are not installed.

## Two clusters: DoC (Slurm) and RCS (PBSPro)

Jobs can run on either of two independent clusters; the Python pipeline is identical, only
the submission wrapper and a few paths differ.

- **DoC GPU cluster (CSG)** — Slurm, A100 **80GB**. The existing `slurm/*.sbatch` workflow.
- **RCS HPC (CX3 Phase 2)** — PBSPro, a larger college pool (default **L40S 48GB**; A100 is
  40GB and scarce). Scripts in `pbs/`, mirroring `slurm/` one-for-one.

A single command submits to whichever cluster you set with `LUQ_CLUSTER` (it also
auto-detects from the hostname):

```bash
export LUQ_CLUSTER=doc   # or rcs
./scripts/submit.sh extract pubmed_qa ID     # -> sbatch on doc, qsub on rcs
```

The cluster-specific paths (repo root, `HF_HOME`, the python env) are resolved in one place —
`pbs/_env.sh` for shell, `src/luq/cluster.py` for Python (`python -m luq.cluster hf_home`).
First-time RCS setup is in **`pbs/SETUP_RCS.md`**. Keep the heaviest job (Gemma-2-9B + the
fp32 Lookback recompute) on DoC's 80GB; use RCS for the many small parallel OOD jobs.

Slurm ↔ PBS command map:

| Action | DoC (Slurm) | RCS (PBSPro) |
|---|---|---|
| submit | `sbatch slurm/x.sbatch` | `qsub pbs/x.pbs` |
| queue | `squeue -u $USER` | `qstat -u $USER` |
| cancel | `scancel <id>` | `qdel <id>` |
| interactive | `salloc --gres=gpu:1 ...` | `qsub -I -l select=1:ngpus=1:...` |

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
