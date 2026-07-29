#!/bin/bash
#SBATCH --job-name=s3_prior_ladder
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fixed_prior_ladder-%j.out

# S3: prior-init pooling, arms A/B/C/D on the long-form OOD rungs, priors content_mass + nll.
# All arms trained in THIS run (guard 5b); arm-A reproduction gate vs the synced §C.3 CSVs (5c);
# cluster/env_hash provenance columns (5d). GPU (per-cell temperature sweep + pooler training).
# Reads pertok (bitbucket symlinks) + records (token_logprobs) + results/probedriftlong_*.csv (gate).
#
#   sbatch slurm/fixed_prior_ladder_s3.sbatch
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== node ==="; hostname; date

python -u scripts/checks/fixed_prior_ladder.py \
  --evals pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa,factscore \
  --rungs DiffTask,LOO \
  --priors content_mass,nll --seeds 1,2,3

echo "=== ALL DONE $(date) ==="
