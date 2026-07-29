#!/bin/bash
#SBATCH --job-name=s3_factscore
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fixed_prior_ladder_factscore-%j.out

# STEP A: factscore's 3 cells as their OWN job, run concurrently with S3 (fixed_prior_ladder is
# per-eval). IDENTICAL config to S3 in every dimension except --evals (same commit, seeds 1,2,3,
# rungs DiffTask,LOO, priors content_mass,nll). SEPARATE --out slug so it can NEVER write S3's path.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== node ==="; hostname; date; echo "commit $(git rev-parse HEAD)"

python -u scripts/checks/fixed_prior_ladder.py \
  --evals factscore \
  --rungs DiffTask,LOO \
  --priors content_mass,nll --seeds 1,2,3 \
  --out results/fixed_prior_ladder_factscore__meta-llama_Meta-Llama-3.1-8B.csv

echo "=== ALL DONE $(date) ==="
