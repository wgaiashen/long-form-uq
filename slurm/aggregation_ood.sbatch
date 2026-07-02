#!/bin/bash
#SBATCH --job-name=agg-ood
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/aggregation_ood-%j.out
# ProbeDriftLight OOD for the aggregation family (train pooled source, test held-out), OLD prompts
# (matches the ID table), existing labels, no spend. Built-in ID-diagonal gate aborts early if the
# cross-dataset wiring is wrong. GPU (attention training). -u for live progress.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "agg-ood start on $(hostname) at $(date)"
python -u scripts/checks/aggregation_ood.py --model meta-llama/Meta-Llama-3.1-8B --seeds 1,2,3
echo "agg-ood DONE at $(date)"
