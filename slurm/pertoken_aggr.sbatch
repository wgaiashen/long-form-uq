#!/bin/bash
#SBATCH --job-name=pertok-aggr
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pertok_aggr-%j.out
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
python scripts/checks/pertoken_aggregate.py --model meta-llama/Meta-Llama-3.1-8B \
  --datasets sciq,trivia_qa,pubmed_qa --layer 15
echo "=== DONE $(date) ==="
