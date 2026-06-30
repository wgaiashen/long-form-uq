#!/bin/bash
#SBATCH --job-name=diag-pertok
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diag_pertok-%j.out
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
for d in sciq pubmed_qa; do
  echo "==== $d ===="
  python scripts/checks/diag_pertoken_vs_feature.py --model meta-llama/Meta-Llama-3.1-8B --dataset $d --layer 15 --n 30
done
echo "=== DONE $(date) ==="
