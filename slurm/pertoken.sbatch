#!/bin/bash
#SBATCH --job-name=pertoken
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pertoken-%j.out

# Re-extract per-token hidden states at layer 15 (the verified SAPLMA window) for the three
# datasets that have a cache, then run the per-token aggregation experiment. The earlier per-token
# cache was stale (its mean-pool gave 0.45 vs the real SAPLMA 0.90), so this rebuilds it correctly;
# 01h prints a SANITY line per dataset (mean-pool of the new per-token must match the cached SAPLMA).
#
#   sbatch slurm/pertoken.sbatch

set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B

echo "=== node ==="; hostname; date

echo; echo "########## 1. Re-extract per-token states at L15 ##########"
for d in sciq trivia_qa pubmed_qa; do
  echo "---- $d ----"
  python scripts/01h_pertoken.py --model "$MODEL" --dataset "$d" --layer 15
done

echo; echo "########## 2. Per-token aggregation experiment (L15) ##########"
python scripts/checks/pertoken_aggregate.py --model "$MODEL" --datasets sciq,trivia_qa,pubmed_qa --layer 15

echo; echo "=== ALL DONE $(date) ==="
