#!/bin/bash
#SBATCH --job-name=attn_family
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/attn_family-%j.out

# The softmax-attention pooling FAMILY (mean -> attention+temperature -> +position -> answer-only),
# temperature selected on a validation split (never test), reported with a seed spread, plus the
# attention-weight diagnostic. Runs on the per-token cache (L15). GPU makes the temperature sweep
# fast. xsum is added once its per-token cache exists (slurm/pertoken_xsum.sbatch).
#
#   sbatch slurm/attn_family.sbatch
#   sbatch slurm/attn_family.sbatch sciq,trivia_qa,pubmed_qa,xsum   # after xsum cache

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
DATASETS=${1:-sciq,trivia_qa,pubmed_qa}

echo "=== node ==="; hostname; date; echo "datasets: $DATASETS"

echo; echo "########## attention-pooling family (L15) ##########"
python scripts/checks/attn_pool.py --model "$MODEL" --datasets "$DATASETS" \
  --layer 15 --seeds 1,2,3 --diag

echo; echo "########## competitor table: mean / last-token / per-sentence (L15) ##########"
python scripts/checks/aggregators.py --model "$MODEL" --datasets "$DATASETS" \
  --layer 15 --seeds 1,2,3

echo; echo "=== ALL DONE $(date) ==="
