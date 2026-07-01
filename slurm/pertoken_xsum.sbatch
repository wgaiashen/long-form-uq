#!/bin/bash
#SBATCH --job-name=pertok_xsum
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pertok_xsum-%j.out

# Extract the xsum per-token hidden states at layer 15 (the long-form unblock for the aggregation
# set), then run the full aggregation table on all four datasets. Submit only AFTER the xsum judge
# labels finish (scripts/02_label.py), because 01h's sanity gate and aggregators.py both need
# r["correctness"]. xsum prompts are long articles, so extraction is the heaviest 01h run.
#
# Runs on the A100 (80GB): keeping fp32 + eager (so the per-token states still match the SAPLMA
# feature the gate checks) means ~32GB for the fp32 8B model, which left too little for the long
# xsum prompts on the 44GB A40 and OOM'd at ~3300/3800 (job 255268). The A100 has the headroom.
#
#   sbatch slurm/pertoken_xsum.sbatch

set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B

echo "=== node ==="; hostname; date

echo; echo "########## 1. Extract xsum per-token states at L15 ##########"
python scripts/01h_pertoken.py --model "$MODEL" --dataset xsum --layer 15

echo; echo "########## 2. Aggregation set on all four datasets (L15) ##########"
python scripts/checks/aggregators.py --model "$MODEL" \
  --datasets sciq,trivia_qa,pubmed_qa,xsum --layer 15 --seeds 1,2,3

echo; echo "=== ALL DONE $(date) ==="
