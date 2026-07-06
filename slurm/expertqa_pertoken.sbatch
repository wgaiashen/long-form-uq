#!/bin/bash
#SBATCH --job-name=eqa-pertok
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/expertqa_pertoken-%j.out

# ExpertQA FULL-SET per-token hidden states at layer 15 (the verified SAPLMA window).
# Reads the cached token IDs from the full-set generation (expertqa_reppen namespace, ood ID)
# and re-runs a teacher-forced forward, caching the [P-1, P+G] per-token L15 states.
#   - Needs the generation job (eqa-full) to have finished first (reads its records + SAPLMA feats).
#   - Does NOT need the judge labels: 01h's SANITY line checks that the mean-pool of the new
#     per-token states matches the cached SAPLMA feature (a feature check, not a label check).
#   - fp32 + eager to match the full-set generation. Per EXPERTQA_PLAN the per-token forward is
#     LESS memory-risky than Xsum (short questions; the lever is generation length G<=384), so
#     an a40 (48GB) has comfortable headroom for fp32 8B at a single layer.
#
#   sbatch slurm/expertqa_pertoken.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
echo "expertqa_pertoken start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01h_pertoken.py --model "$MODEL" --dataset expertqa --ood ID \
  --prompt-regime expertqa_reppen --layer 15 --dtype fp32 --attn eager
echo "expertqa_pertoken DONE at $(date)"
