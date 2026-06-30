#!/bin/bash
#SBATCH --job-name=llama-keystone
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/llama_keystone-%j.out

# Phase-1 keystone reproduction: Llama-3.1-8B SAPLMA vs Hidden Failures Table 14.
# GPU feature extraction ONLY (01_extract -> generation + SAPLMA pooled features),
# for the three keystone datasets. fp32 + eager to match Joe (he passes no
# torch_dtype; --attn eager). RESUMABLE: 01_extract reloads its checkpoint, so on a
# wall-time hit just resubmit.
#
# PREREQUISITES (login node, one time):
#   - accept the meta-llama license on HF and set HF_TOKEN, then pre-download
#     SAFETENSORS ONLY (the --exclude drops the redundant 16GB original/ checkpoint;
#     the 50GB project quota will not hold both formats):
#       hf download meta-llama/Meta-Llama-3.1-8B --exclude "original/*"
#
#   sbatch slurm/llama_keystone.sbatch
#
# After this finishes (per dataset): label with the FULL gpt-5 judge on the login node
#   python scripts/02_label.py --model meta-llama/Meta-Llama-3.1-8B --dataset <ds> --ood ID \
#       --judge gpt-5-2025-08-07 --judge-short-form --promote-judge
# then G2 (judge_distribution.py), then probe (03_probe --method saplma --saplma-batch 1
# --layer 15, sweep 16) and 04_eval.

# NOTE: no `set -e` -- one dataset failing must not abort the rest.
set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B
OOD=ID

echo "llama_keystone start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Order: sciq + pubmed_qa first (the SAPLMA-ID and the PubmedQA headline cell), then
# trivia_qa LAST -- its raw HF dataset download is the largest/least-known, so if it
# strains the 50GB quota the two most important cells are already cached.
for DS in sciq pubmed_qa trivia_qa; do
  echo "================ $DS : 01_extract (fp32 + eager) ================ $(date)"
  if ! python scripts/01_extract.py --dataset "$DS" --ood "$OOD" --model "$MODEL" \
        --dtype fp32 --attn eager; then
    echo "!!! extract $DS FAILED -- skipping rest of $DS"; continue
  fi
  touch "cache/.gpu_done__${DS}__${OOD}"
  echo "================ $DS : GPU FEATURES DONE ================ $(date)"
done

echo "llama_keystone ALL DONE at $(date)"
