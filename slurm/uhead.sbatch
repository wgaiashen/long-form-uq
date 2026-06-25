#!/bin/bash
#SBATCH --job-name=uhead
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/uhead-%j.out

# UHead baseline for all three Gemma datasets, end to end:
#   01d_uhead  (replay cached records through Gemma + the pretrained uhead head ->
#               one uncertainty score per TEST example; trains nothing of ours)
#   04_eval    (adds the `uhead` column to the results CSV and prints its PRR)
#
# Prereqs already satisfied:
#   - 01_extract has run for these datasets (cache/records/...gemma-2-9b-it__*__ID.jsonl exist)
#   - the head is pre-downloaded into HF_HOME (llm-uncertainty-head/uhead_gemma-2-9b-it)
#   - 02_label has run, so 04_eval has correctness for the PRR
#
# a100 (80GB): the head needs the EAGER attention backend (Gemma-2 soft-capping) with
# all-layer attentions in memory, same headroom reason as gemma_gpu.sbatch. Idempotent:
# 01d overwrites its score file and 04_eval re-reads, so resubmitting is safe.
#
#   sbatch slurm/uhead.sbatch

# No `set -e`: one dataset failing must not abort the rest.
set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=google/gemma-2-9b-it
OOD=ID

echo "uhead start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in sciq pubmed_qa xsum; do
  echo "================ $DS : 01d_uhead ================ $(date)"
  if ! python scripts/01d_uhead.py --dataset "$DS" --ood "$OOD" --model "$MODEL"; then
    echo "!!! 01d_uhead $DS FAILED -- skipping eval for $DS"; continue
  fi
  echo "================ $DS : 04_eval ================ $(date)"
  python scripts/04_eval.py --dataset "$DS" --ood "$OOD" --model "$MODEL" \
    || echo "!!! 04_eval $DS FAILED"
done

echo "uhead ALL DONE at $(date)"
