#!/bin/bash
#SBATCH --job-name=gemma-gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/gemma_gpu-%j.out

# All GPU feature extraction for the Gemma-2-9B migration, all three datasets:
#   01_extract  (generation + SAPLMA pooled features; RESUMABLE)
#   01b_ptrue   (P(True) verdict-position features)
#   01c_lookback(Lookback attention-ratio features)
# Writes a sentinel cache/.gpu_done__<ds>__ID after each dataset so the CPU-side
# orchestrator (run_gemma_cpu_pipeline.sh) knows it can label/probe/eval that dataset.
#
# a100 (80GB) on purpose: Gemma-2 needs the EAGER attention backend (soft-capping),
# and eager + the fp32 Lookback pass need the headroom a40 doesn't reliably give.
# If the 20h wall is hit, just resubmit -- 01_extract resumes from its checkpoint and
# the other steps are idempotent.
#
#   sbatch slurm/gemma_gpu.sbatch

# NOTE: no `set -e` -- one dataset/step failing must not abort the rest.
set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=google/gemma-2-9b-it
OOD=ID

echo "gemma_gpu start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in sciq pubmed_qa xsum; do
  echo "================ $DS : 01_extract ================ $(date)"
  if ! python scripts/01_extract.py --dataset "$DS" --ood "$OOD" --model "$MODEL"; then
    echo "!!! extract $DS FAILED -- skipping rest of $DS"; continue
  fi
  echo "================ $DS : 01b_ptrue ================ $(date)"
  python scripts/01b_ptrue.py --dataset "$DS" --ood "$OOD" --model "$MODEL" \
    || echo "!!! ptrue $DS FAILED"
  echo "================ $DS : 01c_lookback ================ $(date)"
  python scripts/01c_lookback.py --dataset "$DS" --ood "$OOD" --model "$MODEL" \
    || echo "!!! lookback $DS FAILED"

  touch "cache/.gpu_done__${DS}__${OOD}"
  echo "================ $DS : GPU FEATURES DONE ================ $(date)"
done

echo "gemma_gpu ALL DONE at $(date)"
