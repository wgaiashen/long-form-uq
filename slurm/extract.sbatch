#!/bin/bash
#SBATCH --job-name=luq-extract
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract-%j.out

# GPU feature-extraction job. The model must already be downloaded on the login node
# (or HF_HOME pre-populated) since compute nodes are not guaranteed internet. The
# partition IS the GPU type on this cluster: t4 | a16 | a30 | a40 | a100. t4 suits
# the 1.5B dev model; a40/a100 for 7-9B (Gemma-2-9B needs the headroom, and the
# Lookback fp32 path needs ~36GB).
#
# 01_extract.py checkpoints every 200 examples and resumes from cache, so if this
# job still hits the 12h wall just submit it again with the same args -- it picks up
# where it left off instead of restarting.
#
# Submit with:   sbatch slurm/extract.sbatch <dataset> <ood> <model>
#   $1 = dataset (default sciq)
#   $2 = ood setting (default ID)
#   $3 = model    (default google/gemma-2-9b-it)

set -euo pipefail

DATASET="${1:-sciq}"
OOD="${2:-ID}"
MODEL="${3:-google/gemma-2-9b-it}"

# Keep all HF downloads/caches on the big volume; home is over quota.
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache

source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

echo "extract: dataset=$DATASET ood=$OOD model=$MODEL on $(hostname)" >&2
python scripts/01_extract.py --dataset "$DATASET" --ood "$OOD" --model "$MODEL"
