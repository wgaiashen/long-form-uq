#!/bin/bash
#SBATCH --job-name=luq-extract
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract-%j.out

# GPU feature-extraction job. The model must already be downloaded on the login node
# (compute nodes have no internet). Adjust --partition / --gres to the cluster's names.
#
# Submit with:   sbatch slurm/extract.sbatch sciq ID
#   $1 = dataset (default sciq)
#   $2 = ood setting (default ID)

set -euo pipefail

DATASET="${1:-sciq}"
OOD="${2:-ID}"

# TODO: activate the conda env, then run the extract script.
# source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
# conda activate luq
# python scripts/01_extract.py --dataset "$DATASET" --ood "$OOD"

echo "Fill in env activation + the python call, then remove this line." >&2
exit 1
