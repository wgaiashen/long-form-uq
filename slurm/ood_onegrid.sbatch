#!/bin/bash
#SBATCH --job-name=ood-onegrid
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ood_onegrid-%j.out

# ONE-HARNESS OOD rigour run: every method (attention/uniform/per-token/per-sentence/last-token/
# mean-pool+MLP + lookback/ptrue_accurate/linear) through the PINNED ProbeDrift-light pools,
# {ID, LOO, DiffTask}, PAIRED seeds (one subsample per seed shared across methods),
# ID-diagonal-gated. Retires the two ad-hoc grids. GPU for the attention pooler; 128G for the
# per-token caches. Judge label throughout.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export NLTK_DATA=/vol/gpudata/gs925-msc_project/nltk_data
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

mkdir -p "$NLTK_DATA"
python -c "import nltk; nltk.download('punkt_tab', download_dir='$NLTK_DATA'); nltk.download('punkt', download_dir='$NLTK_DATA')" || true

echo "ood_onegrid start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PYTHONPATH=src:scripts/checks python scripts/checks/ood_onegrid.py --seeds 1,2,3,4,5
echo "ood_onegrid DONE at $(date)"
