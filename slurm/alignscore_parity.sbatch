#!/bin/bash
#SBATCH --job-name=align-parity
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/alignscore_parity-%j.out

# BUG-HUNT item 1 (GPU-gated): numerically confirm our vendored AlignScore scorer == the
# lm_polygraph_lite AlignScore on real cached records. the scorer calls cuda.synchronize()
# unconditionally, so this cannot run CPU-only; it needs a GPU. Loads only the RoBERTa-large
# AlignScore model (NOT the base Llama), so a small GPU is plenty. Reads cached records; writes
# NOTHING to the cache. Upgrades item 1 from "NOT-BUG by code inspection" to "verified numerically".

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "alignscore_parity start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# AlignScore's sent_tokenize needs NLTK 'punkt_tab' (newer NLTK; the scorer only auto-pulls 'punkt').
# Download to the big volume (home is over quota) and point NLTK at it.
export NLTK_DATA=/vol/gpudata/gs925-msc_project/nltk_data
mkdir -p "$NLTK_DATA"
python -c "import nltk; nltk.download('punkt_tab', download_dir='$NLTK_DATA'); nltk.download('punkt', download_dir='$NLTK_DATA')"

# 100 (output, gold) pairs spread across sciq/trivia/pubmed/xsum; asserts ours==reference to <1e-3.
PYTHONPATH=src python scripts/checks/alignscore_vs_authors.py --n 100
echo "alignscore_parity DONE at $(date)"
