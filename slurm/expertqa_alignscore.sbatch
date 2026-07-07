#!/bin/bash
#SBATCH --job-name=eqa-align
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/expertqa_alignscore-%j.out

# AlignScore-vs-judge read on the labelled ExpertQA set (the 4th gate read at full scale).
# Reads the labels already in the records; scores AlignScore(gen,gold) on GPU and correlates with
# the stored faithfulness + uncovered. Writes results/expertqa/alignscore_vs_faithfulness.txt.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export NLTK_DATA=/vol/gpudata/gs925-msc_project/nltk_data   # punkt_tab for AlignScore's sent_tokenize
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "expertqa_alignscore start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PYTHONPATH=src python -u scripts/checks/alignscore_vs_faithfulness.py --prompt-regime expertqa_rp12
echo "expertqa_alignscore DONE at $(date)"
