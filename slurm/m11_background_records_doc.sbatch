#!/bin/bash
#SBATCH --job-name=m11-bgrec
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/m11_bgrec-%j.out
# Fallback only. The background token corpus is normally rebuilt on the other cluster, where the
# states file it is rebuilt from already lives, and the resulting 40 MB file is copied here. Use this
# only if that file cannot be copied, in which case the corpus is GENERATED rather than rebuilt and
# the two clusters are then fitting backgrounds on samples that follow the same recipe but were
# produced by two different runs. Say so if that happens; do not let it pass silently.
#
# A GPU is requested because every partition here assigns one whether or not it is asked for.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== background records on $(hostname) at $(date) ==="
python -u scripts/01q_background_stats.py --model meta-llama/Meta-Llama-3.1-8B \
    --emit-records --budget 384 --dtype fp32 --attn eager --device-map cuda \
    || { echo "!!! FAILED"; exit 3; }
echo "=== done at $(date) ==="
