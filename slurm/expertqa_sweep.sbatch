#!/bin/bash
#SBATCH --job-name=eqa-sweep
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/expertqa_sweep-%j.out

# Decoding-config sweep on a stratified 50-example subset (5 configs) to find a config that stops
# the no_repeat_ngram-induced degeneration WITHOUT trading it for fluent-off-topic, BEFORE
# regenerating all 2016. Scored by the validated list-aware detector (luq.degeneracy). Writes a
# summary + a side-by-side HTML to results/expertqa/decoding_sweep.{json,html}. No cache writes.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "expertqa_sweep start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u scripts/tools/sweep_decoding.py --n 50 --out results/expertqa/decoding_sweep
echo "expertqa_sweep DONE at $(date)"
