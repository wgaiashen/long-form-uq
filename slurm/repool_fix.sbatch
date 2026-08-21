#!/bin/bash
#SBATCH --job-name=repool-fix
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/repool_fix-%j.out

# Trivia fix: re-pool SAPLMA features with the corrected pooling (last-prompt/pre-answer
# position included, matching the masked-mean). Re-pools from cached token IDs via
# recompute_states -- NO regeneration, so the Tier-1 records and their gpt-5 labels are
# untouched and NO re-labelling is needed. Applies to sciq + trivia_qa (short answers,
# where the old pooling dropped the informative pre-answer state). pubmed is NOT here: its
# gap is D1 (newline-stop), which changes the text and needs re-labelling -- handled separately.
# Also caches middle-layer per-token window states (--cache-per-token) so future aggregation
# experiments are free CPU re-pools.
#
#   sbatch slurm/repool_fix.sbatch
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B
OOD=ID
LOG=logs/repool_fix.log
echo "=== repool fix start $(date) ===" > "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$LOG" 2>&1 || true

for DS in trivia_qa sciq; do
  echo "################ $DS re-pool $(date) ################" >> "$LOG"
  python scripts/01e_repool.py --model "$MODEL" --dataset "$DS" --ood "$OOD" \
    --dtype fp32 --attn eager --cache-per-token >> "$LOG" 2>&1 || { echo "!!! repool $DS FAILED" >> "$LOG"; continue; }
  for L in 16 15; do
    python scripts/03_probe.py --model "$MODEL" --dataset "$DS" --ood "$OOD" \
      --method saplma --saplma-batch 1 --layer "$L" >> "$LOG" 2>&1
    echo "===== EVAL $DS LAYER $L (corrected pooling) =====" >> "$LOG"
    python scripts/04_eval.py --model "$MODEL" --dataset "$DS" --ood "$OOD" 2>&1 | grep -E "saplma|MSP min" >> "$LOG"
  done
  echo "[$(date)] DONE $DS" >> "$LOG"
done
echo "=== repool fix ALL DONE $(date) ===" >> "$LOG"
