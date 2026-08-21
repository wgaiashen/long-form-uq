#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_clean
#SBATCH --output=logs/qwen_clean_sensitivity_%j.out

# Clean-vs-flagged sensitivity on the two hardest OOD rungs, both models.
#
#   sbatch slurm/qwen_clean_sensitivity.sbatch                                  # Qwen
#   sbatch slurm/qwen_clean_sensitivity.sbatch meta-llama/Meta-Llama-3.1-8B     # Llama
#
# WHY. ExpertQA's whole length-label relationship turned out to be produced by degenerate/bleed
# generations -- on clean rows rho(len,label) goes -0.577 -> -0.040 and a length-only rule goes
# PRR +0.666 -> -0.060, and the same collapse happens on Llama. This asks the same question of the
# SUPERVISED comparison: is wMSP@2-vs-SAPLMA on the hardest rungs also partly degeneracy detection?
#
# DIAGNOSTIC, NOT A CORRECTION. Writes its own CSV under results/analysis/. It does not touch
# pdl_master, drops no example from the canonical population, and the CLEAN column is a different,
# easier population (expertqa mean label 0.896 clean vs 0.652 overall) rather than a fixed score.
# The informative quantity is the collapse, not the level.
#
# Two rungs instead of five and two methods instead of fifteen, so this is roughly a fifth of a
# ladder run: expect ~1-2 h. Same pools, same draws, same seeds, same kwargs as probedriftlong, so
# the ALL column is directly checkable against the master.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  LUQ_CARVE=${LUQ_CARVE:-legacy}"

MODEL="${1:-Qwen/Qwen2.5-14B}"
case "$MODEL" in
  Qwen/Qwen2.5-14B)               LAYER=23 ;;
  meta-llama/Meta-Llama-3.1-8B)   LAYER=15 ;;   # each model's own ceil(N/2)-1, never swapped
  *) echo "REFUSING '$MODEL': no fixed layer for it"; exit 2 ;;
esac

echo "=== clean sensitivity $MODEL (L$LAYER) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/checks/qwen_clean_sensitivity.py --model "$MODEL" --layer "$LAYER" --seeds 1,2,3 \
    || { echo "!!! clean sensitivity FAILED for $MODEL"; exit 1; }

echo "=== done at $(date) ==="
