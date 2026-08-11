#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_trhid
#SBATCH --output=logs/qwen_trunc_hidden_%j.out

# Step 2b hidden-state half: FROZEN-PROBE EVAL-SIDE truncation sensitivity.
# Trains once per cell on FULL states, scores the same model on full vs truncated test spans.
# Labels raw throughout. Boundary frozen in luq.template_restart before this ran.
#   sbatch slurm/qwen_trunc_hidden.sbatch [model]
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
MODEL="${1:-Qwen/Qwen2.5-14B}"
case "$MODEL" in
  Qwen/Qwen2.5-14B)             LAYER=23 ;;
  meta-llama/Meta-Llama-3.1-8B) LAYER=15 ;;
  *) echo "REFUSING '$MODEL': no fixed layer"; exit 2 ;;
esac
echo "=== trunc-hidden $MODEL (L$LAYER) on $(hostname) at $(date) ==="
# --skip-poolers: select_temperature dominates the runtime (40 min without emitting a cell) and
# uniform/attention are secondary to the SAPLMA-vs-wMSP question this sensitivity exists to answer.
python -u scripts/checks/truncation_sensitivity_hidden.py --model "$MODEL" --layer "$LAYER" --seeds 1,2,3 --skip-poolers
echo "=== done at $(date) ==="
