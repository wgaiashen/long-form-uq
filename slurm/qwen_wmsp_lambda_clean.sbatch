#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_wlam_clean
#SBATCH --output=logs/qwen_wmsp_lambda_clean_%j.out

# wMSP-shrink@1.5 transfer on the CLEAN-SPAN population. Sibling of slurm/qwen_wmsp_lambda.sbatch
# (read that one for the full cost/partition reasoning -- unchanged here). Two differences:
#
#   1. LUQ_REGIME points samsum/med_quad/expertqa/factscore at their clean-span namespaces (the
#      2026-08-11/12 correction); pubmed_qa/asqa/xsum/cnn_dailymail resolve to canonical.
#   2. --tag qwenclean and --master-csv point wmsp_lambda_qwen.py's outputs and its lambda=2.0
#      outside-control gate at the CLEAN population instead of the raw one -- the raw-span
#      wmsp_lambda_qwen__*.csv results (already backed up to results/prev/2026-08-11/) are never
#      at risk, and the gate compares against the population this run actually used
#      (results/analysis/pdl_master_qwenclean__Qwen_Qwen2.5-14B.csv) rather than one it was never
#      meant to match.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export LUQ_REGIME="samsum=trunc_v1,med_quad=trunc_v1,expertqa=expertqa_rp12_trunc_v1,factscore=factscore_rp12_trunc_v1"
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  LUQ_REGIME=$LUQ_REGIME"

MODEL="Qwen/Qwen2.5-14B"
LAYER=23
MASTER="results/analysis/pdl_master_qwenclean__Qwen_Qwen2.5-14B.csv"

EVALS=("$@")
if [ ${#EVALS[@]} -eq 0 ]; then
  EVALS=(factscore asqa samsum med_quad expertqa xsum cnn_dailymail pubmed_qa)
fi
LIST=$(IFS=,; echo "${EVALS[*]}")

echo "=== wmsp lambda CLEAN-SPAN [$LIST] (L$LAYER) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

python -u scripts/checks/wmsp_lambda_qwen.py --model "$MODEL" --layer "$LAYER" \
    --evals "$LIST" --seeds 1,2,3 --tag qwenclean --master-csv "$MASTER" \
    || { echo "!!! wmsp lambda CLEAN-SPAN FAILED for [$LIST]"; exit 1; }

echo "=== wmsp lambda CLEAN-SPAN [$LIST] done at $(date) ==="
python -u scripts/checks/wmsp_lambda_qwen.py --gate-only --tag qwenclean --master-csv "$MASTER" || true
