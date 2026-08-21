#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_wlam
#SBATCH --output=logs/qwen_wmsp_lambda_%j.out

# wMSP shrinkage lambda = 1.5 on Qwen2.5-14B, with a full paired lambda = 2 control.
# One array task per eval, 3 seeds, the COMPLETE cells_long grid.
#
#   sbatch --mem=96G slurm/qwen_wmsp_lambda.sbatch
#
# POST-HOC CROSS-MODEL TRANSFER, NOT A REGISTERED TEST. The Qwen master was already visible when
# this was run. lambda is FIXED at 2.0 and 1.5 (Llama's W5 pre-committed primary) and nothing is
# tuned on Qwen. Results land in results/wmsp_lambda_qwen__* with provenance = post-hoc-transfer and
# stay OUTSIDE the M2 scorecard. Read the driver's docstring before quoting any number: W5's
# registered claim FAILED on Llama as a TIE with shrink@2 (+0.0019 vs a > +0.010 bar); what
# justifies the transfer is that the 5-arm LODO picks 1.5 on 7 of 8 folds.
#
# THE CONTROL GATES EVERYTHING. The lambda = 2.0 arm must reproduce the master's wmsp_shrink2
# rows on all 40 cells to 4 dp. Each cell prints its own control delta as it lands, so a drift shows
# up on the first cell rather than after eight hours. When all 8 evals are done:
#
#     python scripts/checks/wmsp_lambda_qwen.py --gate-only
#
# Until that passes, lambda = 1.5 is NOT interpretable.
#
# WHY a30 AND 96G. Same shape as the ladder: RAM-bound on the per-token pools (Qwen's are 3-6.5 GB
# each), GPU only for the weighter's training. Measured ladder RSS was ~31 GB with all pools loaded,
# and this holds the same pools with fewer methods. a30 has 515 GB and 8 cards.
#
# COST. The full 15-method ladder ran ~4 h on pubmed_qa (03:08 -> 07:11). This trains 2 wMSP arms
# instead of 8, plus the ~20 min pool load, so expect ~1-1.5 h per eval => ~6-8 h wall at 3
# concurrent. Per-cell atomic flush means a walltime kill costs the cell in flight, not the eval.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  LUQ_CARVE=${LUQ_CARVE:-legacy}"

MODEL="Qwen/Qwen2.5-14B"
LAYER=23

# ONE JOB LOOPING THE EVALS, NOT AN 8-TASK ARRAY. Two reasons, both learned the hard way here:
#   1. DoC counts an array task as a submission, so an 8-task array is rejected outright with
#      QOSMaxSubmitJobPerUserLimit whenever a few other jobs are queued. It was.
#   2. The driver loads the per-token pools ONCE per invocation and then loops `--evals`. An array
#      would pay that ~4 min load eight times over for no benefit.
# Cheapest first, so a walltime kill lands on the most expensive eval with the others banked; the
# driver writes one CSV per eval and flushes per cell, so a kill costs the cell in flight only.
EVALS=("$@")
if [ ${#EVALS[@]} -eq 0 ]; then
  EVALS=(factscore asqa samsum med_quad expertqa xsum cnn_dailymail pubmed_qa)
fi
LIST=$(IFS=,; echo "${EVALS[*]}")

echo "=== wmsp lambda [$LIST] (L$LAYER) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

python -u scripts/checks/wmsp_lambda_qwen.py --model "$MODEL" --layer "$LAYER" \
    --evals "$LIST" --seeds 1,2,3 \
    || { echo "!!! wmsp lambda FAILED for [$LIST]"; exit 1; }

echo "=== wmsp lambda [$LIST] done at $(date) ==="
python -u scripts/checks/wmsp_lambda_qwen.py --gate-only || true
