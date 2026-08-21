#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_pertok
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_pertoken_%A_%a.out

# The CANONICAL Qwen2.5-14B per-token cache: 01h_pertoken at LAYER 23, fp32, one dataset per task.
#
# LAYER 23, BY THE FIXED RULE, DECIDED BY THE AUTHOR 2026-08-09. The fixed protocol is the
# middle-layer convention ceil(N/2)-1 — the rule the Llama grid ACTUALLY consumed (every Llama
# pertok file on disk is __L15 = ceil(32/2)-1, produced by 01h_pertoken fp32; verified from the
# files and the Llama sbatch precedents expertqa_pertoken/asqa_pertoken/cnn_dailymail_pertoken_doc).
# ceil(48/2)-1 = 23 on Qwen. NOT 24: (n_layers+1)//2 is 01e_repool's side-dump formula, which no
# consumed Llama artifact used — the fp16 __L24 dumps it wrote for Qwen were deleted for exactly
# the stale-cache reason (a file that exists in the wrong dtype/layer reads as measured).
#
# fp32 STORAGE IS THE POINT. 01e_repool's --cache-per-token stores fp16, and on Qwen's
# large-magnitude residual states fp16 rounding is 2-6e-3 — far above the consistency guard's 1e-4
# tolerance (that is what failed the guard on 2026-08-09, NOT an inline feature cache). 01h stores
# fp32 (its docstring says fp16; the code at 01h_pertoken.py:75 says float32 and the code is right).
#
# 01h's built-in SANITY (probe-PRR match) is OPPORTUNISTIC here, not the gate: it trains a probe
# with --label-field on a train split, so it CRASHES AFTER CACHING on the three eval-only sets
# (no train rows) and on any dataset whose labels have not landed yet (cnn/med_quad at submit
# time). The cache is written and fsynced BEFORE that block, so the artifact survives the crash.
# The REAL gate is feature_pertok_consistency (label-free, fp32-vs-fp32, ~1e-6), run afterwards.
#
#   sbatch slurm/qwen_pertoken.sbatch
#   # then, once all 8 are done:
#   python scripts/checks/feature_pertok_consistency.py --model Qwen/Qwen2.5-14B --layer 23 \
#       --datasets pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa,asqa,factscore

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}"

MODEL="Qwen/Qwen2.5-14B"
LAYER=23

# dataset : prompt-regime : label-field (for the opportunistic sanity only)
TASKS=(
  "cnn_dailymail::correctness"
  "xsum::correctness"
  "pubmed_qa::correctness"
  "expertqa:expertqa_rp12:factuality"
  "med_quad::correctness"
  "asqa:asqa_rp12:correctness"
  "samsum::correctness"
  "factscore:factscore_rp12:factuality"
)
IFS=':' read -r DS REG FIELD <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "=== pertok $DS (L$LAYER, regime '${REG:-<base>}') on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python -u scripts/01h_pertoken.py --model "$MODEL" --dataset "$DS" --ood ID \
    --layer "$LAYER" --dtype fp32 --attn eager --label-field "$FIELD" \
    ${REG:+--prompt-regime "$REG"} \
    || echo "01h exited non-zero for $DS -- expected for eval-only/unlabelled sanity; the guard below decides"

python -u scripts/checks/feature_pertok_consistency.py --model "$MODEL" \
    --datasets "$DS" ${REG:+--prompt-regime "$REG"} --layer "$LAYER" \
    || { echo "!!! consistency guard FAILED for $DS at L$LAYER"; exit 1; }

echo "=== pertok $DS done at $(date) ==="
