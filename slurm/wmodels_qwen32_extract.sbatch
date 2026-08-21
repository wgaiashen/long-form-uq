#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=wm32_ext
#SBATCH --array=0-5%3
#SBATCH --output=logs/wm32_ext_%A_%a.out

# W-Models / Qwen2.5-32B on DoC. Slurm twin of pbs/wmodels_extract.pbs.
#
# HF_HOME IS OVERRIDDEN TO /vol/bitbucket BEFORE SOURCING _env.sh. The DoC default
# (/vol/gpudata/gs925-msc_project/hf_cache) is inside the HARD 50GB CephFS quota, which already
# holds Qwen-14B (27.5GB) and had ~4GB free. Qwen2.5-32B is ~65GB in bf16. _env.sh assigns with
# `:=`, so an already-exported value wins.
#
# bf16, NOT fp32. fp32 Qwen-32B is ~130GB, over any single card. Dtype is held FIXED within this
# population, so it never differs between the two wMSP variants being compared here -- but it is an
# additional implementation difference when this population is read beside the fp32 ones.
#
# --dtype AND --attn PASSED EXPLICITLY. `auto` resolves to fp16 and token logprobs ARE the
# msp_min / wMSP signal. 01_extract's --require-explicit-dtype would refuse `auto` anyway.
#
# --max-new-tokens IS ONLY LEGAL ALONGSIDE A --prompt-regime (01_extract:116). Passed only
# where a regime exists; --max-new-tokens-cap carries the budget elsewhere.
#
#   sbatch slurm/wmodels_qwen32_extract.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  HF_HOME=$HF_HOME"

# Allocator only -- numerically inert, so it cannot affect comparability. It matters because xsum's
# longest prompt is ~5.9k tokens under the Qwen tokeniser and eager attention allocates/frees a
# large n^2 buffer once PER LAYER, 64 times per forward, which fragments the heap.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-32B"
DTYPE=bf16
# SWITCHED TO sdpa 2026-08-17 (was eager) -- see wmodels_qwen32_postextract.sbatch for the full
# reasoning (the a100 canary/array double-queue cost, the 96%-headroom xsum pass, and the panel-wide
# verification that no method here needs raw attention weights). Record as a prereg deviation.
ATTN=sdpa
LAYER=31                       # ceil(64/2)-1, the fixed rule. Cross-checked against the registry.

# The six-dataset reduced panel, LONGEST-FIRST so no straggler runs alone at the end.
# dataset : budget (LLAMA's, unchanged) : extra flags -- verbatim from qwen_generate_full.sbatch
TASKS=(
  "cnn_dailymail:128:"
  "pubmed_qa:128:--truncate-long"
  "xsum:56:"
  "asqa:256:--prompt-regime asqa_rp12 --repetition-penalty 1.2"
  "samsum:56:"
  "factscore:256:--prompt-regime factscore_rp12 --repetition-penalty 1.2"
)
IFS=':' read -r DS BUD EXTRA <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
case "$EXTRA" in *--prompt-regime*) REGIME="$(printf '%s' "$EXTRA" | sed -E 's/.*--prompt-regime ([^ ]+).*/\1/')" ;; *) REGIME="" ;; esac
RG=""; [ -n "$REGIME" ] && RG="--prompt-regime $REGIME"

MNT=""
case "$EXTRA" in *--prompt-regime*) MNT="--max-new-tokens $BUD" ;; esac

echo "=== $DS (budget $BUD, FULL dataset) on $(hostname) at $(date) ===" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype "$DTYPE" --attn "$ATTN" \
    $MNT --max-new-tokens-cap "$BUD" $EXTRA
STATUS=$?
echo "=== $DS done status=$STATUS at $(date) ===" >&2
exit $STATUS
