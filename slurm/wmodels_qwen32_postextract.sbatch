#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=wm32_post
#SBATCH --array=0-5%3
#SBATCH --output=logs/wm32_post_%A_%a.out

# W-Models / Qwen2.5-32B on DoC. Slurm twin of pbs/wmodels_postextract.pbs.
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
# 01e_repool -> feature/pertok consistency GUARD -> 01h_pertoken.
# 01e_repool IS MANDATORY: the inline capture and the teacher-forced re-pool differ by ~8e-3 on
# the residual stream, and the pertok cache is teacher-forced. The guard turns that silent gap into
# a hard failure, so a non-zero exit stops the job BEFORE pertok is written.
#
#   sbatch slurm/wmodels_qwen32_postextract.sbatch

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
# SWITCHED TO sdpa 2026-08-17 (was eager). eager materialises an n^2 attention buffer per layer,
# which made the a100 slot itself the scarce resource this population needed TWICE (once for the
# memory canary, once for the real array), and the canary passed at only 96% headroom on xsum. sdpa
# computes the same attention with memory-efficient kernels, never materialising that buffer, at a
# ~1e-6 kernel-precision difference from eager -- the same order as other equivalence gates already
# used in this project. Verified safe for this panel: none of its six methods (perplexity, msp_min,
# SAPLMA, attention pooler, wmsp_norm, wmsp_shrink2) call output_attentions or need raw attention
# weights (that requirement is Lookback Lens / uhead, neither of which is in this panel). Record as a
# prereg deviation alongside D1/D2 -- NOT done from DoC, docs are read-only here.
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

echo "=== post-extract $DS layer $LAYER on $(hostname) at $(date) ===" >&2

# pubmed_qa was generated newline-stopped; the re-pool must reproduce the same token window.
TL=""; [ "$DS" = "pubmed_qa" ] && TL="--truncate-long"

echo "--- 1/3 re-pool (teacher-forced) ---" >&2
# shellcheck disable=SC2086
python -u scripts/01e_repool.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype "$DTYPE" --attn "$ATTN" $RG $TL || { echo "!!! repool failed" >&2; exit 6; }

echo "--- 2/3 per-token cache ---" >&2
# shellcheck disable=SC2086
python -u scripts/01h_pertoken.py --model "$MODEL" --dataset "$DS" --ood ID \
    --layer "$LAYER" --dtype "$DTYPE" --attn "$ATTN" $RG || { echo "!!! pertoken failed" >&2; exit 8; }

# ORDER FIXED 2026-08-15. The guard compares the pooled feature cache against the
# TEACHER-FORCED pertok cache, so it needs BOTH. Run before 01h_pertoken it skips every
# dataset, prints "ALL consistent" and exits 0 -- a rubber stamp, not a guard.
# --require-checked makes a SKIP fail the job so this cannot silently regress.
echo "--- 3/3 consistency guard (AFTER pertok, which it compares against) ---" >&2
# shellcheck disable=SC2086
python -u scripts/checks/feature_pertok_consistency.py --model "$MODEL" --datasets "$DS" \
    --layer "$LAYER" $RG --full --require-checked || { echo "!!! GUARD FAILED OR SKIPPED -- caches are NOT verified" >&2; exit 7; }

echo "=== $DS post-extract done at $(date) ===" >&2
