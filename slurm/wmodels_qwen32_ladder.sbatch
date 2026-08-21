#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=wm32_lad
#SBATCH --array=0-5%3
#SBATCH --output=logs/wm32_lad_%A_%a.out

# W-Models / Qwen2.5-32B on DoC. Slurm twin of pbs/wmodels_ladder.pbs.
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
# --sources IS NOT OPTIONAL. --evals restricts TARGETS ONLY; without --sources the rungs are
# still built from the full 8-dataset LONG_SRC, so this "six-dataset panel" would quietly train on
# med_quad and expertqa -- the two datasets the panel exists to drop.
#
# --wmsp-only wmsp_norm,wmsp_shrink2 IS THE PRE-REGISTRATION (prereg/M5 §4), not an optimisation.
# lambda=0 control, lambda=2 pre-specified, both transferred unchanged. NO lambda grid and NO
# model-specific selection runs on a replication population.
#
# --perex-dir is REQUIRED: the verification gate compares per-example vectors, not PRR (PRR moves
# ~1e-4 on 1e-7 round-off).
#
#   sbatch slurm/wmodels_qwen32_ladder.sbatch

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

PANEL="pubmed_qa,xsum,cnn_dailymail,samsum,asqa,factscore"
SLUG="$(printf '%s' "$MODEL" | tr '/' '_')"
OUT="results/wmodels_stageA__${SLUG}__${DS}.csv"
PEREX="results/perex_wmodels/${SLUG}"
mkdir -p "$PEREX"

echo "=== Stage-A ladder $DS layer $LAYER on $(hostname) at $(date) ===" >&2
python -u scripts/checks/probedriftlong.py \
    --model "$MODEL" --layer "$LAYER" \
    --evals "$DS" \
    --sources "$PANEL" \
    --rungs "ID,DiffTask,1ds-Diff" \
    --seeds 1,2,3 \
    --wmsp-only wmsp_norm,wmsp_shrink2 \
    --perex-dir "$PEREX" \
    --out "$OUT"
STATUS=$?
echo "=== $DS ladder done status=$STATUS at $(date) ===" >&2
exit $STATUS
