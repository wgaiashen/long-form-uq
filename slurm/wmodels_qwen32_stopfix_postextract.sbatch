#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=wm32_stopfix_post
#SBATCH --array=0-2%1
#SBATCH --output=logs/wmodels_qwen32_stopfix_post_%A_%a.out

# Postextract for the "chat_qwen32_stopfix" regime (see wmodels_qwen32_stopfix_full.sbatch): the
# three chat-turn-leak-corrected Qwen2.5-32B datasets (expertqa, asqa, factscore), now clean at
# 0.1-1.4% verified residual leak, need the same 01e_repool -> 01h_pertoken -> consistency-guard
# chain the rest of the panel already has, before they can feed the ladder / scoring.
#
# SAME dtype/attn/layer as the rest of the Qwen2.5-32B panel (wmodels_qwen32_postextract.sbatch):
# bf16 + sdpa + LAYER=31 (ceil(64/2)-1, the fixed middle-layer rule). NOT recomputed here -- these
# are properties of the model, not the regime, and must match across every dataset in this
# population for the pooled features to be comparable.
#
# NEITHER 01e_repool NOR 01h_pertoken take --chat-template: both read prompt_token_ids/gen_token_ids
# straight out of the cached Tier-1 record and replay them teacher-forced, so whatever template
# produced those tokens at generation time is already baked in. Nothing to pass here.
#
# LABEL-FIELD PER DATASET (01h_pertoken's --label-field, used only for its OPPORTUNISTIC sanity
# probe, not the real gate): expertqa and factscore are judge-labelled on `factuality`
# (factuality/uncovered/coherent scheme), asqa on plain `correctness` -- same mapping as the
# canonical qwen_pertoken.sbatch (Qwen2.5-14B) uses for the same two datasets. These new
# chat_qwen32_stopfix rows have NOT been judge-labelled yet (02_label not run), so on this first
# pass the sanity probe has no train-split labels to fit and 01h will exit non-zero after caching --
# expected, matches qwen_pertoken.sbatch's own documented behaviour on eval-only/unlabelled sets.
# The real gate is feature_pertok_consistency (label-free, fp32-vs-fp32, ~1e-6 tolerance).
#
# ORDER LONGEST-FIRST (expertqa 2016 rows > asqa 948 > factscore 500), %1 THROTTLE: keeps only one
# task on the shared a100 allocation at a time, same discipline as the generation run this
# postextract follows.
#
# THIS SCRIPT DOES NOT JUDGE-LABEL THE NEW RECORDS OR TOUCH THE CANONICAL RESULTS TABLE. Both
# remain separate, explicit next steps.
#
#   sbatch slurm/wmodels_qwen32_stopfix_postextract.sbatch
#   # then, once all 3 are done:
#   python scripts/checks/feature_pertok_consistency.py --model Qwen/Qwen2.5-32B --layer 31 \
#       --prompt-regime chat_qwen32_stopfix --datasets expertqa,asqa,factscore --full --require-checked

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  HF_HOME=$HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-32B"
DTYPE=bf16
ATTN=sdpa
LAYER=31
REGIME="chat_qwen32_stopfix"

# dataset : label-field (for 01h_pertoken's opportunistic sanity probe only)
TASKS=(
  "expertqa:factuality"
  "asqa:correctness"
  "factscore:factuality"
)
IFS=':' read -r DS FIELD <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "=== post-extract $DS (regime $REGIME) layer $LAYER on $(hostname) at $(date) ===" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "--- 1/3 re-pool (teacher-forced) ---" >&2
python -u scripts/01e_repool.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype "$DTYPE" --attn "$ATTN" --prompt-regime "$REGIME" \
    || { echo "!!! repool failed" >&2; exit 6; }

echo "--- 2/3 per-token cache ---" >&2
python -u scripts/01h_pertoken.py --model "$MODEL" --dataset "$DS" --ood ID \
    --layer "$LAYER" --dtype "$DTYPE" --attn "$ATTN" --prompt-regime "$REGIME" --label-field "$FIELD" \
    || echo "01h exited non-zero for $DS -- expected pre-labelling (no train-split labels yet); the guard below decides" >&2

echo "--- 3/3 consistency guard (AFTER pertok, which it compares against) ---" >&2
python -u scripts/checks/feature_pertok_consistency.py --model "$MODEL" --datasets "$DS" \
    --layer "$LAYER" --prompt-regime "$REGIME" --full --require-checked \
    || { echo "!!! GUARD FAILED OR SKIPPED -- caches are NOT verified" >&2; exit 7; }

echo "=== $DS post-extract done at $(date) ===" >&2
