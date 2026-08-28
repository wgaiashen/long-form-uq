#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=wm32_stopfix_full
#SBATCH --array=0-2%1
#SBATCH --output=logs/wmodels_qwen32_stopfix_full_%A_%a.out

# FULL regeneration of Qwen2.5-32B's three chat-turn-leak-corrupted datasets (asqa, expertqa,
# factscore), replacing the corrupted rows in the "chat_qwen32" population.
#
# BACKGROUND. Direct text inspection of the full "chat_qwen32" population (2026-08-25) found the
# model spelling out a hallucinated next chat turn as ordinary decoded text instead of emitting the
# real end-of-turn token -- structurally invisible to the old fabrication detector. True corrupted
# rates: asqa 62.0%, factscore 49.2%, expertqa 69.1%. --stop-strings (generate.generate's native HF
# stop_strings + tokenizer= mechanism) halts generation the moment the decoded text matches one of a
# set of literal turn-boundary markers. Two rounds of smoke testing (150 rows/dataset, --limit,
# isolated prompt-regimes) drove the residual down: round 1 (narrower list) left 1.3/6.0/4.7%
# genuine leaks on asqa/factscore/expertqa; round 2 (this STOPS list, widened for capitalised /
# "#"-prefixed / colon-terminated / punctuation-glued variants) left 0.0/0.7/1.3%. The remaining
# expertqa residual is a different shape again -- the model echoes the literal system prompt
# ("...marriage licenses.\nYou are a helpful assistant.") rather than a role marker -- caught by
# generation_quality.py's widened detector (_SYSTEM_PROMPT_ECHO) but NOT added to STOPS here, by
# explicit decision: the residual is now ~0.67% pooled across all three datasets, and this run
# proceeds on that basis rather than chasing a further round.
#
# WRITES TO A NEW REGIME, "chat_qwen32_stopfix", NOT "chat_qwen32". 01_extract.py skips
# already-cached rows by regime -- writing into "chat_qwen32" would silently do nothing (all rows
# already "cached" there, corrupted) rather than regenerating. One regime holds all three datasets'
# records side by side, same layout as the original "chat_qwen32" directory.
#
# THIS SCRIPT ONLY GENERATES + QUALITY-VERIFIES. It does NOT repool features, build the per-token
# cache, or point ladder scoring at the new regime -- those are separate, also-multi-hour-GPU steps
# and a separate decision about how "chat_qwen32_stopfix" supersedes the corrupted rows in whatever
# becomes the canonical per-model results table. Do that as an explicit next step once this is
# verified, not automatically here.
#
# Runtime estimate from the round-2 smoke rate (150 rows each): asqa ~4.2h (948 rows), expertqa
# ~10.4h (2016 rows, largest budget), factscore ~2h (500 rows). Ordered LONGEST-FIRST per this
# project's existing convention so no straggler runs alone at the end. %1 throttle keeps only one
# task actually running against the a100 allocation at a time (matches wmodels_qwen32_extract.sbatch
# / _postextract.sbatch's own 3-at-once throttle pattern, tightened to 1 here since this is 3 tasks
# not 6 and each is already large).
#
#   sbatch slurm/wmodels_qwen32_stopfix_full.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-32B"
REPPEN=1.2
REGIME="chat_qwen32_stopfix"

# Same literal list verified in round 2 (wmodels_stopfix_smoke.sbatch): 0.0/0.7/1.3% residual on
# asqa/factscore/expertqa, confirmed by direct text inspection, not just the automated column.
STOPS="_user,_User,_assistant,_Assistant,# user,# User,# assistant,# Assistant,# system,# System,\n\nsystem\n,\nsystem\n,\nSystem\n,\n\nSystem\n,\nUser:,\nAssistant:,\n\nAssistant:,\nSystem:,\nHuman:,.user,.assistant,?assistant,?user,Question:"

DATASETS=(expertqa asqa factscore)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
case "$DS" in
  expertqa)  BUD=384 ;;
  asqa)      BUD=256 ;;
  factscore) BUD=256 ;;
esac

echo "=== FULL stop-strings regeneration: $DS (DoC) ===" >&2
echo "  model=$MODEL dataset=$DS budget=$BUD repetition_penalty=$REPPEN regime=$REGIME" >&2
echo "  stop_strings=$STOPS" >&2
echo "  host=$(hostname) at $(date)" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype bf16 --attn sdpa --chat-template \
    --prompt-regime "$REGIME" --max-new-tokens "$BUD" --max-new-tokens-cap "$BUD" \
    --repetition-penalty "$REPPEN" --stop-strings "$STOPS" \
    || { echo "!!! extraction FAILED"; exit 1; }

echo "--- quality report (widened pct_turn_leak, incl. system-prompt-echo) ---" >&2
python -u scripts/checks/generation_quality.py --model "$MODEL" --datasets "$DS" \
    --regime "$REGIME" \
    --out "results/wmodels_genquality_chat_full_stopfix__Qwen_Qwen2.5-32B__${DS}.csv"

echo "=== $DS FULL regeneration done at $(date) ===" >&2
