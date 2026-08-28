#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=wm_stopsmoke
#SBATCH --array=0-2
#SBATCH --output=logs/wmodels_stopfix_smoke_%A_%a.out

# One-off diagnostic: does a generation-time --stop-strings list fix the chat-turn-leak fabrication
# on Qwen2.5-32B's asqa/expertqa/factscore?
#
# BACKGROUND. The existing pct_fabricated detector (r'\bQuestion\s*:') only catches a hallucinated
# next turn that restates the literal word "Question:" -- the base-Llama few-shot failure shape.
# Direct inspection of chat_qwen32 (2026-08-25) found a DIFFERENT, much larger failure on this
# --chat-template population: the model answers correctly, then instead of emitting the real chat
# end-of-turn token it spells out a hallucinated next turn as ordinary text ('_user\nWrite an
# article about...\nAssistant:\n...'), which pct_fabricated never sees. Added as a new pct_turn_leak
# column in scripts/checks/generation_quality.py. Measured on the FULL rp=1.2 population:
#   asqa      35.0% (pct_fabricated read 0.0%)
#   factscore 35.4% (pct_fabricated read  3.0%)
#   expertqa  21.2% (pct_fabricated read  0.7%)
# vs <=1.4% on every other Qwen32B dataset and 0.0-0.05% on the corresponding
# Llama-3.1-8B-Instruct population (its one "_user" match was real code, '_userRepository', not a
# leak) -- so this is specific to Qwen2.5-32B on these three --repetition-penalty, large-budget,
# --chat-template datasets, not a universal artefact of the detector or the chat-template flag.
#
# THE FIX TESTED HERE: --stop-strings halts generation the moment the DECODED text contains one of
# a set of literal markers seen across the leaked rows (turn-boundary role words and their
# corrupted/no-underscore variants, and 'Question:' as a catch-all for the base-style continuation).
# This is NOT claimed to be exhaustive -- direct inspection also found rarer one-off leak shapes
# ('#Rewritten Prompt#:', a bare 'user\n' glued onto the previous word with no separator) that no
# fixed literal list can fully anticipate. Read this as a real, verified-substantial mitigation, not
# a guarantee of zero leak rate; the residual should still be checked against the (now-visible)
# pct_turn_leak column, not assumed away.
#
# ROUND 2 (2026-08-25, same day). The v1 STOPS list above and the v1 pct_turn_leak detector were both
# tuned to the UNFIXED population's leak shapes and so agreed with each other (both said 0.0% on the
# first v1 smoke run) -- but that was two matching blind spots, not a clean result. Direct inspection
# of the v1-fixed 150-row caches (bypassing both detector and stop list) found real residual leaks in
# shapes neither covered: capitalised ("_User"), "#"-prefixed with a space ("# user", "# assistant",
# "# system"), colon-terminated ("User:"), and glued onto the previous word with no separator
# (".user", "?assistant"). True residual rate under v1: asqa 2/150 (1.3%), factscore 9/150 (6.0%),
# expertqa 7/150 (4.7%) -- a big improvement on the pre-fix 62/49/69% baseline, but not zero. STOPS
# below adds those variants; generation_quality.py's _TURN_LEAK regex was independently widened
# (case-insensitive, colon/newline/punctuation-glued boundaries) so THIS round's report is not trusting
# the same narrow pattern that missed the v1 residual. Written to a v2 prompt-regime, not v1's, because
# 01_extract.py skips already-cached rows by regime -- reusing v1's regime would silently keep the OLD
# generations under a NEW-looking run.
#
# SAME smoke methodology as wmodels_reppen_smoke.sbatch: deterministic first-N rows via --limit,
# isolated into its own prompt-regime so nothing here can collide with or contaminate the real
# chat_qwen32 population. rp is held at 1.2 (the CURRENT canonical value) throughout -- the earlier
# reppen sweep, re-read against the corrected pct_turn_leak column, no longer shows a clear rp=1.1
# advantage on factscore (49.2% vs 49.3% true corrupted), so this diagnostic isolates the
# --stop-strings variable alone rather than compounding it with an unresolved rp question.
#
#   sbatch slurm/wmodels_stopfix_smoke.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-32B"
N="${LUQ_LIMIT:-150}"
REPPEN=1.2

DATASETS=(asqa expertqa factscore)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
case "$DS" in
  asqa)      BUD=256 ;;
  factscore) BUD=256 ;;
  expertqa)  BUD=384 ;;
esac

# Literal substrings that halt generation the moment they appear in the decoded text (HF's native
# stop_strings + tokenizer= mechanism -- see generate.generate's docstring). v1 verified against the
# clean (non-leaked, non-fabricated) rows of the full rp=1.2 population before use: near-zero false
# positives (the one borderline case, Llama-Instruct's real code identifier '_userRepository', is a
# DIFFERENT model/dataset and does not appear here). Round 2 additions (see header) are the exact
# variants direct inspection found surviving v1: capitalised role words, "#"-prefixed with a space,
# colon-terminated, and punctuation-glued with no separator. The glued forms (.user, .assistant,
# ?assistant) carry the most residual false-positive risk (a legitimate answer could in principle
# contain e.g. ".user" as part of a filename or identifier) -- accepted here because a real leak
# poisons the whole row for the judge, while a rare early truncation only shortens one answer.
STOPS="_user,_User,_assistant,_Assistant,# user,# User,# assistant,# Assistant,# system,# System,\n\nsystem\n,\nsystem\n,\nSystem\n,\n\nSystem\n,\nUser:,\nAssistant:,\n\nAssistant:,\nSystem:,\nHuman:,.user,.assistant,?assistant,?user,Question:"

REGIME="chat_qwen32_stopfix_smoke_v2_${DS}"

echo "=== stop-strings smoke ROUND 2 (DoC) ===" >&2
echo "  model=$MODEL dataset=$DS budget=$BUD repetition_penalty=$REPPEN rows=$N regime=$REGIME" >&2
echo "  stop_strings=$STOPS" >&2
echo "  host=$(hostname) at $(date)" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype bf16 --attn sdpa --chat-template \
    --prompt-regime "$REGIME" --max-new-tokens "$BUD" --max-new-tokens-cap "$BUD" \
    --repetition-penalty "$REPPEN" --stop-strings "$STOPS" --limit "$N" \
    || { echo "!!! extraction FAILED"; exit 1; }

echo "--- quality report (with the widened pct_turn_leak column) ---" >&2
python -u scripts/checks/generation_quality.py --model "$MODEL" --datasets "$DS" \
    --regime "$REGIME" \
    --out "results/wmodels_smoke/${DS}_stopfix_v2__Qwen_Qwen2.5-32B.csv"

echo "=== smoke done at $(date) ===" >&2
