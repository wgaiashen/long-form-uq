#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=wm_ext
#SBATCH --output=logs/wmodels_extract_%j.out

# GENERAL W-Models extraction on DoC, one dataset per job. Slurm twin of pbs/wmodels_extract.pbs.
# (slurm/wmodels_qwen32_extract.sbatch stays as the Qwen32-specific array; this one takes any model.)
#
# WHY THIS EXISTS: RCS's 48GB pool went to zero free L40S with 7 of our jobs queued behind it, and
# Llama-3.1-8B-Instruct at fp32 is ~32GB of WEIGHTS ALONE -- so every one of its six datasets needs a
# 48GB card and none can use RCS's abundant RTX6000. DoC's a40 is a separate pool from the jammed
# a100 one, so this does NOT compete with the Qwen32 work.
#
# SAFE TO SPLIT A POPULATION ACROSS CLUSTERS **BY DATASET**. Different datasets write different
# cache files, so two clusters cannot corrupt each other. Running the SAME dataset on both would --
# never do that. Sync the pertok caches into one place before the ladder, which needs all six.
#
# THE PER-DATASET SPEC IS BAKED IN, transcribed from slurm/qwen_generate_full.sbatch. Budgets are
# LLAMA's, unchanged. Passing them by hand is how a population ends up on different decoding settings.
#
# --dtype AND --attn PASSED EXPLICITLY. `auto` resolves to fp16 and token logprobs ARE the msp_min
# and wMSP signal.
#
# med_quad / expertqa are OUTSIDE the primary six-dataset panel and IN SCOPE ONLY for the D6
# eight-dataset sensitivity (prereg/multimodel_far_ood_panel.md §D6). Added here 2026-08-22 so the
# Llama-3.1-8B-Instruct and Qwen2.5-32B D6 extensions can run on DoC the same way gemma-2-9b's did
# on RCS via pbs/wmodels_extract.pbs. expertqa is borderline against the default 12h walltime --
# override with --time=24:00:00 at submit time (do not raise the script default; every other
# dataset's job would then request GPU time it does not need).
#
#   sbatch --export=ALL,LUQ_MODEL=meta-llama/Llama-3.1-8B-Instruct,LUQ_DTYPE=fp32,LUQ_DATASET=xsum \
#          slurm/wmodels_extract.sbatch
#   sbatch --time=24:00:00 --export=ALL,LUQ_MODEL=...,LUQ_DTYPE=fp32,LUQ_DATASET=expertqa,LUQ_CHAT=1 \
#          slurm/wmodels_extract.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache      # MUST precede _env.sh (it assigns with :=)
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  HF_HOME=$HF_HOME  FACTSCORE_DIR=${FACTSCORE_DIR:-<unset>}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${LUQ_MODEL:?set LUQ_MODEL}"
DTYPE="${LUQ_DTYPE:?set LUQ_DTYPE (fp32 | bf16) -- never leave this to auto}"
DS="${LUQ_DATASET:?set LUQ_DATASET}"
ATTN="${LUQ_ATTN:-eager}"
# --chat-template changes the generations, so it MUST get a fresh --prompt-regime, on EVERY dataset
# including asqa/factscore -- their existing asqa_rp12/factscore_rp12 already holds raw-few-shot
# records, and mixing the two under one namespace is exactly what the fresh-regime rule exists to
# stop (01_extract's own prompt-hash guard would refuse it anyway). LUQ_CHAT_REGIME names the new one.
CHAT=""; [ "${LUQ_CHAT:-0}" = "1" ] && CHAT="--chat-template"
CHAT_REGIME="${LUQ_CHAT_REGIME:-llama31i_chat}"

case "$DS" in
  cnn_dailymail) BUD=128; PENALTY="" ;;                    # 3800 rows
  pubmed_qa)     BUD=128; PENALTY=""; TRUNC="--truncate-long" ;;  # 3800, newline-stopped
  xsum)          BUD=56;  PENALTY="" ;;                    # 3800
  asqa)          BUD=256; PENALTY="--repetition-penalty 1.2" ;;   # 948
  samsum)        BUD=56;  PENALTY="" ;;                    # 1800
  factscore)     BUD=256; PENALTY="--repetition-penalty 1.2" ;;   # 500
  # --- outside the primary six-dataset panel; IN SCOPE ONLY for the D6 eight-dataset sensitivity.
  # Specs transcribed verbatim from pbs/wmodels_extract.pbs (the RCS twin already used for gemma-2-9b's
  # D6 generation), so both clusters run the identical D6 spec. Under --chat-template these still keep
  # their own decoding flags (no-repeat-ngram / repetition-penalty); only --prompt-regime is overridden
  # to the chat namespace, same override rule as the six-dataset panel below.
  med_quad)      BUD=128; PENALTY=""; EXTRA_NGRAM="--no-repeat-ngram-size 3" ;;            # 1800 rows
  expertqa)      BUD=384; PENALTY="--repetition-penalty 1.2"; NONCHAT_REGIME="expertqa_rp12" ;;  # 2016 rows
  *) echo "!!! '$DS' is not in the six-dataset panel (the multi-model panel registration §2), and is not one of the"; \
     echo "!!! two datasets the D6 sensitivity adds (med_quad, expertqa). Anything else is a typo." >&2; exit 3 ;;
esac
TRUNC="${TRUNC:-}"
EXTRA_NGRAM="${EXTRA_NGRAM:-}"
NONCHAT_REGIME="${NONCHAT_REGIME:-}"

if [ -n "$CHAT" ]; then
  REGIME="$CHAT_REGIME"
elif [ "$DS" = "asqa" ]; then
  REGIME="asqa_rp12"
elif [ "$DS" = "factscore" ]; then
  REGIME="factscore_rp12"
elif [ -n "$NONCHAT_REGIME" ]; then
  REGIME="$NONCHAT_REGIME"
else
  REGIME=""
fi
EXTRA="$PENALTY $TRUNC $EXTRA_NGRAM $CHAT"
[ -n "$REGIME" ] && EXTRA="--prompt-regime $REGIME $EXTRA"

# --max-new-tokens is only legal alongside a --prompt-regime (01_extract:116).
MNT=""
[ -n "$REGIME" ] && MNT="--max-new-tokens $BUD"

echo "=== W-Models extract (DoC) ===" >&2
echo "  model=$MODEL dtype=$DTYPE attn=$ATTN dataset=$DS budget=$BUD regime=${REGIME:-<default>} extra='$EXTRA'" >&2
echo "  host=$(hostname) at $(date)" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype "$DTYPE" --attn "$ATTN" \
    $MNT --max-new-tokens-cap "$BUD" $EXTRA
STATUS=$?
echo "=== $DS done status=$STATUS at $(date) ===" >&2
exit $STATUS
