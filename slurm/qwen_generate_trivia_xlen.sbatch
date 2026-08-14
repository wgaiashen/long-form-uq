#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_gen_trivia_xlen
#SBATCH --output=logs/qwen_generate_trivia_xlen_%j.out

# Qwen2.5-14B generation, trivia_qa ONLY, for the W-CrossLen cross-length transfer
# replication (DOC_BRIEF_qwen_crosslength.md §3). Single dataset, not an array --
# adapted from slurm/qwen_generate_full.sbatch's pattern (a100, fp32+eager, resumable).
#
# Budget 20 (data.py's table value for trivia_qa, unchanged from Llama). No
# --prompt-regime (trivia_qa is a core dataset -> default namespace). No
# --truncate-long needed: trivia_qa is in SHORT_FORM, so 01_extract truncates at the
# first newline automatically. No --repetition-penalty (that's the asqa/expertqa/
# factscore treatment only).
#
#   sbatch slurm/qwen_generate_trivia_xlen.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}"

MODEL="Qwen/Qwen2.5-14B"

echo "=== trivia_qa (budget 20) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --dtype fp32 --attn eager passed explicitly: 01_extract refuses `auto` for a
# non-Llama model rather than silently resolving to fp16, which would make token
# probabilities (the msp_min signal) incomparable to the rest of the Qwen population.
python -u scripts/01_extract.py --model "$MODEL" --dataset trivia_qa --ood ID \
    --dtype fp32 --attn eager

echo "=== trivia_qa done at $(date) ==="
