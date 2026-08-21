#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_repool
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_repool_%A_%a.out

# Teacher-forced RE-POOL of the Qwen2.5-14B feature cache + the per-token cache, one dataset per
# array task. This is the MANDATORY step after 01_extract (project convention: the SAPLMA feature cache must
# be teacher-forced, matching the pertok cache), ported from the Llama invocation verbatim:
# slurm/overnight_gpu.sbatch ran `01e_repool --dtype fp32 --attn eager --cache-per-token`, and
# scripts/checks/make_pubmed_d1.py records that pubmed takes NO --truncate-long here (its cached
# gen tokens were already newline-cut at extraction; cutting again would double-apply the rule to
# nothing but would desync the labels if the rule ever changed -- so the flag stays off, as on Llama).
#
# The pertok layer is NOT chosen here: 01e_repool dumps mid = (n_hidden_layers+1)//2, the SAME rule
# that produced Llama's cache (33//2 = 16), which on Qwen gives 49//2 = 24. Whether the ladder wants
# L24 (this rule) or L23 (proportional transfer of the ceil(N/2)-1) is W-Qwen layer-selection
# business decided on dev on RCS; if it lands elsewhere, re-dump by re-running with the chosen layer.
# The FEATURE cache is all-layers, so layer selection there is free either way.
#
# a100 on purpose: fp32 Qwen-14B is ~55GB of weights and the teacher-forced forward over an
# ~5.9k-token xsum/cnn source peaks like the generation prefill did (~71GB) -- no smaller card fits.
#
#   sbatch slurm/qwen_repool.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}"

MODEL="Qwen/Qwen2.5-14B"
QWEN_PERTOK_LAYER=24        # (48+1)//2 -- same formula that gave Llama its L16 dump

# dataset : prompt-regime ("" = default namespace). Longest-first, as in generation.
TASKS=(
  "cnn_dailymail:"
  "xsum:"
  "pubmed_qa:"
  "expertqa:expertqa_rp12"
  "med_quad:"
  "asqa:asqa_rp12"
  "samsum:"
  "factscore:factscore_rp12"
)
IFS=':' read -r DS REG <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "=== repool $DS (regime '${REG:-<base>}') on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python -u scripts/01e_repool.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype fp32 --attn eager --cache-per-token \
    ${REG:+--prompt-regime "$REG"} || { echo "!!! repool FAILED for $DS"; exit 1; }

# The guard, immediately, per dataset -- a repool whose output does not match its own pertok dump
# is exactly the silent 8e-3 gap this pipeline once shipped. --layer must match the pertok file key.
python -u scripts/checks/feature_pertok_consistency.py --model "$MODEL" \
    --datasets "$DS" ${REG:+--prompt-regime "$REG"} --layer "$QWEN_PERTOK_LAYER" \
    || { echo "!!! consistency guard FAILED for $DS"; exit 1; }

echo "=== repool $DS done at $(date) ==="
