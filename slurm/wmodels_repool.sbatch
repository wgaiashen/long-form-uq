#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=wm_repool
#SBATCH --output=logs/wmodels_repool_%j.out

# GENERAL, model-agnostic teacher-forced repool + consistency guard, one dataset per job. Generalizes
# slurm/qwen_repool.sbatch (hardcoded to Qwen2.5-14B) the same way wmodels_extract.sbatch generalized
# the six-dataset extraction driver -- model/dataset/regime/layer become environment variables.
#
# WHY THIS IS THE MANDATORY NEXT STEP AFTER 01_extract, NOT AN OPTIONAL ONE (project convention):
# the SAPLMA feature cache must be teacher-forced, matching the per-token pertok cache, or code that
# reads the feature cache directly silently drifts from the pertok-based ladders (the med_quad/cnn/
# ASQA/ExpertQA gap this rule exists to prevent). Runs 01e_repool then immediately
# feature_pertok_consistency.py on the SAME dataset, so a repool that doesn't match its own pertok
# dump fails loudly here rather than silently downstream.
#
# a100 ON PURPOSE, EVEN FOR THE FP32 8B POPULATION: a teacher-forced forward over a long source
# (xsum/cnn's ~5.6k-token articles) peaks like the generation prefill did and needs the headroom.
#
# --layer MUST MATCH THE POPULATION'S PROBE LAYER (15 for Llama-3.1-8B[-Instruct], 31 for
# Qwen2.5-32B, 20 for gemma-2-9b) -- 01e_repool itself always dumps the (n_layers+1)//2 middle layer
# regardless of what you pass here (there is no way to override which layer IT dumps), so LUQ_LAYER
# is only used to check the CONSISTENCY GUARD reads the same file 01e_repool just wrote. If the two
# disagree the guard will fail loudly rather than silently checking the wrong file.
#
#   sbatch --export=ALL,LUQ_MODEL=meta-llama/Llama-3.1-8B-Instruct,LUQ_DATASET=med_quad,\
#       LUQ_REGIME=llama31i_chat,LUQ_DTYPE=fp32,LUQ_ATTN=eager,LUQ_LAYER=15 slurm/wmodels_repool.sbatch
#   (remember the DoC --export=ALL gotcha: export the vars in-shell instead and drop --export)

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${LUQ_MODEL:?set LUQ_MODEL}"
DS="${LUQ_DATASET:?set LUQ_DATASET}"
REGIME="${LUQ_REGIME:-}"
DTYPE="${LUQ_DTYPE:?set LUQ_DTYPE}"
ATTN="${LUQ_ATTN:-eager}"
LAYER="${LUQ_LAYER:?set LUQ_LAYER -- the population probe layer, never guessed}"

echo "=== W-Models repool (DoC) ===" >&2
echo "  model=$MODEL dataset=$DS regime=${REGIME:-<base>} dtype=$DTYPE attn=$ATTN layer=$LAYER" >&2
echo "  host=$(hostname) at $(date)" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01e_repool.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype "$DTYPE" --attn "$ATTN" --cache-per-token \
    ${REGIME:+--prompt-regime "$REGIME"} || { echo "!!! repool FAILED for $DS"; exit 1; }

python -u scripts/checks/feature_pertok_consistency.py --model "$MODEL" \
    --datasets "$DS" ${REGIME:+--prompt-regime "$REGIME"} --layer "$LAYER" \
    || { echo "!!! consistency guard FAILED for $DS"; exit 1; }

echo "=== repool+guard $DS done at $(date) ===" >&2
