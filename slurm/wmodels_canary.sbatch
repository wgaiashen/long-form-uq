#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=wm_canary
#SBATCH --output=logs/wmodels_canary_%j.out

# GENERAL W-Models memory canary on DoC, any model/dataset. Slurm twin of pbs/wmodels_canary.pbs.
# (slurm/wmodels_qwen32_canary.sbatch stays as the Qwen32-specific one.) Runs the LONGEST real
# inputs through the actual extraction path and reports peak GPU memory, before a full array is
# submitted -- an OOM discovered here costs ~1h, not a wasted multi-hour slot on a contended
# partition plus a re-queue.
#
# THE SMOKE DOES NOT COVER THIS. `01_extract --limit 40` takes the FIRST 40 rows by index, and
# the rows that OOM sit in the tail.
#
#   sbatch --partition=a40 --export=ALL,LUQ_MODEL=meta-llama/Llama-3.1-8B-Instruct,LUQ_DTYPE=fp32,LUQ_DATASET=xsum \
#          slurm/wmodels_canary.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${LUQ_MODEL:?set LUQ_MODEL}"
DTYPE="${LUQ_DTYPE:?set LUQ_DTYPE}"
DS="${LUQ_DATASET:-xsum}"
ATTN="${LUQ_ATTN:-eager}"
TOP="${LUQ_TOP:-3}"

echo "=== canary: $MODEL | $DS | $DTYPE + $ATTN | top $TOP longest on $(hostname) at $(date) ===" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/checks/wmodels_memory_canary.py \
    --model "$MODEL" --dataset "$DS" --dtype "$DTYPE" --attn "$ATTN" --top "$TOP"
STATUS=$?
echo "=== canary done status=$STATUS at $(date) ===" >&2
exit $STATUS
