#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=wm32_canary
#SBATCH --output=logs/wmodels_qwen32_canary_%j.out

# W-Models / Qwen2.5-32B memory canary, batch form of the DOC_BRIEF_wmodels_qwen32.md §3.2 srun
# command. Written as sbatch instead of a foreground srun so it queues through Slurm on its own
# (the a100 partition was seen fully allocated on 2026-08-15/16) rather than tying up an
# interactive shell for however long the queue takes.
#
#   sbatch slurm/wmodels_qwen32_canary.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== canary on $(hostname) at $(date) ===" >&2
python -u scripts/checks/wmodels_memory_canary.py --model Qwen/Qwen2.5-32B \
    --dataset xsum --dtype bf16 --attn eager --top 3
STATUS=$?
echo "=== canary done status=$STATUS at $(date) ===" >&2
exit $STATUS
