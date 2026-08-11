#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen_slice
#SBATCH --output=logs/qwen_slice_%j.out
# Slicing the pertok caches to the truncated spans. CPU/IO work, but DoC forces a GPU allocation.
# It runs as a batch job because the first attempt was KILLED on the shell VM mid-write (6.2GB RSS,
# 3.5GB written) -- that box terminates heavy processes, which is documented and keeps biting.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
for M in "Qwen/Qwen2.5-14B" "meta-llama/Meta-Llama-3.1-8B"; do
  echo "########## $M ##########"
  python -u scripts/checks/slice_caches_truncated.py --model "$M" --suffix trunc_v1
done
echo "=== slice done at $(date) ==="
