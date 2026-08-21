#!/bin/bash
#SBATCH --partition=a30
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=qwen_xlen_noop
#SBATCH --output=logs/qwen_xlen_noop_%j.out

# No-op control for the W-CrossLen Qwen replication (DOC_BRIEF_qwen_crosslength.md §6.1).
# CPU-only: the no-op cell only exercises floors/SAPLMA/wMSP, no pooler training.
#
# RUNS AS A BATCH JOB, NOT ON THE SHELL. This machine (cloud-vm-40-244) kills
# long-running processes started directly on it -- documented in
# slurm/qwen_api_jobs.sbatch's own header, and confirmed the hard way: a first
# attempt to run this inline died with no output file after being silently killed.
# A GPU partition is requested only because every DoC partition is GPU-typed; no
# --gres is requested, so it queues as CPU-only and doesn't compete for a GPU.
#
#   sbatch slurm/qwen_xlen_noop.sbatch

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

echo "=== xlen no-op control on $(hostname) at $(date) ==="
python -u scripts/checks/probedriftlong.py --model Qwen/Qwen2.5-14B --layer 23 \
    --seeds 1,2,3 --evals xsum \
    --train-spec pubmed_qa:1800 --rung-name "1ds-Diff-long" \
    --wmsp-only wmsp_norm,wmsp_shrink2,wmsp_shrink10 \
    --out results/xlen_NOOP_qwen_xsum__Qwen_Qwen2.5-14B.csv

echo "=== xlen no-op control done at $(date) ==="
