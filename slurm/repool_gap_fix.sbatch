#!/bin/bash
#SBATCH --job-name=repool-gap
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/repool_gap-%j.out
# FEATURE-CACHE CONSISTENCY FIX — the CORRECT gap set (2026-07-24). This is the DoC/A100 port of
# pbs/repool_feature_fix.pbs. It re-pools the SAPLMA FEATURE cache, teacher-forced, for the FIVE datasets
# whose feature cache was captured inline (NOT teacher-forced) and so disagrees with the pertok cache:
#   med_quad, samsum, expertqa (rp12), asqa (rp12), cnn_dailymail.
# (slurm/repool_fix.sbatch is a DIFFERENT, older job that only did sciq+trivia_qa — NOT the gap. Do not use
#  it for this.) cnn has the long article prompt that OOMs the RCS 48GB L40S; the A100-80GB here handles it,
# which is the whole reason this belongs on DoC.
# Feature-only (NO per-token recompute) -> the pertok cache and every pertok-based ladder result
# (§B/§VII/§VIII) are UNTOUCHED; only §A ID SAPLMA + the ID-gate get corrected (~0.01 PRR).
set -uo pipefail
# env setup copied verbatim from the working slurm/repool_fix.sbatch (conda `luq` on DoC):
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== repool gap fix on $(hostname) at $(date) ===" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
declare -A REGIME=( [med_quad]="" [samsum]="" [cnn_dailymail]="" [expertqa]="expertqa_rp12" [asqa]="asqa_rp12" )
for d in med_quad samsum expertqa asqa cnn_dailymail; do
  echo "---- repool $d (regime='${REGIME[$d]}') $(date) ----" >&2
  python -u scripts/01e_repool.py --model meta-llama/Meta-Llama-3.1-8B --dataset "$d" --ood ID \
    --prompt-regime "${REGIME[$d]}" --dtype fp32 --attn eager \
    && echo "  OK $d" >&2 || echo "  !!! FAILED $d" >&2
done
echo "=== done $(date) ===" >&2
