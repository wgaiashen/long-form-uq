#!/bin/bash
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_pxclean
#SBATCH --output=logs/qwen_perexclean_%j.out

# Slurm/DoC equivalent of pbs/perex_clean_qwen.pbs (that script is PBS/RCS-only: #PBS
# directives, $PBS_O_WORKDIR). Per DOC_HANDOVER_qwen_2026-08-25.md the corrected-span
# Qwen per-example regeneration must run on DoC (the cluster that made the master), so
# this wraps the SAME python invocation the handover specifies (byte-identical to the
# pbs script's) in a Slurm job instead of a PBS one.
#
# CPU-only: the per-token caches already exist on disk (layer 23, all 8 evals verified
# present before this job was ever submitted), so no forward pass through the 14B model
# happens here -- only cached-feature probe fits. Still routed through sbatch rather than
# the login shell: cloud-vm-40-244 kills long-running processes started directly on it.
#
#   sbatch slurm/qwen_perexclean_step.sbatch pubmed_qa   # step 1: probe eval, then certify --partial
#   sbatch slurm/qwen_perexclean_step.sbatch med_quad    # step 3: remaining evals, one job each

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

MODEL="Qwen/Qwen2.5-14B"
LAYER=23
export LUQ_CARVE="${LUQ_CARVE:-legacy}"
export LUQ_REGIME="samsum=trunc_v1,med_quad=trunc_v1,expertqa=expertqa_rp12_trunc_v1,factscore=factscore_rp12_trunc_v1"

EV="${1:?usage: sbatch slurm/qwen_perexclean_step.sbatch <eval>}"

mkdir -p results/perex_clean__Qwen_Qwen2.5-14B

echo "=== corrected-span per-example pass: eval=$EV layer=$LAYER carve=$LUQ_CARVE on $(hostname) at $(date) ==="
echo "commit: $(git rev-parse HEAD 2>/dev/null)"
echo "LUQ_REGIME=$LUQ_REGIME"
free -g | head -2

python -u scripts/checks/probedriftlong.py \
    --model "$MODEL" --layer "$LAYER" \
    --seeds 1,2,3 \
    --evals "$EV" \
    --wmsp-only wmsp_norm,wmsp_shrink2 \
    --perex-dir results/perex_clean__Qwen_Qwen2.5-14B \
    --out "results/pdl_perexclean_${EV}__Qwen_Qwen2.5-14B.csv" \
    || { echo "!!! pass FAILED for $EV"; exit 3; }

echo "=== eval=$EV finished at $(date) ==="
