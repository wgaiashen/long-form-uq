#!/bin/bash
#SBATCH --job-name=luq-basefeats
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baseline_feats_doc-%j.out
#
# STEP A on DoC — extract the two SUPERVISED BASELINE feature sets missing for three datasets, so that
# P(True) and Lookback Lens can appear on the full ladder grid.
#
# WHY DoC. This is a GPU job that sat 5h40m in the RCS v1_gpu72 queue (632 queued / 55 running) without
# starting, and its inputs are tiny: 81MB of records in, ~115MB of features out. That is exactly the
# routing rule's case (2) — needs a GPU, genuinely stuck on RCS, inputs small enough to sync cleanly.
# The LADDERS stay on RCS: they read 21GB of pertok, and cache-bound work belongs where the cache lives.
#
# ⚠️ NAMING. --name ptrue_accurate, matching the seven datasets that already have it and the wording
# "Is the above response accurate?". A different name would silently create a SECOND, non-comparable
# feature set and the ladder would mix wordings across datasets without ever saying so.
#
# ⚠️ This CANNOT be slurm/ptrue_accurate_extract.sbatch reused: that one pins MODEL=google/gemma-2-9b-it,
# the dropped model, and covers only sciq/pubmed.
#
# ⚠️ Do NOT let the RCS job 3535876 run at the same time — both write the same six files and a later
# sync could overwrite one run's output with the other's. qdel it once this starts.
#
#   cd /vol/gpudata/gs925-msc_project/msc-project-gs925 && sbatch slurm/baseline_feats_doc.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
mkdir -p logs

MODEL=meta-llama/Meta-Llama-3.1-8B
DATASETS="cnn_dailymail med_quad samsum"

echo "=== baseline feature fill on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU visible)"

for DS in $DATASETS; do
  echo "======== $DS : 01b_ptrue --name ptrue_accurate ======== $(date)"
  python -u scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" \
      --name ptrue_accurate || echo "!!! ptrue_accurate FAILED for $DS"

  echo "======== $DS : 01c_lookback ======== $(date)"
  python -u scripts/01c_lookback.py --dataset "$DS" --ood ID --model "$MODEL" \
      || echo "!!! lookback FAILED for $DS"
done

echo "======== VERIFY — presence is not evidence ======== $(date)"
python -u scripts/checks/verify_baseline_features.py || echo "!!! verify_baseline_features reported a problem"

echo "======== SIZES + SHA256 for the sync back ======== $(date)"
for DS in $DATASETS; do
  for N in ptrue_accurate lookback; do
    F="cache/features/meta-llama_Meta-Llama-3.1-8B__${DS}__ID__${N}.npz"
    [ -e "$F" ] && printf "%12s  %s  %s\n" "$(stat -c%s "$F")" "$(sha256sum "$F" | cut -d' ' -f1)" "$F" \
                || echo "MISSING  $F"
  done
done
echo "=== DONE $(date) ==="
