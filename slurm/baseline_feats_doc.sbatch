#!/bin/bash
#SBATCH --job-name=luq-basefeats
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
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
# NAMING. --name ptrue_accurate, matching the seven datasets that already have it and the wording
# "Is the above response accurate?". A different name would silently create a SECOND, non-comparable
# feature set and the ladder would mix wordings across datasets without ever saying so.
#
# This CANNOT be slurm/ptrue_accurate_extract.sbatch reused: that one pins MODEL=google/gemma-2-9b-it,
# the dropped model, and covers only sciq/pubmed.
#
# a100 + 96G, NOT a40 (corrected 2026-08-03 after DoC caught it; verified on RCS from the records).
# cnn_dailymail's longest sequence is 2467 tokens, capped to 2048, so lookback's all-layer attention
# tensor is 32 x 32 x 2048^2 x 4B = 17.2 GB on top of ~32 GB of fp32 weights = 49.2 GB — over a 48 GB
# a40. med_quad (max 1358 -> 39.6 GB) and samsum (max 1117 -> 37.1 GB) would each fit an a40, but cnn is
# in this set and does not, so the whole job takes the a100. Host copy peaks ~2x the attention tensor,
# hence 96G.
#
# Do NOT let the RCS job 3535876 run at the same time — both write the same six files and a later
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
  # --store-layers 15 IS NOT OPTIONAL. 01b_ptrue defaults to `all`, which writes real values for all
  # 33 layers (~2.05GB for cnn alone, ~4.0GB total) against ~4.5GB free on DoC — it would very likely hit
  # the CephFS quota MID-WRITE, and a truncated npz here has silently happened before. With the flag the
  # unkept layers are NaN, which compresses to nothing: the seven existing caches are (n, 33, 4096) with
  # ONLY layer 15 finite, 33-44MB each. Without it these three would also be non-comparable with those
  # seven, and verify_baseline_features.py would fail — it checks the finite-layer pattern against sciq.
  python -u scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" \
      --name ptrue_accurate --store-layers 15 || echo "!!! ptrue_accurate FAILED for $DS"

  echo "======== $DS : 01c_lookback ======== $(date)"
  python -u scripts/01c_lookback.py --dataset "$DS" --ood ID --model "$MODEL" \
      || echo "!!! lookback FAILED for $DS"
done

echo "======== VERIFY — presence is not evidence ======== $(date)"
# --datasets IS REQUIRED (bug found on DoC 2026-08-04, fixed here). Without it the script printed
# its usage, exited non-zero, and tripped the `||` — so the log read "!!! verify_baseline_features
# reported a problem" when verification had NEVER RUN. That is this project's recurring failure shape:
# an ABSENT check presenting as a PERFORMED one, and a genuinely corrupt cache would have produced a
# byte-identical log line. The `|| echo` is what makes the two indistinguishable, so the arguments have
# to be right for the guard to mean anything.
python -u scripts/checks/verify_baseline_features.py --datasets cnn_dailymail med_quad samsum \
    || echo "!!! verify_baseline_features reported a problem"

echo "======== SIZES + SHA256 for the sync back ======== $(date)"
for DS in $DATASETS; do
  for N in ptrue_accurate lookback; do
    F="cache/features/meta-llama_Meta-Llama-3.1-8B__${DS}__ID__${N}.npz"
    [ -e "$F" ] && printf "%12s  %s  %s\n" "$(stat -c%s "$F")" "$(sha256sum "$F" | cut -d' ' -f1)" "$F" \
                || echo "MISSING  $F"
  done
done
echo "=== DONE $(date) ==="
