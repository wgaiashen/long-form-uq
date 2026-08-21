#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_feat_trivia_xlen
#SBATCH --output=logs/qwen_features_trivia_xlen_%j.out

# Feature extraction for Qwen2.5-14B trivia_qa, W-CrossLen replication
# (DOC_BRIEF_qwen_crosslength.md §5).
#
# CORRECTS A BUG IN THE BRIEF. §5 of DOC_BRIEF_qwen_crosslength.md says to run
# `01e_repool.py --cache-per-token` for the per-token cache. That is WRONG for Qwen:
# 01e_repool's internal "middle layer" formula is n_layers//2 where n_layers = 49
# (48 hidden + embedding), giving L24 -- and it stores fp16. Neither matches the
# canonical Qwen population. slurm/qwen_pertoken.sbatch's own header documents
# exactly this: fp16 L24 dumps from 01e_repool were built and then DELETED
# (2026-08-09) because Qwen's large-magnitude residual states round 2-6e-3 in fp16,
# far above the 1e-4 consistency-guard tolerance, and because the layer itself was
# wrong -- the canonical layer is L23 (the ceil(N/2)-1 convention, matching
# Llama's L15), produced instead by 01h_pertoken.py --layer 23 --dtype fp32 for all
# 8 existing Qwen long-form datasets. Verified by reading 01e_repool.py's source
# directly (mid = (n_hidden_layers+1)//2 = 24) rather than trusting either script's
# comments alone.
#
# So this job does TWO passes, matching how the canonical 8-dataset population was
# actually built (qwen_repool.sbatch + qwen_pertoken.sbatch as separate jobs):
#   1. 01e_repool.py, NO --cache-per-token -- all-layer SAPLMA pooled features only
#      (layer selection is free at read time for this artifact, so 01e_repool's
#      internal mid-layer formula never applies to it).
#   2. 01h_pertoken.py --layer 23 --dtype fp32 -- the canonical per-token cache.
#
#   sbatch slurm/qwen_features_trivia_xlen.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}"

MODEL="Qwen/Qwen2.5-14B"
LAYER=23

echo "=== 1/2: SAPLMA all-layer repool, trivia_qa, on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u scripts/01e_repool.py --model "$MODEL" --dataset trivia_qa --ood ID \
    --dtype fp32 --attn eager \
    || { echo "!!! 01e_repool FAILED"; exit 1; }

echo "=== 2/2: canonical per-token cache, trivia_qa, L$LAYER, at $(date) ==="
python -u scripts/01h_pertoken.py --model "$MODEL" --dataset trivia_qa --ood ID \
    --layer "$LAYER" --dtype fp32 --attn eager --label-field correctness \
    || echo "01h exited non-zero -- may be the opportunistic sanity-probe step, not the caching; the guard below decides"

echo "=== consistency guard at $(date) ==="
python -u scripts/checks/feature_pertok_consistency.py --model "$MODEL" \
    --datasets trivia_qa --layer "$LAYER" \
    || { echo "!!! consistency guard FAILED"; exit 1; }

echo "=== trivia_qa feature extraction done at $(date) ==="
