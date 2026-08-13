#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_pdl_base
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_pdl_baselines_clean_%A_%a.out

# Score P(True) probe + Lookback Lens into the clean-population ladder (report table methods 12-13).
# --skip-wmsp --skip-poolers avoids retraining the 10 wMSP variants and the pooler temperature
# search (the two expensive parts); floors + SAPLMA still compute (no flag to skip those) along
# with the ~20-min per-token pool load per eval, so this is cheaper than the full ladder, not free.
# Sibling of slurm/qwen_pdl_ladder_clean.sbatch -- same LUQ_REGIME, same population split, writes to
# its OWN output path so it can never collide with the canonical per-eval CSVs or the 9-method
# clean master (per DOC_HANDOVER_qwen_phase2_2026-08-10.md: "never append into or overwrite the
# existing per-eval CSVs or the master").
#
# Both feature caches are confirmed complete before this runs: P(True) probe (job 273070, all 8
# datasets, clean regime for the 4 affected) and Lookback (jobs 273072/273073/273074, all 8
# datasets, same population split -- 6 short-source + xsum/cnn_dailymail longsrc).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  LUQ_CARVE=${LUQ_CARVE:-legacy}"

MODEL="Qwen/Qwen2.5-14B"
LAYER=23
export LUQ_REGIME="samsum=trunc_v1,med_quad=trunc_v1,expertqa=expertqa_rp12_trunc_v1,factscore=factscore_rp12_trunc_v1"

EVALS=(pubmed_qa med_quad asqa xsum cnn_dailymail samsum expertqa factscore)
EV="${EVALS[$SLURM_ARRAY_TASK_ID]}"

OUT="results/analysis/pdl_qwenclean_baselines_${EV}__$(echo "$MODEL" | tr '/' '_').csv"

echo "=== BASELINES (ptrue,lookback) ladder $EV (L$LAYER) on $(hostname) at $(date) ==="
echo "LUQ_REGIME=$LUQ_REGIME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/checks/probedriftlong.py --model "$MODEL" --layer "$LAYER" \
    --evals "$EV" --seeds 1,2,3 --out "$OUT" \
    --baselines ptrue,lookback --skip-wmsp --skip-poolers \
    || { echo "!!! baselines ladder FAILED for $EV"; exit 1; }

echo "=== baselines ladder $EV done at $(date) ==="
