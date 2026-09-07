#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_pdl_clean
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_pdl_clean_%A_%a.out

# CLEAN-SPAN SENSITIVITY ladder for Qwen2.5-14B (2026-08-11, span-contamination audit follow-up).
# Same shape as slurm/qwen_pdl_ladder.sbatch (one eval per array task, 3 seeds, the COMPLETE grid,
# no --rungs subset) with two deliberate differences, mirroring RCS's pdl_medquad_clean.pbs pattern
# for Llama med_quad:
#   1. LUQ_REGIME points samsum/med_quad/expertqa/factscore at their _trunc_v1 shadow caches (built
#      2026-08-11 by build_truncated_records.py + slice_caches_truncated.py, already relabelled —
#      see results/analysis/QWEN_GENERATION_VALIDITY_AUDIT.md). pubmed_qa/asqa/xsum/cnn_dailymail
#      stay canonical (xsum/cnn's truncation-sensitivity delta was ~0 per the audit; not worth the
#      extra namespace this cycle). All 8 evals are still submitted (not just the 4 touched) because
#      OOD training pools mix datasets — an untouched eval's OOD cells can still shift if their
#      training pool draws from a cleaned dataset. The untouched eval/rung combinations where the
#      training pool has NO overlap with the 4 cleaned datasets are the correctness control: they
#      MUST reproduce the canonical master to 4 dp.
#   2. --out goes to results/analysis/ under a sensitivity name — the canonical
#      probedriftlong__Qwen_Qwen2.5-14B__*.csv and pdl_master are NEVER touched by this job. Promoted
#      to canonical only after the invariance check passes.
#
# --wmsp-only pins the wMSP family to the three registry variants (method-set freeze, author
# 2026-08-11): the other variants (seg_flat, seg_softmax, blondel ablations) stay on the
# do-not-rerun list as frozen negative evidence, and are each a separate MLP fit per cell per seed.

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

OUT="results/analysis/pdl_qwenclean_${EV}__$(echo "$MODEL" | tr '/' '_').csv"

echo "=== CLEAN-SPAN ladder $EV (L$LAYER) on $(hostname) at $(date) ==="
echo "LUQ_REGIME=$LUQ_REGIME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

python -u scripts/checks/probedriftlong.py --model "$MODEL" --layer "$LAYER" \
    --evals "$EV" --seeds 1,2,3 --out "$OUT" \
    --wmsp-only wmsp_norm,wmsp_shrink2,wmsp_shrink10 \
    || { echo "!!! clean ladder FAILED for $EV"; exit 1; }

echo "=== clean ladder $EV done at $(date) ==="
