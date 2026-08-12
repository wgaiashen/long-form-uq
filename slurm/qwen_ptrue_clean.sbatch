#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_ptrue_clean
#SBATCH --output=logs/qwen_ptrue_clean_%j.out

# P(True) redo on the CLEAN-SPAN population, for the 4 datasets the 2026-08-11/12 correction
# touched (samsum, med_quad, expertqa, factscore). Sibling of slurm/qwen_ptrue.sbatch (read that
# one for the full fp16/SDPA-vs-fp32, layer-23, why-a40 reasoning -- unchanged here).
#
# job 272317 already ran P(True) for all 8 datasets on the RAW-SPAN population (2026-08-10). This
# job does NOT touch that cache -- it points --prompt-regime at the *_trunc_v1 namespaces, which
# resolve to a different cache root entirely, so the raw-span features stay exactly as they were.
# The other 4 datasets (pubmed_qa, asqa, xsum, cnn_dailymail) are untouched by the correction, so
# their existing raw-span P(True) features are already the final ones -- nothing to redo there.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1

MODEL="Qwen/Qwen2.5-14B"
SLUG="Qwen_Qwen2.5-14B"
LAYER=23

# dataset -> CLEAN regime (matches build_truncated_records.py's namespace map)
regime_for() {
  case "$1" in
    samsum|med_quad) echo "trunc_v1" ;;
    expertqa)  echo "expertqa_rp12_trunc_v1" ;;
    factscore) echo "factscore_rp12_trunc_v1" ;;
    *) echo "REFUSING '$1': not one of the 4 clean-span-affected datasets." >&2; return 1 ;;
  esac
}

DATASETS=(factscore expertqa samsum med_quad)   # cheapest first, same ordering convention as the raw job
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

echo "=== qwen ptrue CLEAN-SPAN (L$LAYER) on $(hostname) at $(date)  datasets: ${DATASETS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for EV in "${DATASETS[@]}"; do
  RG=$(regime_for "$EV")
  ROOT="${LUQ_CACHE_ROOT:-cache}/$RG"
  OUT="$ROOT/features/${SLUG}__${EV}__ID__ptrue_accurate.npz"
  REC="$ROOT/records/${SLUG}__${EV}__ID.jsonl"

  echo "================ $EV (regime '$RG') ================ $(date)"
  [ -f "$REC" ] || { echo "!!! no records at $REC -- skipping $EV"; continue; }
  echo "RECORDS $EV: $(wc -l < "$REC") rows"
  echo "RECORDS SHA256 $EV: $(sha256sum "$REC" | cut -d' ' -f1)"

  if [ -f "$OUT" ] && [ "${LUQ_OVERWRITE:-0}" != "1" ]; then
    echo "SKIPPING $EV: $OUT exists. LUQ_OVERWRITE=1 to force (also how a resubmission resumes)."
    continue
  fi

  python -u scripts/01b_ptrue.py --model "$MODEL" --dataset "$EV" --ood ID \
      --name ptrue_accurate --store-layers "$LAYER" --prompt-regime "$RG" \
      || { echo "!!! ptrue FAILED for $EV"; continue; }
  ls -la "$OUT"
done

echo "=== qwen ptrue CLEAN-SPAN ALL DONE at $(date) ==="
