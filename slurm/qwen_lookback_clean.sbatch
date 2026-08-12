#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=qwen_lb_clean
#SBATCH --output=logs/qwen_lookback_clean_%j.out

# Lookback Lens on Qwen2.5-14B, FIRST successful attempt -- both earlier tries (jobs 272314,
# 272316) were cancelled mid-run with nothing usable produced, so there is no "redo" risk here,
# just a fresh extraction. Six SHORT-SOURCE evals (fits one A100); xsum/cnn_dailymail need two
# cards, use slurm/qwen_lookback_longsrc.sbatch UNCHANGED (both are untouched by the clean-span
# correction, so the existing script is already exactly right for them).
#
# Sibling of slurm/qwen_lookback.sbatch (read that one for the full memory arithmetic, fp32+eager
# reasoning, and per-dataset cost table -- unchanged here). The only difference is the regime map:
# samsum/med_quad/expertqa/factscore point at their CLEAN-SPAN namespaces (the 2026-08-11/12
# correction), pubmed_qa/asqa are untouched so they use their normal base regimes.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1

MODEL="Qwen/Qwen2.5-14B"
SLUG="Qwen_Qwen2.5-14B"

# Cheapest first (same convention as qwen_lookback.sbatch's cost table), CLEAN regime where the
# correction touched the dataset, base regime otherwise.
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(factscore asqa samsum expertqa pubmed_qa med_quad)
fi

regime_for() {
  case "$1" in
    pubmed_qa) echo "" ;;
    asqa)      echo "asqa_rp12" ;;
    samsum|med_quad) echo "trunc_v1" ;;
    expertqa)  echo "expertqa_rp12_trunc_v1" ;;
    factscore) echo "factscore_rp12_trunc_v1" ;;
    xsum|cnn_dailymail)
      echo "REFUSING '$1': needs two cards, use slurm/qwen_lookback_longsrc.sbatch (unchanged, it" \
           "is untouched by the clean-span correction)." >&2; return 1 ;;
    *) echo "REFUSING '$1': not a ProbeDriftLong long eval." >&2; return 1 ;;
  esac
}
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

echo "=== qwen lookback CLEAN-SPAN on $(hostname) at $(date)  datasets: ${DATASETS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in "${DATASETS[@]}"; do
  RG=$(regime_for "$DS")
  ROOT="${LUQ_CACHE_ROOT:-cache}${RG:+/$RG}"
  OUT="$ROOT/features/${SLUG}__${DS}__ID__lookback.npz"
  REC="$ROOT/records/${SLUG}__${DS}__ID.jsonl"

  echo "================ $DS (regime '${RG:-<base>}') ================ $(date)"
  [ -f "$REC" ] || { echo "!!! no records at $REC -- skipping $DS"; continue; }
  echo "RECORDS $DS: $(wc -l < "$REC") rows"
  echo "RECORDS SHA256 $DS: $(sha256sum "$REC" | cut -d' ' -f1)"

  if [ -f "$OUT" ] && [ "${LUQ_OVERWRITE:-0}" != "1" ]; then
    echo "REFUSING $DS: $OUT exists and would be silently overwritten. LUQ_OVERWRITE=1 to force."
    continue
  fi

  python -u scripts/01c_lookback.py --model "$MODEL" --dataset "$DS" --ood ID \
      --prompt-regime "$RG" \
      || { echo "!!! lookback FAILED for $DS"; continue; }
  ls -la "$OUT"
done

echo "=== qwen lookback CLEAN-SPAN ALL DONE at $(date) ==="
