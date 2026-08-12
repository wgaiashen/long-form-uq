#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_ptunsup
#SBATCH --output=logs/qwen_ptrue_unsup_%j.out

# Unsupervised P(True) (the Kadavath training-free baseline -- report method 11: "P(True),
# training-free") for Qwen2.5-14B, first time this method has ever been run for this population.
# Sibling of slurm/ptrue_unsup_doc.sbatch (read that one for the resource/dtype reasoning: a40/48G
# not a100, because 01g only needs next-token logits, no output_attentions, no all-layer states).
#
# WHY ONLY 6 DATASETS, NOT 8. A pre-flight check (`score_ptrue_unsup.py --model Qwen/Qwen2.5-14B`,
# run against whatever was on the CANONICAL records) showed asqa and cnn_dailymail already have
# ptrue_unsup scored -- and since both are untouched by the 2026-08-11/12 clean-span correction,
# those canonical-population numbers are already the final ones. pubmed_qa and xsum have never been
# scored at all. med_quad/samsum/expertqa/factscore DID have values, but on the CANONICAL (raw-span)
# records -- stale for our purposes, since the whole point is the clean population. Verified their
# clean-regime records (trunc_v1 / *_rp12_trunc_v1) have 0/N ptrue_unsup before writing this job.
#
# 01g writes ptrue_unsup/ptrue_unsup_mass/ptrue_unsup_model IN PLACE into the targeted records. For
# the 4 affected datasets that means the CLEAN regime's records, a completely separate file from the
# canonical ones -- no risk of touching the stale canonical values already on disk.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1

MODEL="Qwen/Qwen2.5-14B"

regime_for() {
  case "$1" in
    pubmed_qa|xsum) echo "" ;;
    med_quad|samsum) echo "trunc_v1" ;;
    expertqa)  echo "expertqa_rp12_trunc_v1" ;;
    factscore) echo "factscore_rp12_trunc_v1" ;;
    *) echo "REFUSING '$1': not in the 6-dataset plan for this job." >&2; return 1 ;;
  esac
}

DATASETS=(pubmed_qa xsum med_quad samsum expertqa factscore)
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

echo "=== qwen ptrue-unsup on $(hostname) at $(date)  datasets: ${DATASETS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in "${DATASETS[@]}"; do
  RG=$(regime_for "$DS")
  echo "================ $DS (regime '${RG:-<base>}') ================ $(date)"
  # 01g is resumable (skips records already scored unless --overwrite), so a walltime kill costs
  # only the dataset in flight and a plain resubmission continues.
  python -u scripts/01g_ptrue_unsup.py --model "$MODEL" --dataset "$DS" --ood ID \
      --prompt-regime "$RG" \
      || { echo "!!! ptrue-unsup FAILED for $DS"; continue; }
done

echo "=== qwen ptrue-unsup ALL DONE at $(date) ==="
