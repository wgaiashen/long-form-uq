#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_rsamp
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_degeneracy_rsample_%A_%a.out

# RE-MEASURE Qwen's degeneracy on a RANDOM sample, not the head slice.
#
# WHY THIS RUN EXISTS. The budget probe used `--limit 200`, which is a head slice, not a sample:
# `01_extract` simply breaks after the first n rows. Proven on expertqa -- the first 200 rows have gold
# p90 450 against 349 for the full 2016, and 450 sits OUTSIDE the [316,376] range of 2,000 random
# 200-row draws. So every statistic read off that run (length, capping, AND degeneracy) is biased.
#
# The degeneracy numbers now matter more than the budget numbers did. Degenerate generations carry no
# adjudicable claim, which is what drives a row to `uncovered == 1.0` and so out of the labelled set.
# That is the coverage divergence pre-registered in prereg/qwen_coverage_divergence_prediction.md,
# whose §3 explicitly marks the head-slice column as biased and to be REPLACED by this run. The
# formula and the thresholds in M4 are fixed; only these inputs update.
#
# Head-slice figures being replaced (severe degeneracy, n=200):
#   expertqa 23.0% · factscore 17.0% · asqa 10.0% · med_quad 11.5% · samsum 3.5% · pubmed_qa 1.5%
# against Llama's full-population 11.76 / 0.20 / 1.37 / 12.33 / 0.72 / 1.11.
#
# SEPARATE NAMESPACE, deliberately. This writes to `qwen14b_rsample`, NOT `qwen14b_probe`. The two
# are different populations (random draw vs head slice) and must never be pooled or silently merged
# into one table -- that is the same class of error as pooling two models.
#
# fp32 + eager, matching the real run, on the 80GB A100 (confirmed on-card: 81920 MiB).
#
#   sbatch slurm/qwen_degeneracy_rsample.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset, using cache/ beside the code>}"

MODEL="Qwen/Qwen2.5-14B"
REGIME="qwen14b_rsample"
N=200
SEED=1

# Same raised caps as the budget probe, so generation length is not clipped by the measurement itself.
TASKS=(
  "pubmed_qa:512:--truncate-long"
  "xsum:256:"
  "cnn_dailymail:512:"
  "med_quad:768:"
  "samsum:256:"
  "expertqa:768:--repetition-penalty 1.2"
  "asqa:512:--repetition-penalty 1.2"
  "factscore:512:--repetition-penalty 1.2"
)
IFS=':' read -r DS CAP EXTRA <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "=== $DS (cap $CAP, RANDOM n=$N seed=$SEED) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --sample-n replaces --limit. The drawn indices are stored in each record's `idx`, so the sample is
# auditable after the fact rather than having to be trusted.
# shellcheck disable=SC2086
python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --prompt-regime "$REGIME" --sample-n "$N" --sample-seed "$SEED" \
    --dtype fp32 --attn eager \
    --max-new-tokens "$CAP" --max-new-tokens-cap "$CAP" $EXTRA

echo "=== $DS done at $(date) ==="
