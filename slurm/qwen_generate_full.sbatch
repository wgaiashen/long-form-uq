#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_gen
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_generate_full_%A_%a.out

# FULL Qwen2.5-14B generation, all 8 long-form datasets, 18,464 rows. This is the long pole of the
# second-model replication; everything downstream (features, per-token, labelling, the ladder) waits
# on the records this produces.
#
# ⚠️ BUDGETS ARE LLAMA'S, UNCHANGED. Decided 2026-08-08 after the budget probe. Two datasets sit below
# the pre-registered 1.0 gate -- med_quad at 0.21x and asqa at 0.96x -- and BOTH are kept anyway,
# deliberately:
#   * med_quad's defect is INHERITED from Llama (0.22x there), not introduced by Qwen.
#   * asqa's is tokeniser efficiency on identical gold text: Llama's own ratio is 1.02x, and Qwen
#     tokenises the same references ~6% longer. Not a difference in model behaviour.
# Changing a budget on Qwen alone would make the two populations differ in budget, which is precisely
# the confound the replication exists to avoid. Both are reported as limitations, not silently fixed.
#
# ⚠️ WHY a100 AND ONLY a100. fp32 is required for comparability with the Llama caches (token
# probabilities ARE the msp_min signal, so an fp16/bf16 cache is not comparable). fp32 Qwen-14B is
# ~59GB of weights, which does not fit a 48GB A40 or a 24GB A30. Sharding across 2xA40 would need a
# code change AND two GPUs per job against a 3-GPU cap -- strictly worse than queueing here.
#
# ⚠️ --max-new-tokens IS PASSED EXPLICITLY AND EQUALS THE TABLE VALUE. It changes nothing (budget =
# min(want, cap) either way), but it forces 01_extract to print the resolved BUDGET line, which is the
# only way the log can EVIDENCE that the budget was not silently clipped to the 128 default. Trap 4.2
# is invisible precisely when you do not ask for it to be printed.
#
# Resume is safe: 01_extract checkpoints every 200 examples and skips what is already cached, so a
# walltime kill costs queue time, not work. 12h is chosen over a longer request on purpose -- the a100
# partition is saturated with short jobs, and a smaller ask backfills sooner.
#
#   sbatch slurm/qwen_generate_full.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}"

MODEL="Qwen/Qwen2.5-14B"

# dataset : budget (LLAMA's, unchanged) : extra flags
# Ordered LONGEST-FIRST. With only 3 running at once, starting the expensive datasets last would
# leave one straggler running alone at the end; longest-first minimises total wall-clock.
TASKS=(
  "cnn_dailymail:128:"                                                        # 3800 rows, ~5.6k-token sources
  "expertqa:384:--prompt-regime expertqa_rp12 --repetition-penalty 1.2"       # 2016 rows, longest budget
  "pubmed_qa:128:--truncate-long"                                             # 3800 rows, newline-stopped
  "xsum:56:"                                                                  # 3800 rows, ~5.6k-token sources
  "med_quad:128:"                                                             # 1800 rows
  "asqa:256:--prompt-regime asqa_rp12 --repetition-penalty 1.2"               # 948 rows
  "samsum:56:"                                                                # 1800 rows
  "factscore:256:--prompt-regime factscore_rp12 --repetition-penalty 1.2"     # 500 rows, eval-only
)
IFS=':' read -r DS BUD EXTRA <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "=== $DS (budget $BUD, FULL dataset) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ⚠️ --dtype fp32 --attn eager passed EXPLICITLY. `auto` resolves to fp16 for any non-Gemma model;
# 01_extract refuses `auto` for a non-Llama model rather than producing a quietly incomparable cache.
# No --limit and no --sample-n: this is the whole dataset, not a subset.
# shellcheck disable=SC2086
python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype fp32 --attn eager \
    --max-new-tokens "$BUD" --max-new-tokens-cap "$BUD" $EXTRA

echo "=== $DS done at $(date) ==="
