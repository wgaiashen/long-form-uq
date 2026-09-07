#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_gen
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_generate_full_%A_%a.out

# FULL Qwen2.5-14B generation, all 8 long-form datasets, 18,464 rows. This is the long pole of the
# second-model replication; everything downstream (features, per-token, labelling, the ladder) waits
# on the records this produces.
#
# BUDGETS ARE LLAMA'S, UNCHANGED. Decided 2026-08-08 after the budget probe. Two datasets sit below
# the pre-registered 1.0 gate -- med_quad at 0.21x and asqa at 0.96x -- and BOTH are kept anyway,
# deliberately:
#   * med_quad's defect is INHERITED from Llama (0.22x there), not introduced by Qwen.
#   * asqa's is tokeniser efficiency on identical gold text: Llama's own ratio is 1.02x, and Qwen
#     tokenises the same references ~6% longer. Not a difference in model behaviour.
# Changing a budget on Qwen alone would make the two populations differ in budget, which is precisely
# the confound the replication exists to avoid. Both are reported as limitations, not silently fixed.
#
# WHY a100 AND ONLY a100. fp32 is required for comparability with the Llama caches (token
# probabilities ARE the msp_min signal, so an fp16/bf16 cache is not comparable). fp32 Qwen-14B is
# ~59GB of weights, which does not fit a 48GB A40 or a 24GB A30. Sharding across 2xA40 would need a
# code change AND two GPUs per job against a 3-GPU cap -- strictly worse than queueing here.
#
# --max-new-tokens IS PASSED EXPLICITLY AND EQUALS THE TABLE VALUE. It changes nothing (budget =
# min(want, cap) either way), but it forces 01_extract to print the resolved BUDGET line, which is the
# only way the log can EVIDENCE that the budget was not silently clipped to the 128 default. Trap 4.2
# is invisible precisely when you do not ask for it to be printed.
#
# Memory: 48G. Measured peak is ~11GB -- one 3.4GB safetensors shard at a time (device_map="cuda"
# streams them to the GPU) plus the pooled-feature list, which is the part that grows with the
# dataset: 49 layers x 5120 x fp32 = 0.96MB per example, so 3.6GB at cnn/xsum/pubmed's 3800 rows.
# This was 96G, which is not free: 4 concurrent jobs at 96G exceed vm-he-a100's 451GB, so an
# oversized ask can leave a job pending on memory while a GPU sits idle.
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

# Allocator only -- changes NOTHING numerically, so it cannot affect comparability with the Llama
# caches. It matters for xsum, whose longest prompt is 5,908 tokens (the probe's head slice only
# reached 1,704, so this was never exercised). At that length fp32 Qwen-14B needs roughly: 55GB of
# weights + 5.5GB of all-layer hidden states + 3.3GB of prefill logits + 2.2GB of KV cache, plus a
# 5.2GB eager-attention buffer that is allocated and freed once PER LAYER, 48 times per forward.
# That last pattern is what fragments the heap, and ~71GB against 79.3GB usable is not much slack.
# Llama had far more room (32GB of weights), which is why this never came up on the first model.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  # med_quad has TWO independent settings that are easy to confuse. Both are pinned to v1 here.
  #
  # (1) BUDGET 128, i.e. v1. med_quad was regenerated at cap 768 (v2), and v2 was ARCHIVED with v1
  #     re-emerging as canonical. v1 is
  #     hash-pinned in scripts/checks/CANONICAL_v1_MANIFEST.sha256 and the file on disk verifies against it.
  #     So this is deliberately the CAPPED version, 97.7% capped and all, not the long one.
  #
  # (2) --no-repeat-ngram-size 3, and NOT --repetition-penalty. Measured on the records rather than
  #     read off a script: 0/1800 canonical med_quad generations contain a repeated 3-gram, against
  #     17-55% on every other dataset (pubmed 17.6% is the working control for the detector). The
  #     archived v2 and v2_span arms are also 0/1800, so the anti-loop flag is CONSTANT across the
  #     whole med_quad lineage -- it is NOT what distinguishes v1 from v2. Mean generation length is
  #     what does: 127 for v1 (cap 128) against 612 for v2 (cap 768).
  #     Do not confuse this with the v1..v4 numbering in pbs/extract_neighbour.pbs, which is a
  #     separate decoding lineage for the OOD neighbour extraction and uses the same words.
  #     rep_pen is specifically excluded: on med_quad it penalises the long few-shot context and
  #     forces immediate EOS, which produced ~35% EMPTY generations. no_repeat_ngram has no such
  #     side effect. Generating Qwen with plain decoding would put the two models on different
  #     decoding populations for the one dataset whose known failure mode is looping.
  "med_quad:128:--no-repeat-ngram-size 3"                                     # 1800 rows
  "asqa:256:--prompt-regime asqa_rp12 --repetition-penalty 1.2"               # 948 rows
  "samsum:56:"                                                                # 1800 rows
  "factscore:256:--prompt-regime factscore_rp12 --repetition-penalty 1.2"     # 500 rows, eval-only
)
IFS=':' read -r DS BUD EXTRA <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "=== $DS (budget $BUD, FULL dataset) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --max-new-tokens IS ONLY LEGAL ALONGSIDE A --prompt-regime, and that guard is right.
# 01_extract:116 refuses the flag without a regime, to stop non-default-budget records being written
# into the default namespace beside the frozen v1 records. It cannot tell that the value we pass
# equals the table value, and it should not have to trust that. An earlier version of this script
# passed it for all eight datasets to force the resolved-budget line to be logged; the five datasets
# with no regime died on the guard two minutes in. The budget is EVIDENCED AFTER THE FACT instead --
# max(len(gen_token_ids)) per dataset must equal the intended budget, which is a stronger check than
# a printed intention anyway, since it reads what was actually generated.
# So: pass it only where a regime exists. The five default-budget datasets take the table value, and
# --max-new-tokens-cap alone (a pure ceiling, no guard) lifts the 128 default where it is too low.
MNT=""
case "$EXTRA" in *--prompt-regime*) MNT="--max-new-tokens $BUD" ;; esac

# --dtype fp32 --attn eager passed EXPLICITLY. `auto` resolves to fp16 for any non-Gemma model;
# 01_extract refuses `auto` for a non-Llama model rather than producing a quietly incomparable cache.
# No --limit and no --sample-n: this is the whole dataset, not a subset.
# shellcheck disable=SC2086
python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype fp32 --attn eager \
    $MNT --max-new-tokens-cap "$BUD" $EXTRA

echo "=== $DS done at $(date) ==="
