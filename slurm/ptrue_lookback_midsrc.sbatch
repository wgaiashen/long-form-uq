#!/bin/bash
#SBATCH --job-name=ptrue-lookback-mid
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ptrue_lookback_mid-%j.out
#
# P(True) + Lookback Lens for the MID-LENGTH sets: med_quad and samsum.
#
#   sbatch slurm/ptrue_lookback_midsrc.sbatch samsum
#   sbatch slurm/ptrue_lookback_midsrc.sbatch med_quad
#
# DO NOT SUBMIT until these two are regenerated (or the regeneration is called off). Both are
# on the regeneration list, so features extracted from their CURRENT generations would be thrown
# away. Written in advance so it is ready the moment that resolves.
#
# STATE OF THAT DECISION (samsum pilot, 1 Aug): the pilot confirmed OVER-generation rather than
# truncation, so raising the cap does not help, which removes the original reason to regenerate
# samsum. What is left is the weaker one, unifying the decoding regime. Not decided yet.
#
# ---- WHY THIS IS SPLIT FROM ptrue_lookback_longsrc.sbatch --------------------------------------
# Same code, different card. Neither of these reaches `lookback_vector`'s max_seq=2048 cap, so the
# attention tensor (32 layers x 32 heads x seq^2 x 4 B) is much smaller than xsum's 17.2 GB:
#
#     med_quad  max 1358 tok ->  7.6 GB   ->  7.6 + 32 (fp32 weights) = ~40 GB, fits a 48 GB a40
#     samsum    max 1117 tok ->  5.1 GB   ->  5.1 + 32                = ~37 GB, fits a 48 GB a40
#
# So these do NOT need an 80 GB card, and asking for one would just queue behind jobs that do.
# Host RAM peaks at roughly 2 x the attention size (the attentions are moved to CPU and then
# torch.stack-ed, which copies), so 64G is comfortable at ~15 GB.
#
# ---- SIZING (fitted on measured runs, not guessed) ----------------------------------------------
# Lookback minutes ~ k * sum(min(seq,2048)^2); P(True) minutes ~ k * sum(min(seq,2048)), since it
# is one fp16 SDPA forward. Fitted on the seven measured extractions (1 July + 31 July job logs);
# the fit reproduces pubmed_qa 46.9 vs 45.7 actual and xsum 103.5 vs 103.8 actual.
#
#   dataset    rows  med tok  max tok  >2048  attn peak   ptrue   lookback   total   output
#   med_quad   1800     1333     1358      0     7.6 GB   10 min    220 min   ~3.8 h    26 MB
#   samsum     1800      176     1117      0     5.1 GB    2 min     10 min   ~0.2 h    26 MB
#
# med_quad is the surprise here. It is only 1800 rows, half of xsum's 3800, but its MEDIAN
# sequence is 1333 tokens against xsum's 428 -- almost every row is long, whereas xsum is a short
# median with a long tail. Lookback cost goes as seq^2 summed over rows, so med_quad lands at
# ~3.8 h. Do not size it by row count.
#
# IF THESE ARE REGENERATED, THE TIMINGS ABOVE BECOME OVERESTIMATES -- BUT THE PARTITION DOES NOT
# CHANGE. Regeneration would use repetition-penalty 1.2, which makes generations SHORTER, not longer
# (samsum's median generation went 56 -> 21 in the pilot). The numbers above are measured off the
# CURRENT v1 records, so they would be a little high. How much depends on how much of the sequence
# is the source rather than the generation:
#
#     med_quad  median prompt 1205, median generation 128  ->  generation is  9.6% of the sequence
#     samsum    median prompt  132, median generation  56  ->  generation is 31.8% of the sequence
#
# Lookback cost goes as sum(min(seq,2048)^2), so even in the absurd limit of generations shrinking to
# NOTHING, cost only falls to 82% (med_quad) / 70% (samsum) of the estimates above. That is the floor.
# samsum is the more sensitive of the two, but its lookback is ~10 min either way, so it does not
# matter; med_quad's ~3.8 h barely moves, because its sequence is nearly all source.
#
# The consequence that matters: DO NOT DOWNSIZE THE PARTITION on the strength of shorter generations.
# The attention peak follows the MAXIMUM sequence, which is dominated by the source. Both sets still
# need a >32 GB card, and this script should stay on a40/a100 regardless.
#
# Re-measure from the new records if you want exact timings, but the resource class is settled.

set -uo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then echo "usage: sbatch $0 <med_quad|samsum> [...]"; exit 2; fi

# Allowlist. Both live in the TOP-LEVEL cache namespace (no prompt regime), unlike
# asqa/expertqa/factscore which are namespaced under cache/<dataset>_rp12/.
regime_for() {
  case "$1" in
    med_quad|samsum) echo "" ;;
    xsum|cnn_dailymail)
      echo "REFUSING '$1': use slurm/ptrue_lookback_longsrc.sbatch -- both hit the 2048 cap, which" \
           "needs ~49 GB and would OOM the a40 this script may land on." >&2; return 1 ;;
    *) echo "REFUSING '$1': not a mid-length dataset. asqa/expertqa/factscore are already done" \
            "(slurm/ptrue_lookback_v1perm.sbatch)." >&2; return 1 ;;
  esac
}
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

# ---- WHICH CACHE ROOT: v1 OR A REGENERATED v2 --------------------------------------------------
# A v2 regeneration does NOT overwrite v1; it writes into its own prompt-regime namespace, e.g.
# `01_extract.py --prompt-regime v2_med_quad` -> cache/v2_med_quad/. So after regeneration there
# are TWO med_quad record sets on disk and the dataset name alone no longer identifies one.
#
# Left to itself this script would read the v1 records and produce v1 features under a name that
# looks current, which is the silent-wrong-input failure this project keeps paying for. So the
# cache root is explicit, using the SAME LUQ_REGIME convention the ladder drivers use
# (scripts/checks/attn_pool.py), one syntax across the repo:
#
#     sbatch slurm/ptrue_lookback_midsrc.sbatch med_quad                          # v1, on purpose
#     LUQ_REGIME="med_quad=v2_med_quad" sbatch slurm/... med_quad                 # the v2 cache
#
# It is exported so verify_baseline_features.py resolves the same root (it reads PROMPT_REGIME from
# attn_pool, which honours LUQ_REGIME). Both halves of the job therefore agree by construction
# rather than by us remembering to pass the same string twice.
export LUQ_REGIME="${LUQ_REGIME:-}"
regime_of() {                     # "med_quad=v2_med_quad,samsum=v2_samsum" -> the value for $1, or ""
  local ds="$1" item items
  IFS=',' read -ra items <<< "${LUQ_REGIME:-}"
  for item in "${items[@]:-}"; do
    item="${item// /}"
    [ -z "$item" ] && continue
    case "$item" in "$ds"=*) echo "${item#*=}"; return 0 ;; esac
  done
  echo ""
}

# ---- OVERWRITE GUARD ---------------------------------------------------------------------------
# 01b/01c have NO skip-if-exists and NO resume: they recompute and overwrite whatever is at the
# output path, with no warning. Neither of these has a ptrue/lookback cache today, so this guard
# should not fire -- which is exactly why it is worth having. If it DOES fire, something was
# extracted since this was written and you want to know before it is destroyed.
existing=()
for DS in "${DATASETS[@]}"; do
  RG=$(regime_of "$DS"); ROOT="cache${RG:+/$RG}"        # regime-aware: the v2 root has its own caches
  for M in ptrue_accurate lookback; do
    F="$ROOT/features/meta-llama_Meta-Llama-3.1-8B__${DS}__ID__${M}.npz"
    [ -f "$F" ] && existing+=("$F")
  done
done
if [ ${#existing[@]} -gt 0 ] && [ "${LUQ_OVERWRITE:-0}" != "1" ]; then
  echo "REFUSING: these caches already exist and would be OVERWRITTEN with no warning:"
  printf '  %s\n' "${existing[@]}"
  echo "If that is intended (e.g. re-extracting after regeneration), rerun with LUQ_OVERWRITE=1."
  exit 2
fi

# Check the guards above WITHOUT running anything:  LUQ_DRYRUN=1 bash slurm/<this> med_quad
# Testing a guard by letting the script run until it fails is how you accidentally start a job on a
# login/jump node. This exits before the model is ever loaded.
if [ "${LUQ_DRYRUN:-0}" = "1" ]; then
  echo "DRY RUN: allowlist and overwrite guards passed for: ${DATASETS[*]}"
  echo "DRY RUN: would extract ptrue_accurate + lookback, then run the format gate. Exiting."
  exit 0
fi

bash scripts/tools/preflight_space.sh 1 1 || { echo "ABORTING: preflight failed"; exit 1; }

# DoC Slurm spools --output on the compute node and only copies it back on a clean exit, so a
# walltime or OOM kill loses the log. tee to the shared filesystem as well.
LIVE=logs/live_ptrue_lookback_mid_${SLURM_JOB_ID:-manual}.log
{
  echo "start on $(hostname) at $(date)  datasets: ${DATASETS[*]}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

  for DS in "${DATASETS[@]}"; do
    RG=$(regime_of "$DS"); ROOT="cache${RG:+/$RG}"
    # PROVENANCE. Stamp WHICH generation these features came from. Both of these are regeneration
    # targets, so "is this cache v1 or v2?" is a question that will actually be asked -- and the
    # .npz carries no record of it. The regime plus the sha256 of the records file is the answer.
    R="$ROOT/records/meta-llama_Meta-Llama-3.1-8B__${DS}__ID.jsonl"
    echo "CACHE ROOT $DS: $ROOT  (regime '${RG:-<v1 default>}')"
    if [ ! -f "$R" ]; then
      # Loud, not a fallback to v1. Asking for a regime whose records do not exist means the
      # regeneration has not landed, and quietly reading the v1 file instead would be the whole
      # bug this block exists to prevent.
      echo "!!! REFUSING $DS: no records at $R -- the regime was named but that cache is not there."
      continue
    fi
    echo "RECORDS PROVENANCE $DS: $(wc -l < "$R") rows  $(stat -c '%s bytes  mtime %y' "$R")"
    echo "RECORDS SHA256 $DS: $(sha256sum "$R" | cut -d' ' -f1)"

    echo "================ $DS : 01b_ptrue (fp16 sdpa, L15 only) ================ $(date)"
    python scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" \
        --name ptrue_accurate --store-layers 15 --prompt-regime "$RG" \
      || echo "!!! ptrue_accurate extract $DS FAILED"

    echo "================ $DS : 01c_lookback (fp32 eager) ================ $(date)"
    python scripts/01c_lookback.py --dataset "$DS" --ood ID --model "$MODEL" \
        --prompt-regime "$RG" \
      || echo "!!! lookback extract $DS FAILED"
  done

  echo "================ format gate ================ $(date)"
  python scripts/checks/verify_baseline_features.py --datasets "${DATASETS[@]}" \
    || echo "!!! VERIFY FAILED -- do NOT sync these files to RCS"

  echo "ALL DONE at $(date)"
  getfattr --absolute-names -n ceph.dir.rbytes /vol/gpudata/gs925-msc_project 2>/dev/null \
    | awk -F\" '/rbytes/{printf "quota used after run: %.2f GB\n",$2/1e9}'
} 2>&1 | tee "$LIVE"
