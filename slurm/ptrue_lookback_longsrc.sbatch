#!/bin/bash
#SBATCH --job-name=ptrue-lookback-long
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ptrue_lookback_long-%j.out
#
# P(True) + Lookback Lens for the LONG-SOURCE sets: xsum and cnn_dailymail.
#
#   sbatch slurm/ptrue_lookback_longsrc.sbatch xsum
#   sbatch slurm/ptrue_lookback_longsrc.sbatch cnn_dailymail
#
# ⛔ DO NOT SUBMIT until the regeneration question is settled for these two. Written in advance so
# it is ready the moment it unblocks.
#
# STATE OF THAT DECISION (samsum pilot, 1 Aug): the pilot confirmed OVER-generation rather than
# truncation, so raising the cap does not help. That removes the original reason to regenerate
# these two. What is left is the weaker one, unifying the decoding regime. If that is not judged
# worth it, they keep their v1 generations and this script goes straight in; if it is, the records
# change underneath it and the run waits for the new ones. Not decided yet either way.
#
# ONE DATASET PER SUBMISSION is preferred. Together they are ~6.3 h, which queues worse than two
# jobs and loses everything if the second one walltimes.
#
# ---- WHY a100 AND 96G, WHICH IS THE WHOLE REASON THIS IS A SEPARATE SCRIPT ----------------------
# 01c loads the model in fp32 (eager attention in fp16 overflows to NaN in the softmax), so the
# weights alone are ~32 GB. On top of that `recompute_states` sets output_attentions=True, which
# materialises every block's attention at once: 32 layers x 32 heads x seq^2 x 4 B. Both of these
# sets run into `lookback_vector`'s max_seq=2048 cap, so that term is 17.2 GB:
#
#     32 * 32 * 2048^2 * 4 B = 17.2 GB   ->   17.2 + 32 = ~49 GB, past a 48 GB a40
#
# Then the attentions are moved to CPU and `torch.stack`-ed, which copies, so host RAM peaks at
# roughly 2 x 17.2 = ~35 GB. Hence 96G. (For contrast, asqa/expertqa/factscore peak at 1.2 GB of
# attention and ran fine on an a40 with 64G -- the card follows the LONGEST SEQUENCE, never the row
# count.)
#
# ---- SIZING (fitted on measured runs, not guessed) ----------------------------------------------
# Lookback minutes ~ k * sum(min(seq,2048)^2); P(True) minutes ~ k * sum(min(seq,2048)) since it is
# one fp16 SDPA forward. Fitted on the seven measured extractions (1 July + 31 July job logs); the
# fit reproduces pubmed_qa 46.9 vs 45.7 actual and xsum 103.5 vs 103.8 actual.
#
#   dataset        rows  med tok  max tok  >2048  attn peak   ptrue   lookback   total   output
#   xsum           3800      428     5866     21    17.2 GB   8 min    104 min   ~1.9 h    55 MB
#   cnn_dailymail  3800      822     2467     54    17.2 GB  14 min    251 min   ~4.4 h    55 MB
#
# xsum's max sequence is 5866 tokens, so 21 rows lose their oldest context to the 2048 cap (cnn: 54).
# That truncation is `lookback_vector`'s documented behaviour, identical to how the existing v1 xsum
# cache was built, so it is not a new deviation -- but it is worth stating in the writeup.
#
# ⚠️ IF THESE ARE REGENERATED, THE TIMINGS ABOVE BECOME OVERESTIMATES -- BUT THE PARTITION DOES NOT
# CHANGE. Regeneration would use repetition-penalty 1.2, which makes generations SHORTER, not longer
# (samsum's median generation went 56 -> 21 in the pilot). The numbers above are measured off the
# CURRENT v1 records, so they would be a little high. Only a little, though, because the source
# dominates the sequence here:
#
#     xsum           median prompt 390, median generation  33  ->  generation is  7.7% of the sequence
#     cnn_dailymail  median prompt 765, median generation  39  ->  generation is  4.7% of the sequence
#
# Lookback cost goes as sum(min(seq,2048)^2), so even in the absurd limit of generations shrinking to
# NOTHING, cost only falls to 91% (xsum) / 90% (cnn) of the estimates above. That is the floor.
#
# The consequence that matters: DO NOT DOWNSIZE THE PARTITION on the strength of shorter generations.
# The attention peak follows the MAXIMUM sequence, which is almost entirely the article, and both
# sets already saturate the 2048 cap. 17.2 GB + 32 GB of fp32 weights still needs the 80 GB card no
# matter what the generations do.

set -uo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then echo "usage: sbatch $0 <xsum|cnn_dailymail> [...]"; exit 2; fi

# Allowlist. These two live in the TOP-LEVEL cache namespace (no prompt regime) -- unlike
# asqa/expertqa/factscore, which are namespaced under cache/<dataset>_rp12/. Sending a short set
# here would work but would waste an a100 on a job an a40 can do.
regime_for() {
  case "$1" in
    xsum|cnn_dailymail) echo "" ;;
    med_quad|samsum)
      echo "REFUSING '$1': use slurm/ptrue_lookback_midsrc.sbatch -- it fits a 48 GB card and this" \
           "script would hold an a100 for no reason." >&2; return 1 ;;
    *) echo "REFUSING '$1': not a long-source dataset. asqa/expertqa/factscore are already done" \
            "(slurm/ptrue_lookback_v1perm.sbatch)." >&2; return 1 ;;
  esac
}
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

# ---- WHICH CACHE ROOT: v1 OR A REGENERATED v2 --------------------------------------------------
# A v2 regeneration does NOT overwrite v1; it writes into its own prompt-regime namespace, e.g.
# `01_extract.py --prompt-regime v2_xsum` -> cache/v2_xsum/. So after regeneration there
# are TWO record sets on disk and the dataset name alone no longer identifies one.
#
# Left to itself this script would read the v1 records and produce v1 features under a name that
# looks current, which is the silent-wrong-input failure this project keeps paying for. So the
# cache root is explicit, using the SAME LUQ_REGIME convention the ladder drivers use
# (scripts/checks/attn_pool.py), one syntax across the repo:
#
#     sbatch slurm/ptrue_lookback_longsrc.sbatch cnn_dailymail                    # v1, on purpose
#     LUQ_REGIME="cnn_dailymail=v2_cnn" sbatch slurm/... cnn_dailymail           # the v2 cache
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
# 01b/01c have NO skip-if-exists and NO resume: they recompute from scratch and overwrite whatever
# is at the output path. xsum ALREADY HAS both caches (written 2026-07-01), and they are the
# v1-permanent files the format gate snapshots. So an accidental rerun would destroy a reference
# cache silently. Refuse unless the overwrite is stated out loud:
#
#     LUQ_OVERWRITE=1 sbatch slurm/ptrue_lookback_longsrc.sbatch xsum
#
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

# Check the guards above WITHOUT running anything:  LUQ_DRYRUN=1 bash slurm/<this> xsum
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
LIVE=logs/live_ptrue_lookback_long_${SLURM_JOB_ID:-manual}.log
{
  echo "start on $(hostname) at $(date)  datasets: ${DATASETS[*]}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

  for DS in "${DATASETS[@]}"; do
    RG=$(regime_of "$DS"); ROOT="cache${RG:+/$RG}"
    # PROVENANCE. Stamp WHICH generation these features came from. If a dataset is later
    # regenerated, this line is what tells you whether a cache on disk is v1 or v2 -- the .npz
    # itself carries no such record, and "which generation was this?" is exactly the question the
    # v1/v2 split makes expensive to get wrong.
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
