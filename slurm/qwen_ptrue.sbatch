#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_ptrue
#SBATCH --output=logs/qwen_ptrue_%j.out

# P(True) verdict-position features on Qwen2.5-14B, all eight long evals in one job.
#
#   sbatch slurm/qwen_ptrue.sbatch                       # all 8, in cost order
#   sbatch slurm/qwen_ptrue.sbatch pubmed_qa xsum        # or a named subset
#
# ONE JOB RATHER THAN AN 8-TASK ARRAY, on purpose: DoC caps submitted jobs per user (an array
# counts as one submission PER TASK, and an 8-task array was rejected outright with
# QOSMaxSubmitJobPerUserLimit while the ladder and the lookback jobs were queued). P(True) is one
# fp16 forward per row -- Llama measured 8-14 min per long dataset -- so the whole set is a few
# hours sequentially and does not need the parallelism.
#
# WHY THIS EXISTS. prereg/M2_qwen14b_replication.md §4 names P(True) and Lookback in the scored
# baseline set, but neither had a Qwen feature cache -- cache/features held only __saplma.npz for
# all 8 -- so the first ladder run left both columns ABSENT. Absent is honest, but it is a stated
# gap against our own pre-registration, so we close it. No registered claim (R1a/R1b/R2/R2-desc/
# R3/R4') reads either method, so nothing about the replication verdict depends on this job.
#
# WHY fp16 + SDPA AND NOT fp32, WHEN EVERY OTHER QWEN ARTIFACT IS fp32. Because that is what the
# Llama P(True) cache is, and this is a configuration TRANSFER, not a re-selection.
# slurm/llama_ptrue_accurate.sbatch says so in as many words: "Uses the default forward (fp16/SDPA)
# to match how the unsupervised ptrue and the Gemma ptrue features were extracted -- the ptrue
# family's own convention." Llama generation was fp32 and its P(True) was fp16 for the same reason.
# Matching Llama here is the faithful choice; matching the rest of the Qwen pipeline would be a
# silent change of method between the two populations, which is the comparison this whole
# workstream exists to make. Stated out loud so the write-up can say it.
#
# WHY --store-layers 23. The layer is fixed by Joe's rule ceil(N/2)-1 on both models (Llama 15,
# Qwen 23) and is not selected, so the other 48 layers are never read. Unstored layers are written
# as NaN rather than dropped, so the array stays full height and a probe of the wrong layer fails
# LOUDLY on the NaN instead of quietly returning a plausible number. Same treatment the Llama
# long-source extraction used (slurm/ptrue_lookback_longsrc.sbatch: --store-layers 15).
#
# ⚠️ DOWNSTREAM PORT NEEDED BEFORE THE LADDER CAN READ THIS. scripts/checks/probedriftlong.py:90
# hardcodes BASE_FEATS = {"ptrue": ("ptrue_accurate", 15, True), ...} -- layer 15, the Llama layer.
# Pointed at Qwen it would index the NaN plane and abort, which is the right failure but still a
# failure. That file is on BOTH workstreams' collision lists, so the edit is RCS's to make and is
# in the hand-off prompt. This job is the predecessor; it does not depend on the port.
#
# WHY a40 AND NOT a100. One fp16 forward per row: 30 GB of weights plus a small activation
# footprint, comfortably inside 48 GB. The a100s are wanted for the lookback extraction, which
# genuinely needs them (see slurm/qwen_lookback.sbatch).
#
# ⛔ 01b HAS NO SKIP-IF-EXISTS AND NO RESUME. It recomputes from scratch and overwrites the output
# path. The guard below refuses to clobber an existing Qwen cache unless LUQ_OVERWRITE=1, and it
# can never touch a Llama file: every path it builds is stamped Qwen_Qwen2.5-14B.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1

MODEL="Qwen/Qwen2.5-14B"
SLUG="Qwen_Qwen2.5-14B"
LAYER=23

# dataset -> cache namespace. The three repetition-penalty sets were generated under their own
# prompt regime, so their records and features live in cache/<ns>/ and NOT the top-level namespace.
# Passing the wrong one here would read a records file that does not exist (loud) -- but getting it
# right is cheaper than finding out.
regime_for() {
  case "$1" in
    pubmed_qa|xsum|cnn_dailymail|med_quad|samsum) echo "" ;;
    asqa)      echo "asqa_rp12" ;;
    expertqa)  echo "expertqa_rp12" ;;
    factscore) echo "factscore_rp12" ;;
    *) echo "REFUSING '$1': not a ProbeDriftLong long eval." >&2; return 1 ;;
  esac
}

# Cheapest first, so a walltime kill costs the most expensive one only. 01b has no resume, but it
# writes one file per dataset, so finished datasets survive a kill and are skipped on resubmission
# by the overwrite guard below.
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(factscore asqa expertqa samsum med_quad pubmed_qa xsum cnn_dailymail)
fi
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

echo "=== qwen ptrue (L$LAYER) on $(hostname) at $(date)  datasets: ${DATASETS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for EV in "${DATASETS[@]}"; do
  RG=$(regime_for "$EV")
  ROOT="${LUQ_CACHE_ROOT:-cache}${RG:+/$RG}"
  OUT="$ROOT/features/${SLUG}__${EV}__ID__ptrue_accurate.npz"
  REC="$ROOT/records/${SLUG}__${EV}__ID.jsonl"

  echo "================ $EV (regime '${RG:-<base>}') ================ $(date)"
  # Provenance: WHICH generation these features came from. The .npz carries no such record.
  [ -f "$REC" ] || { echo "!!! no records at $REC -- skipping $EV"; continue; }
  echo "RECORDS $EV: $(wc -l < "$REC") rows  $(stat -c '%s bytes  mtime %y' "$REC")"
  echo "RECORDS SHA256 $EV: $(sha256sum "$REC" | cut -d' ' -f1)"

  if [ -f "$OUT" ] && [ "${LUQ_OVERWRITE:-0}" != "1" ]; then
    echo "SKIPPING $EV: $OUT exists and 01b would overwrite it with no warning."
    echo "  (this is also how a resubmission after a walltime kill resumes. LUQ_OVERWRITE=1 to force.)"
    continue
  fi

  python -u scripts/01b_ptrue.py --model "$MODEL" --dataset "$EV" --ood ID \
      --name ptrue_accurate --store-layers "$LAYER" --prompt-regime "$RG" \
      || { echo "!!! ptrue FAILED for $EV"; continue; }
  ls -la "$OUT"
done

echo "=== qwen ptrue ALL DONE at $(date) ==="
