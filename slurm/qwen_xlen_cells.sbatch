#!/bin/bash
#SBATCH --partition=a30
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=qwen_xlen_cells
#SBATCH --output=logs/qwen_xlen_cells_%j.out

# The 8 W-CrossLen ladder cells for Qwen (DOC_BRIEF_qwen_crosslength.md §6.2), ONE cell
# per submission via a positional index argument (0-7) -- NOT a Slurm array. An array
# submission of all 8 at once hit QOSMaxSubmitJobPerUserLimit on this account. Not using
# --export for the index either: sbatch --export=ALL,VAR=val gets jobs held on this
# cluster ("user env retrieval failed", a documented trap here) -- a plain positional
# script argument sidesteps that entirely.
#
# CPU-only (floors/SAPLMA/wMSP, no pooler training) -- but still submitted as a batch
# job, not run on the shell: this machine (cloud-vm-40-244) kills long processes started
# directly on it (slurm/qwen_api_jobs.sbatch's own header; confirmed the hard way when
# the no-op control died with no output on a first direct-shell try).
#
# Every pool totals 1,800 (trivia-only matched arms, per the brief's §2 resolution of
# the matched-pool problem -- Qwen has no SciQ, so the Llama-side sciq+trivia arms are
# not directly comparable and RCS built trivia-only Llama arms to match against instead).
#
#   sbatch slurm/qwen_xlen_cells.sbatch <0-7>

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

MODEL="Qwen/Qwen2.5-14B"

# eval : rung-tag (for the --out filename) : rung-name-or-"AUTO" : train-spec (empty = no --train-spec)
TASKS=(
  "asqa:LONGSHORT:LONG+SHORT:pubmed_qa:128,xsum:128,cnn_dailymail:128,med_quad:128,samsum:128,expertqa:128,factscore:128,trivia_qa:904"
  "asqa:SHORT:SHORT:trivia_qa:1800"
  "xsum:LONGSHORT:LONG+SHORT:pubmed_qa:128,cnn_dailymail:128,med_quad:128,samsum:128,expertqa:128,asqa:128,factscore:128,trivia_qa:904"
  "xsum:SHORT:SHORT:trivia_qa:1800"
  "factscore:LONGSHORT:LONG+SHORT:pubmed_qa:128,xsum:128,cnn_dailymail:128,med_quad:128,samsum:128,expertqa:128,asqa:128,trivia_qa:904"
  "factscore:SHORT:SHORT:trivia_qa:1800"
  "trivia_qa:LongToShort:AUTO:"
  "trivia_qa:IDshort:ID-short:trivia_qa:1800"
)
IDX="${1:?usage: sbatch qwen_xlen_cells.sbatch <0-7>}"
LINE="${TASKS[$IDX]}"
EV="${LINE%%:*}"; REST="${LINE#*:}"
TAG="${REST%%:*}"; REST="${REST#*:}"
RUNG="${REST%%:*}"; SPEC="${REST#*:}"

echo "=== cell $IDX: eval=$EV tag=$TAG rung=$RUNG on $(hostname) at $(date) ==="

if [ "$RUNG" = "AUTO" ]; then
  # trivia_qa / Long->Short: no --train-spec -- cells_long emits this rung automatically
  # now that trivia_qa has a cache, per the brief's §6.2 row 7 instruction.
  python -u scripts/checks/probedriftlong.py --model "$MODEL" --layer 23 \
      --seeds 1,2,3 --evals "$EV" --rungs "Long->Short" \
      --wmsp-only wmsp_norm,wmsp_shrink2,wmsp_shrink10 \
      --perex-dir results/perex/xlen_qwen \
      --out "results/xlen_${TAG}_${EV}__Qwen_Qwen2.5-14B.csv"
else
  python -u scripts/checks/probedriftlong.py --model "$MODEL" --layer 23 \
      --seeds 1,2,3 --evals "$EV" \
      --train-spec "$SPEC" --rung-name "$RUNG" \
      --wmsp-only wmsp_norm,wmsp_shrink2,wmsp_shrink10 \
      --perex-dir results/perex/xlen_qwen \
      --out "results/xlen_${TAG}_${EV}__Qwen_Qwen2.5-14B.csv"
fi

echo "=== cell $IDX done at $(date) ==="
