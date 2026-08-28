#!/bin/bash
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=wm_llamailabel_eqa_cont
#SBATCH --output=logs/wmodels_llamai_label_eqa_cont_%j.out

# Continuation of job 278651 array task 1 (expertqa labelling for Llama-3.1-8B-Instruct).
# That job's 4-hour time limit will kill it before all 2016 rows are labelled: measured rate
# ~2.76 rows/min (450/2016 after 2h43m -- ~6x slower than med_quad's labelling, plausibly because
# the three-state factuality/uncovered/coherent judge makes more calls per row and expertqa's
# generations are longer). 02_label_expertqa.py checkpoints every 25 rows and skips already-labelled
# ones on restart ("already done: N"), so simply re-running the identical command resumes cleanly --
# no separate resume flag needed.
#
# Submitted with --dependency=afterany:278651_1 so it starts automatically the moment that job ends,
# whether by completion or by hitting its time limit, with no manual resubmission needed.
# If 12h is still not enough, chain another one the same way once this one's own status is checked.
#
#   sbatch --dependency=afterany:278651_1 slurm/wmodels_llamai_label_expertqa_cont.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  KEYFILE="/vol/gpudata/gs925-msc_project/.openai_key"
  [ -f "$KEYFILE" ] && { . "$KEYFILE"; echo "OPENAI_API_KEY loaded (value not shown)" >&2; }
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set and no key file found}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "=== resuming expertqa labelling at $(date) ===" >&2
python -u scripts/02_label_expertqa.py --model "$MODEL" \
    --prompt-regime llama31i_chat --judge gpt-5-mini
STATUS=$?
echo "=== done status=$STATUS at $(date) ===" >&2
exit $STATUS
