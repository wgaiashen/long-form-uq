#!/bin/bash
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=wm_llamailabel
#SBATCH --array=0-1
#SBATCH --output=logs/wmodels_llamai_label_%A_%a.out

# Label the two Llama-3.1-8B-Instruct D6-extension datasets that generation_quality.py already
# verified clean (med_quad's clean_v2 span cut: 0/1800 rows needed cutting; expertqa: 0.0% leak,
# 1.19% severe/degraded). API-bound, no GPU compute -- every DoC partition's QoS forces
# --gres=gpu:1 regardless of whether the job needs one, not a real requirement here.
#
# SUBMITTED AS A REAL SLURM JOB, NOT RUN INTERACTIVELY, on purpose (2026-08-25): a background
# process left running in an interactive login shell on this cluster is not reliable across a long
# API-bound job -- the shell (and once, the whole tmux server) has been torn down mid-job, killing
# anything backgrounded with nohup/tmux, while Slurm-tracked work survives the same event
# untouched. So: anything that must run unattended for hours goes through sbatch.
#
#   sbatch slurm/wmodels_llamai_label.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  KEYFILE="/vol/gpudata/gs925-msc_project/.openai_key"
  [ -f "$KEYFILE" ] && { . "$KEYFILE"; echo "OPENAI_API_KEY loaded (value not shown)" >&2; }
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set and no key file found}"

MODEL="meta-llama/Llama-3.1-8B-Instruct"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    echo "=== med_quad, clean_v2 regime ===" >&2
    python -u scripts/02_label.py --model "$MODEL" --dataset med_quad --ood ID \
        --prompt-regime llama31i_chat_med_quad_clean_v2 --judge gpt-5-mini
    ;;
  1)
    echo "=== expertqa, llama31i_chat regime (dedicated three-state labeller) ===" >&2
    python -u scripts/02_label_expertqa.py --model "$MODEL" \
        --prompt-regime llama31i_chat --judge gpt-5-mini
    ;;
  *)
    echo "!!! unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 3 ;;
esac
STATUS=$?
echo "=== task $SLURM_ARRAY_TASK_ID done status=$STATUS at $(date) ===" >&2
exit $STATUS
