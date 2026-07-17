#!/bin/bash
#SBATCH --job-name=pertok_cnn
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pertok_cnn-%j.out

# v2 of cnn_dailymail_pertoken_doc.sbatch. TWO changes vs the original, both operational (workload
# identical):
#   1. walltime 5h -> 12h. The original TIMED OUT at exactly 5h without finishing. cnn's forward is
#      mean 896 / max 2467 tokens per record (vs xsum's mean 529), so at ~1.7x xsum's per-record cost
#      plus heavier quadratic attention, the full fp32 3800-record pass needs ~6h -- more than the 5h
#      the original allowed. xsum's equivalent pass (lighter) finished inside 4h, which is the anchor.
#   2. Live log to SHARED storage via tee. DoC's Slurm spools the --output file on the compute node
#      and only copies it back on CLEAN exit, so a walltime/OOM kill loses it (that is why the first
#      two attempts left no log). Teeing to logs/live_pertok_cnn_<jobid>.log on the shared FS with
#      line-buffered, unbuffered output makes progress ("200/3800" ...) visible during the run and
#      survives an unclean kill, so a hang can be told apart from slow-but-progressing.
#
# Prereqs unchanged: the LABELLED records must already be in cache/records (01h's SANITY gate needs
# r["correctness"]); the SAPLMA L15 feature cache must exist (the gate compares against it). Both were
# verified present. After it finishes, rsync the pertok cache DoC->RCS and verify size + np.load():
#   cache/pertok/meta-llama_Meta-Llama-3.1-8B__cnn_dailymail__ID__L15.npz

set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export PYTHONUNBUFFERED=1
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B
LIVE=logs/live_pertok_cnn_${SLURM_JOB_ID}.log

{
  echo "=== node ==="; hostname; date
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo; echo "########## cnn_dailymail per-token L15 (fp32) ##########"
  stdbuf -oL -eL python -u scripts/01h_pertoken.py --model "$MODEL" --dataset cnn_dailymail --layer 15
  echo; echo "=== DONE $(date) ==="
} 2>&1 | tee "$LIVE"
