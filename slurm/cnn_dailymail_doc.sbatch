#!/bin/bash
#SBATCH --job-name=luq-cnn-extract
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/cnn_extract-%j.out

# cnn_dailymail generation on DoC's A100-80GB. cnn's source articles are long, so fp32 8B + the
# article prompt OOMs RCS's 48GB cards (the same wall xsum hit) -> this is the ONE workstream that
# genuinely needs DoC. Every other job (ExpertQA per-token, med_quad/samsum, SAR) fits RCS 48GB.
#
# FAITHFUL to the ProbeDrift repo: data.load('cnn_dailymail','ID') dispatches to
# get_datasets(eval_dataset='cnn_dailymail', ood='ID'), which uses ProbeDrift's OWN prompt
# ("Here's the text and its short summary...Summary (one sentence):"), train_split='train',
# eval_split='test', seed=1 2000-subsample (ProbeDrift/probe_drift/dataset_configs.py:102). We do
# NOT diverge for cnn -- it is a first-class ProbeDrift long-form eval target.
#
# Generates BOTH splits in one run (train 1800 + test 2000). 01_extract checkpoints every 200 and
# resumes from cache, so if it hits the 16h wall just resubmit -- it picks up where it left off.
#
# fp32 keeps it dtype-consistent with every other frozen Llama-3.1-8B run (the whole project is fp32).
#
#   sbatch slurm/cnn_dailymail_doc.sbatch
#
# NEXT (after this finishes): rsync records -> RCS, judge on RCS (gpt-5-mini summ), then the
# per-token L15 pass (also DoC 80GB, needs labels first) -- see the runbook in
# PLAN_post_10July_meeting.md P1.8 / W4.

set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo; echo "########## cnn_dailymail generation (train + test), fp32 ##########"
python scripts/01_extract.py --dataset cnn_dailymail --ood ID --model "$MODEL" --dtype fp32

echo; echo "=== DONE $(date) ==="
