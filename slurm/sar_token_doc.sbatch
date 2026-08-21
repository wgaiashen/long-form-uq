#!/bin/bash
#SBATCH --job-name=sar_token
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sar_token-%j.out
#
# Token-level SAR relevance for the long-form + cnn datasets, on DoC (RCS free but this is a long
# GPU job; DoC is idle). The cross-encoder (stsb-roberta-large, ~1.3GB) fits any GPU -- a100 here just
# to skip queues. Token-level (leave-one-TOKEN-out) gives a per-token signal even on the single-sentence
# outputs that made sentence-level SAR look uniform (verified: 67-73% of pubmed/xsum are 1 sentence).
#
# PREREQ on DoC: the RECORDS for each dataset must be present (01s reads only cache/records/..., not
# per-token). cnn records are already on DoC (generated here); rsync the other 4 from RCS if missing:
#   rsync -avP <user>@login.cx3.hpc.imperial.ac.uk:'/rds/.../cache/records/*__{pubmed_qa,xsum,samsum,med_quad}__ID.jsonl' cache/records/
#
# Faithful (question prepended, matches lm-polygraph):  sbatch slurm/sar_token_doc.sbatch
# No-prepend ablation (answer-only relevance):          NO_PREPEND=1 sbatch slurm/sar_token_doc.sbatch
# After: rsync cache/sar/*__token*.npz back DoC->RCS.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1  # everything (cross-encoder + Llama tokenizer) is cached; avoids node-internet dependency
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
DATASETS="${DS:-pubmed_qa xsum samsum med_quad cnn_dailymail}"
NP=""; [ -n "${NO_PREPEND:-}" ] && NP="--no-prepend"
echo "=== token SAR on $(hostname) at $(date) | datasets: $DATASETS | prepend-mode: ${NO_PREPEND:+NO-PREPEND}${NO_PREPEND:-faithful} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for d in $DATASETS; do
  echo "---- $d ($(date)) ----"
  python -u scripts/01s_sar_relevance.py --dataset "$d" --granularity token $NP \
    || echo "!!! SAR token FAILED for $d"
done
echo "=== done $(date) ==="
