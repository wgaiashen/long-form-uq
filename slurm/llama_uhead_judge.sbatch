#!/bin/bash
#SBATCH --job-name=uhead-judge
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/llama_uhead_judge-%j.out

# BUG-HUNT item 2 + main-table uniformity: retrain uhead (the luh, v1+v2) on the JUDGE label
# instead of AlignScore, and re-evaluate. Two purposes:
#   (1) item-2 verdict: does the pubmed judge-eval inversion (-0.19/-0.21, AlignScore-trained)
#       persist when the TRAINING label is healthy (judge)? gone = downstream of the label, closed;
#       persists = independent (token-probability-feature inheritance / wiring). pubmed runs FIRST
#       so the verdict lands early in the log.
#   (2) main table: with Path A dropped this table is final; uhead is the one method still stamped
#       "AlignScore-trained". Judge-training all four makes the table uniformly judge-trained.
#
# Writes to DISTINCT score names (uhead_judge / uhead_judge_v2) so the frozen AlignScore-trained
# uhead / uhead_v2 scores are PRESERVED for the item-2 comparison. fp32 + eager (the config;
# eager REQUIRED for the attention features). If xsum OOMs, add --dtype bf16 to its calls.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
echo "llama_uhead_judge start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# pubmed FIRST (settles item 2 early), then the rest to make the table uniformly judge-trained.
for DS in pubmed_qa sciq trivia_qa xsum; do
  echo "================ $DS : uhead_judge v1 (his luh, JUDGE label) ================ $(date)"
  PYTHONPATH=src python scripts/01j_uhead_reference.py --dataset "$DS" --ood ID --model "$MODEL" \
      --variant v1 --name uhead_judge --label-field correctness \
    || echo "!!! uhead_judge v1 $DS FAILED"
  echo "================ $DS : uhead_judge v2 (his luh, JUDGE label) ================ $(date)"
  PYTHONPATH=src python scripts/01j_uhead_reference.py --dataset "$DS" --ood ID --model "$MODEL" \
      --variant v2 --name uhead_judge_v2 --label-field correctness \
    || echo "!!! uhead_judge v2 $DS FAILED"
done
echo "llama_uhead_judge ALL DONE at $(date)"
# item-2 read (per dataset), judge-eval of the judge-trained head vs the frozen AlignScore-trained one:
#   for DS in pubmed_qa sciq trivia_qa xsum; do
#     python - <<PY
#   import numpy as np; from luq.results import prr; import json
#   ...compare uhead_judge vs uhead PRR against correctness...
#   PY
#   done
