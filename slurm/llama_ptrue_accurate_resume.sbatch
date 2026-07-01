#!/bin/bash
#SBATCH --job-name=llama-ptrue-r
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/llama_ptrue_acc_resume-%j.out

# RESUME of llama_ptrue_accurate.sbatch: sciq + trivia_qa ptrue_accurate features were saved
# before the first run died on the project quota; only pubmed_qa + xsum remain.
# The probe reads a single layer (--layer 15), so we STORE ONLY L15 (--store-layers 15), fp32:
# ~60 MB/dataset instead of ~1.2 GB (all 33 layers), which dissolves the quota problem structurally
# rather than squeezing under it. dtype is unchanged, so no precision change to the probe input.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B

# Loud free-space preflight (step 5): L15-only pubmed+xsum features are ~60MB each (~0.12GB total);
# require 1GB footprint + 1GB margin, else abort before loading the model.
bash scripts/tools/preflight_space.sh 1 1 || { echo "ABORTING: preflight failed"; exit 1; }

echo "llama_ptrue_accurate RESUME start on $(hostname) at $(date)"
for DS in pubmed_qa xsum; do
  echo "================ $DS : 01b_ptrue --name ptrue_accurate --store-layers 15 ================ $(date)"
  python scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" --name ptrue_accurate \
      --store-layers 15 \
    || echo "!!! ptrue_accurate extract $DS FAILED"
done
echo "llama_ptrue_accurate RESUME ALL DONE at $(date)"
