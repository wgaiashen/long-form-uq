#!/bin/bash
#SBATCH --job-name=ptrue-acc-extract
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ptrue_acc_extract-%j.out
# Extract P(True) features under the NEW unified wording ("Is the above response accurate?").
# NO verdict-mass gate: P(True) is a HIDDEN-STATE probe (ptrue.py reads states[-1], not the
# emitted token), so the verdict-mass sanity check is irrelevant to whether the probe works.
# Saves under name 'ptrue_accurate'; the old 'ptrue' (true-wording) features are left untouched.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=google/gemma-2-9b-it
for DS in sciq pubmed_qa; do
  echo "================ $DS : 01b_ptrue --name ptrue_accurate ================ $(date)"
  python scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" --name ptrue_accurate \
    || echo "!!! ptrue_accurate extract $DS FAILED"
done
echo "PTRUE_ACCURATE EXTRACT DONE at $(date)"
