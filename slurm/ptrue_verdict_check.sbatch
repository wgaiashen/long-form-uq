#!/bin/bash
#SBATCH --job-name=ptrue-verdict
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ptrue_verdict-%j.out

# Validate the NEW P(True) wording ("Is the above response accurate?") before re-extracting:
# does yes/no still dominate the verdict next-token mass (old wording was ~0.96)?
# Gemma-2 needs bf16 + eager attention; generate.load_model handles that.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=google/gemma-2-9b-it

echo "ptrue verdict check on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in sciq pubmed_qa; do
  echo "================ $DS ================"
  python scripts/checks/check_ptrue_verdict.py --dataset "$DS" --ood ID --model "$MODEL" \
    || echo "!!! verdict check $DS FAILED"
done
echo "VERDICT CHECK DONE at $(date)"
