#!/bin/bash
#SBATCH --job-name=ptrue-acc-xsum
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ptrue_acc_xsum-%j.out
# xsum P(True) features under the NEW unified wording ("Is the above response accurate?").
# NO gate (hidden-state probe). a40/64G headroom for xsum's long articles; the old-wording xsum
# ptrue extracted fine uncapped, so same lengths are safe here.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "start $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/01b_ptrue.py --dataset xsum --ood ID --model google/gemma-2-9b-it --name ptrue_accurate \
  || echo "!!! xsum ptrue_accurate extract FAILED"
echo "XSUM PTRUE_ACCURATE DONE $(date)"
