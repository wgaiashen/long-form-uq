#!/bin/bash
#SBATCH --job-name=uhead-verify
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/uhead_verify-%j.out

# Verify OUR uhead glue against the authors' own inference path on a few sciq records
# (drives luh.CalculatorInferLuh.infer_cached + LuhClaimEstimator and compares to our
# uhead_instance_score). Light: ~10 short records, so a40 is plenty.
#
#   sbatch slurm/uhead_verify.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "uhead-verify start on $(hostname) at $(date)"
python scripts/checks/uhead_vs_authors.py --dataset sciq --ood ID \
    --model google/gemma-2-9b-it --n 10
echo "uhead-verify done at $(date)"
