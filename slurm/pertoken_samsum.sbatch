#!/bin/bash
#SBATCH --job-name=pertok-samsum
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pertoken-samsum-%j.out

# Per-token L15 hidden states for samsum (fp32 + eager, matching every other dataset's pertok cache).
# samsum is train-only (no test split), so the trailing SANITY line prints NaN / MISMATCH -- that is
# EXPECTED and harmless (the sanity reproduces PRR on the test split, and there is none). The cache is
# valid as soon as "cached per-token (L15) -> ...__samsum__ID__L15.npz" has printed; only a real error
# (model load / OOM / dataset) is a failure. No `set -e` so the trailing sanity does not fail the job.
#
#   sbatch slurm/pertoken_samsum.sbatch

set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B

echo "=== node ==="; hostname; date
python scripts/01h_pertoken.py --model "$MODEL" --dataset samsum --ood ID --layer 15 --dtype fp32 --attn eager
echo "=== done ==="; date
