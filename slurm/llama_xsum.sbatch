#!/bin/bash
#SBATCH --job-name=llama-xsum
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/llama_xsum-%j.out

# Extract Llama-3.1-8B xsum (summarisation) to complete the task set for the cross-task
# matrix. generate.py pooling is already fixed (last-prompt + all answer tokens), so these
# features are correct from the start. fp32 + eager to match the rest of the Llama keystone.
# Long-form, so NOT newline-truncated (our convention); xsum summaries are ~single-line so
# the D1 question is near-moot here anyway. Labelling (AlignScore free + optional mini) and
# probing happen afterwards on CPU/GPU via the overnight orchestrator. RESUMABLE.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "llama_xsum start $(hostname) $(date)"
python scripts/01_extract.py --model meta-llama/Meta-Llama-3.1-8B --dataset xsum --ood ID \
  --dtype fp32 --attn eager
echo "llama_xsum DONE $(date)"
