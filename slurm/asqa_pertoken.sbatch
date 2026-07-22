#!/bin/bash
#SBATCH --job-name=pertok_asqa
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/pertok_asqa-%j.out

# ASQA per-token L15 on DoC's A100-80GB. ASQA has NO long source (closed-book question-only prompt,
# gen <=256), so it fits ANY GPU -- the A100 just matches the fp32+eager comparability of the other
# Llama sets. Teacher-forced fp32+eager over cache/asqa_rp12 (records must be judge-labelled first),
# writes cache/asqa_rp12/pertok/...__L15.npz. As a self-check 01h prints the PRR of mean-pooling the
# per-token states it caches; that must match the cached SAPLMA feature's PRR at L15.
#
# PREREQ: cache/asqa_rp12/ present with records + the SAPLMA feature npz (both from asqa_full.sbatch).
#
#   sbatch slurm/asqa_pertoken.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo; echo "########## ASQA per-token L15 (fp32 + eager) ##########"
python -u scripts/01h_pertoken.py --model meta-llama/Meta-Llama-3.1-8B \
    --dataset asqa --ood ID --prompt-regime asqa_rp12 \
    --layer 15 --dtype fp32 --attn eager

echo; echo "=== DONE $(date) ==="
