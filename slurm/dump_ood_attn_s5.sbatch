#!/bin/bash
#SBATCH --job-name=dump_ood_attn
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dump_ood_attn-%j.out

# S5: OOD attention sidecar dumps (E1 / B2 diagnostic). Trains the attention pooler on the LONG
# ladder's OOD source pool (LOO, DiffTask) for asqa + med_quad and saves per-token pool_w sidecars
# (cache/viz/...__attn__<rung>-long.npz) + the trained poolers (cache/probes/...attnpool_...pkl).
# High --mem: it loads the full 8-source LONG_SRC per-token pool. a100 for speed + headroom.
#
#   sbatch slurm/dump_ood_attn_s5.sbatch
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== node ==="; hostname; date

python -u scripts/tools/dump_ood_attention.py \
  --evals asqa,med_quad --rungs LOO,DiffTask --seed 1 --save-pooler

echo "=== ALL DONE $(date) ==="
