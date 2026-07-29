#!/bin/bash
#SBATCH --job-name=dump-s23
#SBATCH --partition=a40,a100,a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dump-s23-%j.out
# FOLLOW-UP (DoC handover 2026-07-29): 3A ran SINGLE-SEED. Three cells are the ones we would QUOTE and need
# seed support (RCS re-reads the post-hoc aggregations off the saved poolers, reports 3-seed mean+variance
# for THESE CELLS ONLY -- no full 3D retrain):
#   cnn_dailymail DiffTask-long  (armA -0.107 -> multimax +0.039 ; rescue from negative)
#   xsum          DiffTask-long  (armA +0.133 -> multimax +0.138)
#   pubmed_qa     LOO-long       (armA +0.168 -> V2 +0.262 ; the pre-registration miss)
# Seed 1 poolers for these already exist; this adds SEEDS 2 and 3 only. Same widened pool + same rung
# construction as the ladder (dump_ood_attention reuses cells_long/build_rows). Submitted with a Slurm
# dependency so it runs AFTER JOB 1 (267806) and JOB 2 (267790), never before -- the critical-path base run
# gets the GPU first.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "dump-s23 start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

set -e
for S in 2 3; do
  echo "=== seed ${S}: cnn_dailymail + xsum DiffTask-long ==="
  python -u scripts/tools/dump_ood_attention.py --evals cnn_dailymail,xsum --rungs DiffTask --seed "${S}" --save-pooler
  echo "=== seed ${S}: pubmed_qa LOO-long ==="
  python -u scripts/tools/dump_ood_attention.py --evals pubmed_qa --rungs LOO --seed "${S}" --save-pooler
done

echo "=== written seed-2/3 quote-cell poolers ==="
ls -la cache/probes/meta-llama_Meta-Llama-3.1-8B__cnn_dailymail__ID__attnpool_DiffTask-long_s{2,3}__L15.pkl \
       cache/probes/meta-llama_Meta-Llama-3.1-8B__xsum__ID__attnpool_DiffTask-long_s{2,3}__L15.pkl \
       cache/probes/meta-llama_Meta-Llama-3.1-8B__pubmed_qa__ID__attnpool_LOO-long_s{2,3}__L15.pkl 2>/dev/null || true
echo "dump-s23 DONE at $(date)"
