#!/bin/bash
#SBATCH --job-name=dump-poolers
#SBATCH --partition=a40,a100,a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dump-poolers-%j.out
# JOB 2 (DoC handover 2026-07-29): dump the MISSING saved poolers so RCS's post-hoc pass (MultiMax /
# Rolling-Means / score-standardised variants read straight off the nn.Linear head) goes from 18 cells to
# the full 8-eval x 3-rung grid. Missing cells:
#   samsum    x {LOO-long, DiffTask-long}   (samsum currently has ID only)
#   factscore x {ID, LOO-long, DiffTask-long}
# dump_ood_attention.py maps --rungs {ID,LOO,DiffTask} -> the LONG rung names {ID, LOO-long, DiffTask-long}
# (it strips "-long" to match base_rung). Widened pool + build_rows == the ladder's own construction. seed 1.
# CPU-only per the script's docstring, but a GPU speeds train_attn and the GPU nodes carry the RAM the ~21GB
# pertok pool needs; device auto-selects cuda. Committed code only.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "dump-poolers start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

set -e
echo "=== samsum: LOO-long + DiffTask-long ==="
python -u scripts/tools/dump_ood_attention.py --evals samsum --rungs LOO,DiffTask --seed 1 --save-pooler
echo "=== factscore: ID + LOO-long + DiffTask-long ==="
python -u scripts/tools/dump_ood_attention.py --evals factscore --rungs ID,LOO,DiffTask --seed 1 --save-pooler

echo "=== written poolers ==="
ls -la cache/probes/*attnpool_*samsum* cache/probes/*factscore*attnpool_* 2>/dev/null || true
ls -la cache/probes/meta-llama_Meta-Llama-3.1-8B__samsum__ID__attnpool_*_s1__L15.pkl \
       cache/probes/meta-llama_Meta-Llama-3.1-8B__factscore__ID__attnpool_*_s1__L15.pkl 2>/dev/null || true
echo "dump-poolers DONE at $(date)"
