#!/bin/bash
#SBATCH --partition=a40,a100,a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# The missing 2x2 cell (aggregation x head): all four cells recomputed together per eval, ALL 5 rungs,
# seeds 1,2,3, widened pool. Submit once per eval:  EVAL=<eval> sbatch -J h2x2-<eval> slurm/head_agg_2x2_per_eval.sbatch
# (set EVAL in the caller env, NO --export=ALL,VAR -- that triggers get-user-env and holds the job on DoC.)
# The three reference cells reproduce the base run's uniform/saplma/attention by construction; the gate prints
# per-rung margins vs the widened base CSV where one exists on this cluster. --save-pooler writes the NEW
# attention_mlp bundle under a DISTINCT namespace (headagg2x2) so it never collides with base/arm poolers.
set -uo pipefail
: "${EVAL:?set EVAL=<dataset> in the caller env}"
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "h2x2-${EVAL} start on $(hostname) at $(date)  [git $(git rev-parse --short HEAD)]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SLUG=meta-llama_Meta-Llama-3.1-8B
OUT="results/head_agg_2x2_${EVAL}__${SLUG}.csv"
echo "resolved --out = ${OUT}"
if [ -e "${OUT}" ]; then
  echo "ABORT: ${OUT} already exists -- refusing to clobber." >&2
  exit 3
fi

python -u scripts/checks/head_agg_2x2.py --evals "${EVAL}" --seeds 1,2,3 --layer 15 --save-pooler --out "${OUT}"

echo "=== h2x2-${EVAL} DONE -> ${OUT} ($(wc -l < "${OUT}") lines) at $(date) ==="
ls -la "${OUT}"
