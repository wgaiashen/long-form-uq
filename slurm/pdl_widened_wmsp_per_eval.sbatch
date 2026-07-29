#!/bin/bash
#SBATCH --partition=a40,a100,a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# JOB 1 per-eval (DoC handover 2026-07-29, revised): base {floors, uniform, SAPLMA, attention, wMSP} for ONE
# eval across ALL 5 RUNGS, with --save-pooler. Submit once per eval (asqa, med_quad, samsum, factscore) via
#   sbatch -J pdl-<eval> --export=ALL,EVAL=<eval> slurm/pdl_widened_wmsp_per_eval.sbatch
# WHY per-eval + 5 rungs:
#   - probedriftlong writes its CSV at EVAL-END, so one job per eval means a walltime kill costs at most one
#     eval, and evals run concurrently where slots allow.
#   - 5 rungs: the headline comparison (attention vs SAPLMA vs msp_min vs wMSP) lives in the base method set,
#     so 1ds-Diff-long (single source + task shift) is PRIMARY evidence, not supplementary. No --rungs filter.
#   - --save-pooler: the attention pooler is trained in every cell anyway, so persisting it (seed-1, pkl +
#     sidecar, dump_ood_attention format) gives 5 rungs of saved poolers at ~zero extra compute, which covers
#     the samsum/factscore SameTask/1ds-Diff pooler gap (no separate dump needed).
# wMSP: --wmsp-only keeps norm/shrink2/shrink10 (torchsort absent on DoC -> blondel would raise; seg not in
#   the brief). Provenance: the driver stamps git_sha/cluster/env_hash per row and ABORTS if the tracked tree
#   is dirty. 3-guard overwrite protection below. seeds 1,2,3.
set -uo pipefail
: "${EVAL:?set EVAL=<dataset> via --export}"
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "pdl-${EVAL} start on $(hostname) at $(date)  [git $(git rev-parse --short HEAD)]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SLUG=meta-llama_Meta-Llama-3.1-8B
OUT="results/probedriftlong_${EVAL}_widened_wmsp__${SLUG}.csv"
echo "resolved --out = ${OUT}"
if [ -e "${OUT}" ]; then
  echo "ABORT: ${OUT} already exists -- refusing to clobber." >&2
  exit 3
fi

python -u scripts/checks/probedriftlong.py \
    --evals "${EVAL}" --seeds 1,2,3 --layer 15 \
    --wmsp-only wmsp_norm,wmsp_shrink2,wmsp_shrink10 \
    --save-pooler --out "${OUT}"

echo "=== ${EVAL} DONE -> ${OUT} ($(wc -l < "${OUT}") lines) at $(date) ==="
ls -la "${OUT}"
