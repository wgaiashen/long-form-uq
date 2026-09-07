#!/bin/bash
#SBATCH --job-name=pdl-wide-wmsp
#SBATCH --partition=a40,a100,a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/pdl-wide-wmsp-%j.out
# JOB 1 (DoC handover 2026-07-29): base {floors, uniform, SAPLMA, attention, wMSP} WIDENED for the 4 evals
# that are narrow-only and block the canonical table: asqa, med_quad, samsum, factscore.
#   - WIDENED pool is the DEFAULT post-Task-A sampler (no flag): probedriftlong loads all LONG_SRC + evals.
#   - WITHOUT --skip-wmsp, so the wMSP KEEP variants are widened too. But the DoC luq env has NO torchsort,
#     and the 3 blondel variants hard-raise without it. By design: each wMSP variant is a
#     SEPARATE MLP fit (verified) so dropping saves real compute; the canonical widened wMSP set is the three
#     the brief named -- wmsp_norm (reference), wmsp_shrink2 (materially diverges from @10 in the narrow-pool
#     numbers, often stronger -> kept), wmsp_shrink10 (load-bearing for the "floor strong -> shrink wins"
#     finding). EXCLUDED: wmsp_blondel / wmsp_shrink2_blondel / wmsp_shrink10_blondel (alt-loss, torchsort-
#     blocked, known weak) and the two seg variants (not in the brief). --wmsp-only leaves their columns ABSENT
#     (never zero). Their narrow-pool results stay in the supplementary table, population-labelled.
#   - seeds 1,2,3 (same as the rest of the grid). SAPLMA is a base method row (always emitted).
# OVERWRITE GUARDS: distinct --out per eval; print the resolved path; ASSERT it does not exist (abort else);
#   never touch the protected --skip-wmsp CSVs or the RCS 3500195-98 core-set files.
# PRIORITY if the queue is tight: asqa first (it is one of two datasets where the attention pooler LOSES to
#   mean-pool OOD -- a counterexample the argument must account for), then med_quad, samsum, factscore.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "pdl-wide-wmsp start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SLUG=meta-llama_Meta-Llama-3.1-8B
WMSP_KEEP=wmsp_norm,wmsp_shrink2,wmsp_shrink10
# 3 MASTER-GRID rungs only (2026-07-29 projection: full 5-rung run ~13h/no margin vs 16h wall -> risked
# losing a PRIMARY eval; SameTask-long/1ds-Diff-long are NOT in the master grid so this costs the canonical
# table nothing). base names ID,LOO,DiffTask map to ID + LOO-long + DiffTask-long.
RUNGS=ID,LOO,DiffTask

# priority order: asqa first, then med_quad, samsum, factscore
for EVAL in asqa med_quad samsum factscore; do
  OUT="results/probedriftlong_${EVAL}_widened_wmsp__${SLUG}.csv"
  echo "======================================================================"
  echo "[$(date)] EVAL=${EVAL}  ->  resolved --out = ${OUT}"
  if [ -e "${OUT}" ]; then
    echo "ABORT(${EVAL}): ${OUT} already exists -- refusing to clobber. Skipping this eval." >&2
    continue
  fi
  # run it; on failure, leave a note and continue to the next eval (do not kill the whole job)
  if python -u scripts/checks/probedriftlong.py \
        --evals "${EVAL}" --seeds 1,2,3 --layer 15 \
        --rungs "${RUNGS}" --wmsp-only "${WMSP_KEEP}" --out "${OUT}"; then
    echo "[$(date)] ${EVAL} DONE -> ${OUT} ($(wc -l < "${OUT}") lines)"
  else
    echo "[$(date)] ${EVAL} FAILED (see trace above); continuing to next eval" >&2
  fi
done

echo "=== all target CSVs ==="
ls -la results/probedriftlong_{asqa,med_quad,samsum,factscore}_widened_wmsp__${SLUG}.csv 2>/dev/null || true
echo "pdl-wide-wmsp DONE at $(date)"
