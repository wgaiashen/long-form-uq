#!/bin/bash
#SBATCH --job-name=aux_pinned
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --output=logs/aux_pinned-%j.out

# PINNED-LAMBDA DIAGNOSTIC for the B.1 re-run. Single head, one target, lambda fixed rather than selected.
#
# ⚠️ THIS IS A DIAGNOSTIC, NOT A PROTOCOL RESULT. Lambda was pinned by us, not chosen by the method, so
# these PRR numbers are not comparable with the protocol arm and must never be quoted as if they were.
#
# WHY IT EXISTS. Held-out-source selection has picked lambda=0 on 100% of DiffTask rows and ~96% of ID
# rows. When it does, the auxiliary term never fires and real, shuffled and baseline come out
# byte-identical -- the cell then says nothing about the TARGET, only about the SELECTION RULE. Those are
# two different claims:
#     "the target carries no information"        <- NOT supported by a lambda=0 cell
#     "the selection rule declined the target"   <- what a lambda=0 cell actually shows
# Pinning lambda is the only way to separate them. On nll this already mattered: at lambda=1.0 the
# DiffTask cells produced real > baseline > shuffled, the paper's ordering, which the protocol arm never
# reaches because it selects lambda=0 there.
#
# Kept separate from slurm/aux_attention_rerun.sbatch deliberately: that script has live jobs queued
# against it, and editing a script while jobs wait to read it changes what those jobs run.
#
#   sbatch slurm/aux_pinned_lambda.sbatch pubmed_qa orgad 1.0
#   sbatch slurm/aux_pinned_lambda.sbatch xsum      orgad 1.0

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

EVALS=${1:-pubmed_qa}
TARGETS=${2:-orgad}
LAM=${3:-1.0}

TAG=$(echo "$EVALS" | tr ',' '_')
TGT=$(echo "$TARGETS" | tr ',:' '_-')
# `pinned` in the name so a diagnostic CSV can never be mistaken for a protocol one, and so it cannot
# collide with results/aux_rerun_*.csv (that collision was a real bug, caught before either job ran).
OUT=results/aux_pinned_${TGT}_lam${LAM}_${TAG}.csv

if [ -e "$OUT" ]; then
  echo "REFUSING TO RUN: $OUT already exists -- move it aside first." >&2
  exit 3
fi

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "git: $(git log --oneline -1)"
echo "evals=$EVALS  targets=$TARGETS  PINNED lambda=$LAM  (diagnostic, not protocol)"
echo "out: $OUT"

# A single-value --lambdas makes selection a no-op: the grid has one point, so best_lambda is that point.
# lambda_selection in the CSV still records held_out_source vs same_dataset_slice, which stays true --
# the split was still carved that way, it just had nothing to choose between.
python scripts/checks/aux_attention_ladder.py \
  --evals "$EVALS" \
  --targets "$TARGETS" \
  --lambdas "$LAM" \
  --seeds 1,2,3 \
  --layer 15 \
  --aux-normalise \
  --out "$OUT"
rc=$?

echo; echo "=== exit code: $rc  $(date) ==="
exit $rc
