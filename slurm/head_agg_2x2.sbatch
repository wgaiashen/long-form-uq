#!/bin/bash
#SBATCH --job-name=h2x2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/head_agg_2x2-%j.out

# The missing 2x2 cell: learned attention x MLP head (scripts/checks/head_aggregation_2x2.py).
# ONE eval per job (distinct --out) so a walltime kill loses only that eval. PER-EVAL because the
# driver writes the CSV once at the end.
#
#   sbatch slurm/head_agg_2x2.sbatch pubmed_qa            # THE DECISION GATE -- run this ALONE first
#   sbatch slurm/head_agg_2x2.sbatch xsum                 # ...the rest ONLY after go-ahead
#   sbatch slurm/head_agg_2x2.sbatch pubmed_qa --early-stop   # the fallback if fixed 60ep overfits
#
# GPU (per-cell temperature sweep + pooler/MLP training over seeds). Reads pertok (bitbucket symlinks)
# + records; reuses probedriftlong's EXACT pool construction so the linear/saplma cells reproduce the
# reference runs. Does NOT recompute floors/wMSP, and does NOT touch anything RCS is writing.
#
# PROVENANCE: the driver ABORTS if the tracked tree is dirty -- commit the attn_pool.py edit + the new
# files first, else this job exits immediately.
set -uo pipefail
EVAL="${1:?usage: sbatch slurm/head_agg_2x2.sbatch <eval> [--early-stop]}"
EXTRA="${2:-}"
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

SLUG=meta-llama_Meta-Llama-3.1-8B
OUT=results/head_aggregation_2x2_${EVAL}__${SLUG}.csv

echo "=== head_agg_2x2 eval=$EVAL extra='$EXTRA' ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --save-pooler persists the seed-1 attention_linear + attention_mlp poolers (method-tagged, no clobber
# of armA sidecars). The 5ep/wd0 diagnostic ladder is on by default (required, not optional).
python -u scripts/checks/head_aggregation_2x2.py \
    --evals "$EVAL" --seeds 1,2,3 --layer 15 \
    --out "$OUT" --save-pooler $EXTRA

echo "=== $EVAL DONE $(date); verifying gates ==="
python -u scripts/checks/verify_2x2_gates.py --result "$OUT" || echo "GATE CHECK reported failures (see above)"
echo "=== ALL DONE $(date) ==="
