#!/bin/bash
# One command to submit a batch job on either cluster. Picks the scheduler from
# LUQ_CLUSTER (doc -> Slurm/sbatch, rcs -> PBSPro/qsub), so the same call works on both:
#
#   ./scripts/submit.sh <job> [args...]
#
# <job> is the base name shared by slurm/<job>.sbatch and pbs/<job>.pbs, e.g.
#   ./scripts/submit.sh extract pubmed_qa ID            # parametrised job
#   ./scripts/submit.sh gemma_gpu                       # fixed job
#   ./scripts/submit.sh end_to_end
#
# On DoC, args are passed through as Slurm positional args (matching the existing
# sbatch scripts). On RCS, PBS has no positional script args, so for the parametrised
# `extract` job we translate the positional args into the LUQ_* env vars the .pbs reads.
set -euo pipefail

cd "$(cd "$(dirname "$0")" && pwd)/.."   # repo root

# Resolve the cluster (honours LUQ_CLUSTER, else hostname). Reuse the shell resolver so
# detection matches the batch scripts exactly.
source pbs/_env.sh   # sets LUQ_CLUSTER (and LUQ_REPO/HF_HOME, unused here)

JOB="${1:?usage: submit.sh <job> [args...]}"; shift || true

case "$LUQ_CLUSTER" in
  doc)
    script="slurm/$JOB.sbatch"
    [ -f "$script" ] || { echo "no $script on this (doc) cluster" >&2; exit 1; }
    set -x; exec sbatch "$script" "$@"
    ;;
  rcs)
    script="pbs/$JOB.pbs"
    [ -f "$script" ] || { echo "no $script on this (rcs) cluster" >&2; exit 1; }
    case "$JOB" in
      extract)
        # positional ($1 dataset, $2 ood, $3 model) -> env vars the .pbs reads
        vars="LUQ_DATASET=${1:-sciq},LUQ_OOD=${2:-ID},LUQ_MODEL=${3:-google/gemma-2-9b-it}"
        set -x; exec qsub -v "$vars" "$script"
        ;;
      *)
        if [ "$#" -gt 0 ]; then
          echo "note: $JOB.pbs takes no positional args; ignoring: $*" >&2
        fi
        set -x; exec qsub "$script"
        ;;
    esac
    ;;
  *)
    echo "unknown LUQ_CLUSTER='$LUQ_CLUSTER' (expected doc|rcs)" >&2
    exit 1
    ;;
esac
