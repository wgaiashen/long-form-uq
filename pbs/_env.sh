#!/bin/bash
# Cluster-resolution layer for shell (sourced by every pbs/*.pbs script, and usable
# from a login shell). It picks ONE set of cluster-specific paths so the same batch
# script body runs on either cluster. Nothing below is RCS-only by construction:
# LUQ_CLUSTER=doc reproduces the existing DoC/Slurm values exactly, so a script that
# sources this file behaves identically to the hand-written slurm/*.sbatch on DoC.
#
# This is the SHELL twin of src/luq/cluster.py (which does the same resolution for
# Python). Keep the two in step if you change a default here.
#
# Override anything by exporting it before `qsub`/`sbatch` (or in your shell rc):
#   LUQ_CLUSTER   doc | rcs           (else auto-detected from the hostname)
#   LUQ_REPO      path to the repo checkout
#   HF_HOME       HuggingFace cache (MUST be off $HOME -- weights are multi-GB)
#   LUQ_VENV      (rcs) path to the python venv to activate
#   LUQ_CONDA_SH / LUQ_CONDA_ENV   conda profile + env name (used on doc; optional on rcs)

# --- detect the cluster ------------------------------------------------------
: "${LUQ_CLUSTER:=}"
if [ -z "$LUQ_CLUSTER" ]; then
  _host="$(hostname -f 2>/dev/null || hostname)"
  case "$_host" in
    *gpucluster*|*.doc.ic.ac.uk) LUQ_CLUSTER=doc ;;
    *cx3*|*hpc*|*.rcs.*|*.hpc.ic.ac.uk) LUQ_CLUSTER=rcs ;;
    # A pbs script only ever runs on RCS, so default unknown hosts there. The DoC
    # path is selected by hostname above or by exporting LUQ_CLUSTER=doc.
    *) LUQ_CLUSTER=rcs ;;
  esac
  unset _host
fi

# --- resolve paths -----------------------------------------------------------
if [ "$LUQ_CLUSTER" = doc ]; then
  # The existing DoC/CSG layout (everything on the /vol/gpudata CephFS allocation).
  : "${LUQ_REPO:=/vol/gpudata/gs925-msc_project/msc-project-gs925}"
  : "${HF_HOME:=/vol/gpudata/gs925-msc_project/hf_cache}"
  : "${LUQ_CONDA_SH:=/vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh}"
  : "${LUQ_CONDA_ENV:=luq}"
  # ⚠️ BIG CACHES MUST NOT LAND ON THE CEPH ALLOCATION. /vol/gpudata is CephFS with a HARD 50GB
  # quota (~4GB free as of 2026-08-08); /vol/bitbucket/gs925 is NFS, no quota, 7.5TB free. The Qwen
  # caches are ~90GB (16GB pooled features + ~74GB per-token states), so they go to bitbucket.
  # Verified 2026-08-08: writable, no scheduled purge, NOT backed up -- which is fine, because
  # everything under cache/ is regenerable by design. Records and results are NOT stored here.
  # Read speed, CORRECTED 2026-08-08: ~137-232 MB/s sustained ON A COMPUTE NODE (measured migrating
  # 16.4 GB / 296 files in ~2 min, byte-identical, npz all load). An earlier ~20-24 MB/s figure was
  # the JUMP BOX's link, and a second one conflated shard read with bf16->fp32 conversion and the
  # host-to-device copy. So a 74 GB cold read is ~6-9 min, not ~50 -- which makes DoC-hosted
  # extraction MORE attractive, not less. Still prefer co-locating a job with the cache it consumes.
  : "${LUQ_CACHE_ROOT:=/vol/bitbucket/gs925/luq_cache}"
  export LUQ_CACHE_ROOT
else
  # ---- RCS (CX3 Phase 2) ----
  # On RCS the home directory has a large allocation (~930GB), so the repo, the HF
  # cache, and the venv all just live in $HOME -- no RDS indirection needed. PBS starts
  # the job in $HOME and exports the submit dir as $PBS_O_WORKDIR, so if you `qsub` from
  # the repo root LUQ_REPO resolves itself.
  # The one place NOT to use is $EPHEMERAL: it is wiped after 30 days, so model weights
  # cached there would vanish monthly and re-download constantly.
  : "${LUQ_REPO:=${PBS_O_WORKDIR:-$PWD}}"
  : "${HF_HOME:=$HOME/hf_cache}"   # large home allocation; do NOT use $EPHEMERAL
  : "${LUQ_VENV:=$HOME/venv}"      # python venv created per pbs/SETUP_RCS.md
  # FActScore assets. src/luq/factscore.py defaults to the DoC master
  # (/vol/gpudata/gs925-msc_project/factscore_data), which does not exist on RCS -- so a factscore
  # extraction here died with FileNotFoundError on prompt_entities.txt (2026-08-15).
  # ⚠️ ENTITIES ONLY. This directory holds the 500-line prompt_entities.txt and NOT the ~4GB
  # enwiki-20230401.db, so GENERATION works on RCS but factscore LABELLING does not -- the judge
  # reads the enwiki sqlite. Label factscore on DoC, or rsync the db here first.
  : "${FACTSCORE_DIR:=$(dirname "${LUQ_REPO}")/factscore_data}"
  export FACTSCORE_DIR
fi
export LUQ_CLUSTER LUQ_REPO HF_HOME

# --- activate the python environment ----------------------------------------
# Call this from each batch script AFTER sourcing this file. PBS does not source
# ~/.bashrc, so the environment must be loaded explicitly inside the job.
luq_activate() {
  if [ "$LUQ_CLUSTER" = doc ]; then
    # shellcheck disable=SC1090
    source "$LUQ_CONDA_SH"
    conda activate "$LUQ_CONDA_ENV"
  else
    # ---- RCS ----
    # `tools/prod` exposes the EasyBuild software stack on CX3; Python/3.11.5 matches the
    # project's 3.11 env (confirmed available via `module avail Python` on login-ai).
    module load tools/prod 2>/dev/null || true
    module load Python/3.11.5-GCCcore-13.2.0
    if [ -n "${LUQ_CONDA_ENV_RCS:-}" ]; then
      # Optional: use a conda env on RCS instead of a venv (see the RCS conda guide).
      # shellcheck disable=SC1090
      source "${LUQ_CONDA_SH_RCS:?set LUQ_CONDA_SH_RCS to your conda profile.d/conda.sh}"
      conda activate "$LUQ_CONDA_ENV_RCS"
    else
      if [ ! -f "$LUQ_VENV/bin/activate" ]; then
        echo "luq_activate: no venv at $LUQ_VENV -- run pbs/SETUP_RCS.md first, or export LUQ_VENV" >&2
        return 1
      fi
      # shellcheck disable=SC1090
      source "$LUQ_VENV/bin/activate"
    fi
  fi
  # The luq package is not pip-installed (the pipeline scripts self-insert src/ onto
  # sys.path); put it on PYTHONPATH so `python -m luq.cluster` works in this shell too.
  export PYTHONPATH="$LUQ_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
}
