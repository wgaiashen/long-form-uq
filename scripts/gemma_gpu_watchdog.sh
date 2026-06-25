#!/bin/bash
# Watchdog for the Gemma GPU job. If the job leaves the queue (wall hit, failure,
# preemption) before all three datasets' features exist, resubmit it -- 01_extract
# resumes from its checkpoint, so this just nudges it forward. Bounded resubmits so a
# persistently-broken job can't loop. Exits cleanly once all GPU features are done.
set -uo pipefail
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
LOG=logs/gemma_migration.md
OOD=ID
MAX_RESUBS=4
POLL=600   # 10 min

log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') | WATCHDOG: $*" | tee -a "$LOG"; }

resubs=0
log "started (pid $$)"
while true; do
  if [ -f "cache/.gpu_done__sciq__${OOD}" ] \
  && [ -f "cache/.gpu_done__pubmed_qa__${OOD}" ] \
  && [ -f "cache/.gpu_done__xsum__${OOD}" ]; then
    log "all GPU sentinels present -> GPU work complete; exiting"
    break
  fi
  # If no gemma-gpu job is queued or running, the previous one ended with work left.
  if ! squeue -u "$USER" -n gemma-gpu -h 2>/dev/null | grep -q .; then
    if [ "$resubs" -ge "$MAX_RESUBS" ]; then
      log "GPU job absent, features incomplete, resubmit cap ($MAX_RESUBS) hit -> giving up; needs a look in the morning"
      break
    fi
    resubs=$((resubs + 1))
    jid=$(sbatch slurm/gemma_gpu.sbatch 2>&1 | grep -oE '[0-9]+' | tail -1)
    log "GPU job was absent with features incomplete -> resubmitted (#${resubs}, job ${jid:-?})"
  fi
  sleep "$POLL"
done
