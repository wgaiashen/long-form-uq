#!/bin/bash
# RCS/PBSPro twin of scripts/gemma_gpu_watchdog.sh. If the gemma GPU job leaves the
# queue (wall hit, failure, preemption) before all three datasets' features exist,
# resubmit it -- 01_extract resumes from its checkpoint, so this just nudges it forward.
# Bounded resubmits so a persistently-broken job can't loop. Exits once GPU work is done.
#
# Run on an RCS login node (it only polls the queue and resubmits; no GPU):
#   nohup bash pbs/gemma_gpu_watchdog.sh >/dev/null 2>&1 &
set -uo pipefail

# Resolve repo + cluster paths the same way the batch scripts do.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."                         # repo root (pbs/ is one below)
source pbs/_env.sh
cd "$LUQ_REPO"

LOG=logs/gemma_migration.md
OOD=ID
MAX_RESUBS=4
POLL=600   # 10 min
JOBNAME=luq_gemma   # must match #PBS -N in gemma_gpu.pbs (PBS truncates to ~15 chars)

log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') | WATCHDOG(pbs): $*" | tee -a "$LOG"; }

resubs=0
log "started (pid $$)"
while true; do
  if [ -f "cache/.gpu_done__sciq__${OOD}" ] \
  && [ -f "cache/.gpu_done__pubmed_qa__${OOD}" ] \
  && [ -f "cache/.gpu_done__xsum__${OOD}" ]; then
    log "all GPU sentinels present -> GPU work complete; exiting"
    break
  fi
  # If no luq_gemma job is queued or running, the previous one ended with work left.
  # TODO(confirm on RCS): `qstat -u $USER` truncates the job name column; if matching is
  # flaky, switch to `qstat -fw` and grep `Job_Name = luq_gemma`.
  if ! qstat -u "$USER" 2>/dev/null | grep -q "$JOBNAME"; then
    if [ "$resubs" -ge "$MAX_RESUBS" ]; then
      log "GPU job absent, features incomplete, resubmit cap ($MAX_RESUBS) hit -> giving up; needs a look in the morning"
      break
    fi
    resubs=$((resubs + 1))
    jid=$(qsub pbs/gemma_gpu.pbs 2>&1 | grep -oE '^[0-9]+' | head -1)
    log "GPU job was absent with features incomplete -> resubmitted (#${resubs}, job ${jid:-?})"
  fi
  sleep "$POLL"
done
