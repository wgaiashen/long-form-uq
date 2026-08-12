#!/bin/bash
# Overnight watchdog for the Qwen clean-span ladder (2026-08-11/12).
# Not a committed pipeline script -- a throwaway ops script for one night's monitoring.
# Detects: (a) real completion/failure per eval, for reporting; (b) an eval whose task has
# disappeared from squeue (timed out / node died / OOM-killed by the scheduler) without ever
# printing a "done at" or "!!! FAILED for" line -- SLURM kills those silently from the script's
# point of view, so this is the only way to notice. Requeues just that eval, same array index
# (so log naming stays consistent across retries), capped at 2 retries so a REAL bug reports
# instead of looping forever.
set -uo pipefail
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

EVALS=(pubmed_qa med_quad asqa xsum cnn_dailymail samsum expertqa factscore)
SEEN=/tmp/.qwen_clean_seen_overnight
RETRY_CAP=2
: > "$SEEN" 2>/dev/null || true   # fresh state for this watch session

active_indices() {
  # array-task indices currently running or pending (excludes the plain interactive job,
  # which has no "_N" suffix in squeue's JobID column)
  squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -oE '_[0-9]+$' | tr -d '_'
}

is_done() { grep -ql "=== clean ladder ${1} done at" logs/qwen_pdl_clean_*_"${2}".out 2>/dev/null; }
is_failed() { grep -ql "!!! clean ladder FAILED for ${1}" logs/qwen_pdl_clean_*_"${2}".out 2>/dev/null; }

while true; do
  active="$(active_indices)"
  n_done=0
  for idx in "${!EVALS[@]}"; do
    ev="${EVALS[$idx]}"
    if is_done "$ev" "$idx"; then
      n_done=$((n_done+1))
      tag="DONE:$ev"
      grep -qF "$tag" "$SEEN" 2>/dev/null || { echo "EVENT: $ev finished"; echo "$tag" >> "$SEEN"; }
      continue
    fi
    if is_failed "$ev" "$idx"; then
      tag="SCRIPTFAIL:$ev"
      grep -qF "$tag" "$SEEN" 2>/dev/null || { echo "EVENT: $ev FAILED at the script level (not a timeout -- a real error, check the log)"; echo "$tag" >> "$SEEN"; }
      continue
    fi
    if echo "$active" | grep -qx "$idx"; then
      continue   # still running or queued, nothing to do
    fi
    # not done, not script-failed, not in squeue at all -> died silently (timeout / node / OOM)
    cf="/tmp/.qwen_retry_count_${idx}"
    cnt=$(cat "$cf" 2>/dev/null || echo 0)
    if [ "$cnt" -ge "$RETRY_CAP" ]; then
      tag="GAVEUP:$ev"
      grep -qF "$tag" "$SEEN" 2>/dev/null || { echo "EVENT: $ev has died $RETRY_CAP times with no script-level error -- STOPPING auto-retry, needs a human look"; echo "$tag" >> "$SEEN"; }
      continue
    fi
    cnt=$((cnt+1)); echo "$cnt" > "$cf"
    echo "EVENT: $ev (index $idx) is not running, not pending, and never finished -- looks like a timeout/kill. Requeuing (attempt $cnt/$RETRY_CAP)."
    sed "s/#SBATCH --array=0-7%3/#SBATCH --array=${idx}/" slurm/qwen_pdl_ladder_clean.sbatch > /tmp/qwen_pdl_clean_retry_${idx}.sbatch
    out=$(sbatch /tmp/qwen_pdl_clean_retry_${idx}.sbatch 2>&1)
    echo "REQUEUE_RESULT[$ev]: $out"
  done
  if [ "$n_done" -ge 8 ]; then
    echo "ALL_8_EVALS_DONE"
    break
  fi
  ngaveup=$(grep -c "^GAVEUP:" "$SEEN" 2>/dev/null || echo 0)
  if [ "$ngaveup" -gt 0 ] && [ "$((n_done + ngaveup))" -ge 8 ]; then
    echo "STOPPING_WATCH: ${ngaveup} eval(s) gave up after ${RETRY_CAP} retries, rest are done -- needs a human look"
    break
  fi
  sleep 150
done
