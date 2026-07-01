#!/bin/bash
# Loud free-space preflight for the project cephfs volume (the 50GB byte-quota that silently
# killed the first baseline runs when it filled mid-job). Call at the START of any sbatch that
# writes sizeable features, so a shortfall aborts BEFORE the GPU work instead of crashing
# silently mid-write (a failed stdout write leaves no traceback).
#
#   scripts/tools/preflight_space.sh <footprint_gib> [margin_gib]
#
# Exits 1 (loud) if available < footprint + margin. Uses cephfs xattrs for an instant exact
# read (quota + recursive bytes), falling back to du and a hardcoded 50GiB quota.
set -uo pipefail
NEED_GIB="${1:-3}"
MARGIN_GIB="${2:-2}"
ROOT=/vol/gpudata/gs925-msc_project

maxb=$(getfattr --only-values -n ceph.quota.max_bytes "$ROOT" 2>/dev/null || true)
used=$(getfattr --only-values -n ceph.dir.rbytes   "$ROOT" 2>/dev/null || true)
[ -z "${maxb:-}" ] || [ "$maxb" = "0" ] && maxb=$((50 * 1024*1024*1024))
[ -z "${used:-}" ] && used=$(du -sb "$ROOT" 2>/dev/null | awk '{print $1}')

avail=$(( maxb - used ))
need=$(( (NEED_GIB + MARGIN_GIB) * 1024*1024*1024 ))
gib() { awk "BEGIN{printf \"%.1f\", $1/1073741824}"; }
echo "PREFLIGHT: quota $(gib "$maxb")GiB  used $(gib "$used")GiB  avail $(gib "$avail")GiB  need $(gib "$need")GiB (footprint ${NEED_GIB} + margin ${MARGIN_GIB})"
if [ "$avail" -lt "$need" ]; then
  echo "PREFLIGHT ABORT: not enough room on the project volume. Free space (cache/pdnew, cache/pertok, old-model features) or raise the quota, then resubmit. NOT starting GPU work."
  exit 1
fi
echo "PREFLIGHT OK"
