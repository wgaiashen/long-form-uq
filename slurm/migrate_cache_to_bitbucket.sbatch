#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=luq_cachemv
#SBATCH --output=logs/migrate_cache_%j.out

# Move the cache off the CephFS allocation onto bitbucket, per LUQ_CACHE_ROOT (99884ad).
# /vol/gpudata has a HARD ~51GB quota and was 91.5% full; /vol/bitbucket has no quota and 7.5TB free.
#
# COPY, VERIFY, then (separately, by hand, after review) delete the source. rsync -avP, never cp/scp:
# a plain copy silently truncated a 2.3GB file on this project once. The source is left in place by
# this job on purpose -- deleting it is the irreversible half and is not automated.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SRC=/vol/gpudata/gs925-msc_project/msc-project-gs925/cache/
DST=/vol/bitbucket/gs925/luq_cache/
echo "=== $(date) rsync $SRC -> $DST ==="
rsync -a --info=progress2 --no-inc-recursive "$SRC" "$DST"
echo "rsync exit: $?"
echo "=== verify: file counts and bytes must match ==="
echo "src files: $(find "$SRC" -type f | wc -l)   dst files: $(find "$DST" -type f | wc -l)"
echo "src bytes: $(du -sb "$SRC" | cut -f1)   dst bytes: $(du -sb "$DST" | cut -f1)"
echo "=== $(date) done ==="
