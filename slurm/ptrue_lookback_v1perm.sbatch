#!/bin/bash
#SBATCH --job-name=ptrue-lookback
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ptrue_lookback-%j.out
#
# P(True) + Lookback Lens features for the datasets whose generations are V1-PERMANENT.
#
#   sbatch slurm/ptrue_lookback_v1perm.sbatch asqa
#   sbatch slurm/ptrue_lookback_v1perm.sbatch expertqa factscore
#
# WHY ONLY THESE THREE. med_quad, samsum and cnn_dailymail are being regenerated, and xsum's cache
# goes stale for the same reason, so features extracted from their CURRENT generations would be
# thrown away. Only asqa, expertqa and factscore stay on v1 generations. The allowlist below turns
# that into a hard stop rather than a thing to remember: 01b/01c have no skip-if-exists, so a
# mistyped dataset would silently OVERWRITE an existing cache with no warning at all.
#
# PARTITION. a40 or a100, never a30/t4: 01c loads the model in fp32 (eager attention in fp16
# overflows to NaN in the softmax), so the weights alone are ~32 GB. The attention tensor on top is
# 32 layers x 32 heads x seq^2 x 4 B -- only ~1.2 GB at these three sets' longest sequence (549
# tokens), against ~17 GB at the 2048-token cap xsum hits. That is the whole reason these three fit
# on a 48 GB card and xsum/cnn needed an a100.
#
# WHY --store-layers 15 IS NOT OPTIONAL. The four existing ptrue caches store layer 15 only (every
# other layer NaN). Without the flag the file has the same shape and dtype but is ~46x larger and
# has a different stored-layer pattern from every cache it will sit beside.
# WHY --name ptrue_accurate. That is the name the ladder drivers read (scripts/checks/ood_onegrid.py)
# and it marks the unified "Is the above response accurate?" wording. There is no `__ptrue.npz`.

set -uo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then echo "usage: sbatch $0 <dataset> [dataset ...]"; exit 2; fi

# Allowlist + the cache namespace each dataset uses (canonical map: scripts/checks/attn_pool.py:60).
regime_for() {
  case "$1" in
    asqa)      echo asqa_rp12 ;;
    expertqa)  echo expertqa_rp12 ;;
    factscore) echo factscore_rp12 ;;
    *) echo "REFUSING '$1': not a v1-permanent dataset. med_quad/samsum/cnn_dailymail/xsum are" \
            "awaiting regeneration and must NOT be extracted yet." >&2; return 1 ;;
  esac
}
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

# Abort BEFORE loading the model if the project volume is short of room. The CephFS byte-quota has
# silently killed a mid-job write before; a failed write leaves no traceback, just a corrupt file.
bash scripts/tools/preflight_space.sh 1 1 || { echo "ABORTING: preflight failed"; exit 1; }

# DoC Slurm spools --output on the compute node and only copies it back on a CLEAN exit, so a
# walltime or OOM kill loses the log entirely. tee to the shared filesystem as well.
LIVE=logs/live_ptrue_lookback_${SLURM_JOB_ID:-manual}.log
{
  echo "start on $(hostname) at $(date)  datasets: ${DATASETS[*]}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

  for DS in "${DATASETS[@]}"; do
    REGIME=$(regime_for "$DS")
    echo "================ $DS : 01b_ptrue (fp16 sdpa, L15 only) ================ $(date)"
    python scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" \
        --name ptrue_accurate --store-layers 15 --prompt-regime "$REGIME" \
      || echo "!!! ptrue_accurate extract $DS FAILED"

    echo "================ $DS : 01c_lookback (fp32 eager) ================ $(date)"
    python scripts/01c_lookback.py --dataset "$DS" --ood ID --model "$MODEL" \
        --prompt-regime "$REGIME" \
      || echo "!!! lookback extract $DS FAILED"
  done

  echo "ALL DONE at $(date)"
  getfattr --absolute-names -n ceph.dir.rbytes /vol/gpudata/gs925-msc_project 2>/dev/null \
    | awk -F\" '/rbytes/{printf "quota used after run: %.2f GB\n",$2/1e9}'
} 2>&1 | tee "$LIVE"
