#!/bin/bash
# Shared body for the multi-layer per-token extraction jobs. Sourced by the wrappers in this
# directory, which differ only in the cards they request and the datasets they cover.
#
# Registered in prereg/M9_layer_distance_sensitivity.md, which fixes the layer set and the analysis
# before any result exists.
#
# THE FROZEN LAYER SET is range(0, 32, 3) = 0 3 6 9 12 15 18 21 24 27 30. Layer 15 is ALREADY CACHED
# and is deliberately absent from the chunks below: the extractor refuses to overwrite an existing
# layer, and the canonical middle-layer caches are the input to every published result here.
#
# CHUNKED FOR HOST MEMORY. One teacher-forced forward computes every layer, so extracting several at
# once costs one forward rather than several. The extractor holds each layer's states in memory until
# the dataset is done, and ten layers of the largest dataset would be roughly 70 GB resident. Three
# chunks keeps the peak near 32 GB. The boundaries are a memory decision and change no value.
#
# RESUMABLE. Before each chunk this asks the cache which of those layers already exist and passes
# only the missing ones, which is what the extractor's overwrite guard tells the caller to do. A
# chunk whose layers are all present is skipped, so a job that runs out of walltime, and a job that
# covers only part of the dataset list, both compose with a later run instead of aborting on the
# first completed layer.
#
# THE REGIME OVERRIDE IS NOT OPTIONAL. Without it the corrected-span dataset resolves to the
# uncorrected namespace and the sensitivity would silently sit on a different population from the
# grid it is compared against.
#
# Parameters, set by the wrapper before sourcing this file:
#   LUQ_PT_DATASETS  space-separated dataset list
#   LUQ_PT_DEVMAP    "cuda" for one card, "auto" to shard
#   LUQ_PT_MAXMEM    per-device ceilings when sharding, e.g. "0=17GiB,1=17GiB"; empty for one card
#   LUQ_PT_SKIPHEAD  1 to compute the unread vocabulary projection at one position instead of all;
#                    unset or 0 keeps the original call path

set -uo pipefail
cd "$PBS_O_WORKDIR"; source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

mkdir -p logs
LIVE="logs/pertoken_multilayer__${PBS_JOBID%%.*}.log"
exec > >(tee -a "$LIVE") 2>&1
echo "live log: $LIVE"

export HF_HOME
export HF_HUB_OFFLINE=1
# Fragmentation, not capacity, ended one attempt at the longest dataset: the allocator held 3.73 GiB
# reserved but unallocated while a 3.67 GiB request failed against 3.52 GiB free. Expandable segments
# let those reserved blocks be reused instead of stranded.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="meta-llama/Meta-Llama-3.1-8B"
export LUQ_REGIME="med_quad=cleanv2"

CHUNKS=("0,3,6,9" "12,18,21,24" "27,30")

echo "=== multi-layer per-token extraction, $MODEL === host=$(hostname) at $(date)"
echo "commit: $(git rev-parse HEAD 2>/dev/null)"
echo "datasets: $LUQ_PT_DATASETS"
echo "device map: $LUQ_PT_DEVMAP  ceilings: ${LUQ_PT_MAXMEM:-(none)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
df -h /rds/general/user/gs925/home | tail -1

# Ask the cache which layers are already on disk, using the same Config and key the extractor uses so
# the answer cannot drift from the paths it will write. Prints the still-missing layers, or nothing.
missing_layers () {
  LUQ_DS="$1" LUQ_RG="$2" LUQ_LAYERS="$3" python - <<'PYEOF'
import os
from pathlib import Path
from luq import cache
from luq.config import Config
ds, rg = os.environ["LUQ_DS"], os.environ["LUQ_RG"]
layers = [int(x) for x in os.environ["LUQ_LAYERS"].split(",") if x.strip()]
cfg = Config(model_name="meta-llama/Meta-Llama-3.1-8B", dataset=ds, ood_setting="ID",
             prompt_regime=rg)
key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
pdir = Path(cfg.cache_dir) / "pertok"
print(",".join(str(L) for L in layers if not (pdir / f"{key}__L{L}.npz").exists()))
PYEOF
}

DEV_ARGS=(--device-map "$LUQ_PT_DEVMAP")
if [ -n "$LUQ_PT_MAXMEM" ]; then DEV_ARGS+=(--max-memory "$LUQ_PT_MAXMEM"); fi
# Opt-in, and off unless a wrapper asks for it, so the sixty files already written and any future
# single-card run stay on exactly the call path that produced them.
if [ "${LUQ_PT_SKIPHEAD:-0}" = "1" ]; then DEV_ARGS+=(--skip-lm-head); fi

for DS in $LUQ_PT_DATASETS; do
  case "$DS" in
    med_quad)  REGIME="cleanv2" ;;
    asqa)      REGIME="asqa_rp12" ;;
    expertqa)  REGIME="expertqa_rp12" ;;
    factscore) REGIME="factscore_rp12" ;;
    *)         REGIME="" ;;
  esac
  for CH in "${CHUNKS[@]}"; do
    TODO="$(missing_layers "$DS" "$REGIME" "$CH")" || { echo "!!! could not resolve cache paths for $DS"; exit 4; }
    if [ -z "$TODO" ]; then
      echo "---- $DS  layers $CH  already cached, skipping"
      continue
    fi
    echo; echo "---- $DS  layers $TODO  (of chunk $CH, regime='${REGIME:-canonical}') at $(date) ----"
    python -u scripts/01p_pertoken_multi.py --model "$MODEL" --dataset "$DS" --ood ID \
        --layers "$TODO" --prompt-regime "$REGIME" --dtype fp32 --attn eager \
        "${DEV_ARGS[@]}" \
        || { echo "!!! extraction FAILED for $DS layers $TODO"; exit 3; }
  done
  echo "---- $DS done at $(date); free space now:"; df -h /rds/general/user/gs925/home | tail -1
done

echo; echo "=== done at $(date) ==="
