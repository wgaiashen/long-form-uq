#!/bin/bash
#SBATCH --partition=t4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=wm_dl
#SBATCH --output=logs/wmodels_download_%j.out

# GENERAL W-Models download on DoC, any model. Slurm twin of pbs/wmodels_download.pbs.
# (slurm/wmodels_qwen32_download.sbatch stays as the Qwen32-specific job.)
#
# ⚠️ NO --gres=gpu HERE, AND partition DEFAULTS TO t4, NOT a40/a100. This script is
# snapshot_download + a tokenizer load -- no CUDA at all -- so it should never wait behind a
# GPU-bound partition. Override --partition at submit time only if t4 itself is contended; any
# idle GPU-labelled partition works identically for this job since it does no GPU compute.
#
# ⚠️ HF_HOME IS OVERRIDDEN TO /vol/bitbucket, AND THAT IS THE WHOLE POINT OF THIS SCRIPT.
# pbs/_env.sh defaults DoC's HF_HOME to /vol/gpudata/gs925-msc_project/hf_cache, which sits inside
# the HARD 50GB CephFS quota. /vol/bitbucket is NFS with no user quota (verified 2026-08-15).
#
#   sbatch --export=ALL,LUQ_MODEL=meta-llama/Llama-3.1-8B-Instruct,LUQ_LAYERS=32,LUQ_HIDDEN=4096,LUQ_MIN_GIB=14 \
#          slurm/wmodels_download.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache      # MUST precede _env.sh (it assigns with :=)
mkdir -p "$HF_HOME"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME
unset HF_HUB_OFFLINE                              # this job MUST reach the network

MODEL="${LUQ_MODEL:?set LUQ_MODEL}"
LAYERS="${LUQ_LAYERS:?set LUQ_LAYERS (num_hidden_layers to assert)}"
HIDDEN="${LUQ_HIDDEN:?set LUQ_HIDDEN (hidden_size to assert)}"
MIN_GIB="${LUQ_MIN_GIB:-10}"

echo "=== download $MODEL -> $HF_HOME on $(hostname) at $(date) ===" >&2
echo "    asserting layers=$LAYERS hidden=$HIDDEN min_size=${MIN_GIB}GiB" >&2

if ! timeout 60 curl -sSf -o /dev/null https://huggingface.co 2>&1; then
  echo "!!! NO INTERNET on $(hostname). Cannot download here." >&2; exit 2
fi
echo "  network OK" >&2

MODEL="$MODEL" LAYERS="$LAYERS" HIDDEN="$HIDDEN" MIN_GIB="$MIN_GIB" python -u - <<'PY'
import os, json, glob, math
from huggingface_hub import snapshot_download

MODEL = os.environ["MODEL"]
LAYERS = int(os.environ["LAYERS"]); HIDDEN = int(os.environ["HIDDEN"])
MIN_GIB = float(os.environ["MIN_GIB"])

path = snapshot_download(MODEL, max_workers=4,
                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("snapshot at:", path, flush=True)

cfg = json.load(open(os.path.join(path, "config.json")))
n, h = cfg["num_hidden_layers"], cfg["hidden_size"]
print(f"  layers={n} hidden={h} vocab={cfg.get('vocab_size')} dtype={cfg.get('torch_dtype')}", flush=True)
assert n == LAYERS, f"expected {LAYERS} blocks, got {n}"
assert h == HIDDEN, f"expected hidden {HIDDEN}, got {h}"
print(f"  fixed rule ceil(N/2)-1 -> layer {math.ceil(n/2)-1}", flush=True)

shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
total = sum(os.path.getsize(f) for f in shards)
print(f"  {len(shards)} safetensors shards, {total/2**30:.1f} GiB", flush=True)
assert total > MIN_GIB * 2**30, f"only {total/2**30:.1f} GiB downloaded — incomplete"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"  tokenizer OK: vocab={len(tok)} n_special={len(tok.all_special_ids)}", flush=True)
print("DOWNLOAD VERIFIED", flush=True)
PY
STATUS=$?
echo "=== done status=$STATUS at $(date) ===" >&2
exit $STATUS
