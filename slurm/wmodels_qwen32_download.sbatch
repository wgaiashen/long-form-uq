#!/bin/bash
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=wm32_dl
#SBATCH --output=logs/wmodels_qwen32_download_%j.out

# W-Models: download Qwen/Qwen2.5-32B on DoC and verify it. Slurm twin of pbs/wmodels_download.pbs.
#
# ⚠️ HF_HOME IS OVERRIDDEN TO /vol/bitbucket, AND THAT IS THE WHOLE POINT OF THIS SCRIPT.
# pbs/_env.sh defaults DoC's HF_HOME to /vol/gpudata/gs925-msc_project/hf_cache, which sits inside
# the HARD 50GB CephFS quota that already holds Qwen-14B (27.5GB) and had ~4GB free as of
# 2026-08-08. Qwen2.5-32B is ~65GB in bf16 and cannot possibly fit there; the download would die
# with "Disk quota exceeded" partway through and leave a cache that looks present but is not.
# /vol/bitbucket is NFS with no user quota (183T, ~15T free, verified 2026-08-15) and already holds
# 88GB of luq_cache. Exporting BEFORE sourcing works because _env.sh assigns with `:=`, which
# respects an already-set value.
#
# ⚠️ NO --gres=gpu HERE (fixed 2026-08-16, was --gres=gpu:1). This script never touches CUDA -- it
# is snapshot_download + a tokenizer load, both CPU/net, matching its own PBS twin
# (pbs/wmodels_download.pbs) and the Qwen-14B precedent (pbs/download_qwen.pbs), neither of which
# requests a GPU. The original comment claiming "a GPU is requested so the load test is meaningful"
# did not match the code (no model is ever loaded here) and cost this job a ~5-day backfill queue
# behind other users' long-running jobs on the 3-node a100 partition, whose GPUs were all allocated
# while CPU/mem had ample headroom (verified via `scontrol show node`). Do not re-add it.
#
#   sbatch slurm/wmodels_qwen32_download.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache      # MUST precede _env.sh (see above)
mkdir -p "$HF_HOME"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME
unset HF_HUB_OFFLINE                              # this job MUST reach the network

MODEL="Qwen/Qwen2.5-32B"
echo "=== download $MODEL -> $HF_HOME on $(hostname) at $(date) ===" >&2
df -h /vol/bitbucket 2>/dev/null | tail -1 >&2

if ! timeout 60 curl -sSf -o /dev/null https://huggingface.co 2>&1; then
  echo "!!! NO INTERNET on $(hostname). Cannot download here." >&2; exit 2
fi
echo "  network OK" >&2

python -u - <<'PY'
import os, json, glob, math
from huggingface_hub import snapshot_download
MODEL = "Qwen/Qwen2.5-32B"
path = snapshot_download(MODEL, max_workers=4,
                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("snapshot at:", path, flush=True)

cfg = json.load(open(os.path.join(path, "config.json")))
n, h = cfg["num_hidden_layers"], cfg["hidden_size"]
print(f"  layers={n} hidden={h} vocab={cfg.get('vocab_size')} dtype={cfg.get('torch_dtype')}", flush=True)
# These are the values the fixed layer rule and the fail-loud hidden-dim registry were built on.
assert n == 64, f"expected 64 blocks, got {n}"
assert h == 5120, f"expected hidden 5120, got {h}"
print(f"  fixed rule ceil(N/2)-1 -> layer {math.ceil(n/2)-1}   (registry says 31)", flush=True)

shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
total = sum(os.path.getsize(f) for f in shards)
print(f"  {len(shards)} safetensors shards, {total/2**30:.1f} GiB", flush=True)
assert total > 55 * 2**30, f"only {total/2**30:.1f} GiB downloaded — incomplete"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"  tokenizer OK: vocab={len(tok)} n_special={len(tok.all_special_ids)}", flush=True)
print("DOWNLOAD VERIFIED", flush=True)
PY
STATUS=$?
echo "=== done status=$STATUS at $(date) ===" >&2
exit $STATUS
