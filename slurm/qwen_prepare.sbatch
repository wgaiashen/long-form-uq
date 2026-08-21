#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=qwen_prep
#SBATCH --output=logs/qwen_prepare_%j.out

# PREPARE the DoC side of the Qwen2.5-14B second-model replication. Two jobs in one allocation:
#   1. report the A100's actual memory ON THE CARD (the decision that shapes the whole run), and
#   2. download the weights.
#
# Why both here. fp32 Qwen-14B is ~59GB. On a 40GB card it must be sharded across two GPUs with
# max_memory + an explicit assert that it really spans >1 device (device_map="auto" alone fills GPU 0
# first, so a "sharding test" built on it is vacuous). On an 80GB card none of that applies and the
# run is a plain single-GPU job. `sinfo` advertises nvidia_a100_80gb_pcie, but sinfo reports what the
# scheduler was CONFIGURED with, not what the process is handed, so confirm on the card.
#
# Why the download is in a Slurm job and not on the shell VM: cloud-vm-* is a small jump box and kills
# long processes; a parallel snapshot_download of a multi-GB model is exactly what it kills. The
# --mem=64G here is what lets the parallel download succeed.
#
# HF_HUB_OFFLINE is deliberately NOT set -- this is the one job that needs the network.
# HF_HOME resolves to /vol/gpudata/.../hf_cache, which is a SYMLINK to /vol/bitbucket/gs925/hf_cache,
# i.e. OUTSIDE the 50GB CephFS quota. The 27.5GB of weights therefore do not count against it.
# Qwen2.5-14B is ungated apache-2.0, so no HF token is needed.
#
#   sbatch slurm/qwen_prepare.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME

echo "=== node $(hostname) at $(date) ==="

echo "--- 1. THE GPU (this is the answer to brief section 2) ---"
nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'EOF'
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible -- the answer below would be a guess, so failing loudly")
p = torch.cuda.get_device_properties(0)
gib = p.total_memory / 2**30
print(f"torch sees: {p.name}  {gib:.1f} GiB")
# fp32 14B weights ~= 14.77e9 params * 4 bytes. Leave room for activations + the KV cache.
print(f"fp32 Qwen-14B weights ~59 GB -> {'FITS on one card' if gib > 66 else 'DOES NOT FIT: shard across 2 GPUs'}")
EOF

echo "--- 2. DOWNLOAD Qwen/Qwen2.5-14B ---"
python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-14B", max_workers=4,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("snapshot at:", p)
EOF

echo "--- 3. VERIFY the download (config shape + shard count, not mere presence) ---"
python - <<'EOF'
from transformers import AutoConfig, AutoTokenizer
c = AutoConfig.from_pretrained("Qwen/Qwen2.5-14B")
print(f"layers={c.num_hidden_layers} hidden={c.hidden_size} vocab={c.vocab_size}")
assert (c.num_hidden_layers, c.hidden_size, c.vocab_size) == (48, 5120, 152064), "unexpected Qwen config"
t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
# Trap 4.7: Qwen prepends no BOS (Llama prepends <|begin_of_text|>) and its specials sit at ~151643,
# not >=128000. Record both so the RCS side can see what this population actually used.
print("bos_token_id:", t.bos_token_id, "| eos_token_id:", t.eos_token_id)
print("special ids :", sorted(t.all_special_ids)[:6], "...")
print("no-BOS check:", t("hello")["input_ids"][:3])
EOF

echo "--- 4. FOOTPRINT (weights must land off the quota) ---"
du -sh "$HF_HOME/hub/models--Qwen--Qwen2.5-14B" 2>/dev/null
ls "$HF_HOME/hub/models--Qwen--Qwen2.5-14B"/snapshots/*/ | grep -c safetensors || true
echo "quota now:"
python -c "
import subprocess
g=lambda a:int(subprocess.run(['getfattr','--only-values','-n',a,'/vol/gpudata/gs925-msc_project'],capture_output=True,text=True).stdout.strip())
u,q=g('ceph.dir.rbytes'),g('ceph.quota.max_bytes')
print(f'  used {u/2**30:.2f} GiB / {q/2**30:.2f} GiB -> FREE {(q-u)/2**30:.2f} GiB')"

echo "=== done at $(date) ==="
