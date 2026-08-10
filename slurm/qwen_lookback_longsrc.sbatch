#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_lb_long
#SBATCH --output=logs/qwen_lookback_long_%j.out

# Lookback Lens on Qwen2.5-14B for the TWO LONG-SOURCE evals, xsum and cnn_dailymail.
#
#   sbatch slurm/qwen_lookback_longsrc.sbatch xsum
#   sbatch slurm/qwen_lookback_longsrc.sbatch cnn_dailymail
#
# ONE DATASET PER SUBMISSION. Together they are the better part of a day, which queues worse than
# two jobs and loses the second one entirely to a walltime kill. 01c has no resume.
#
# ---- WHY TWO CARDS, WHICH IS THE ONLY REASON THIS IS A SEPARATE SCRIPT ------------------------
# The attention forward materialises every block at once: n_layers * n_heads * min(seq,2048)^2 * 4 B
# in fp32, on top of fp32 weights. For Qwen2.5-14B (48 layers, 40 heads, 59 GB of weights):
#
#     48 * 40 * 2048^2 * 4 B = 32.2 GB      ->   59.0 + 32.2 = 91 GB
#
# which is past one A100-80GB. Both of these sets saturate `lookback_vector`'s max_seq = 2048 cap
# (measured max sequence: xsum 5964, cnn_dailymail 2537), so this is the real peak for both, not a
# worst case. The other six Qwen evals top out at 1365 tokens and fit one card -- they go through
# slurm/qwen_lookback.sbatch.
#
# ---- WHAT WE CHANGED, AND WHAT WE DELIBERATELY DID NOT ----------------------------------------
# We shard the WEIGHTS across two cards. That changes no number: max_seq, the dtype, the ratio and
# the pooling are all untouched, so these are the same features a single 128 GB card would have
# produced. The three alternatives were all rejected because each one would have made Qwen's
# Lookback a different method from Llama's, and a cross-model comparison is the entire point:
#
#   * lower max_seq below 2048   -> changes how much source each row sees. Llama truncated 23 xsum
#                                   and 61 cnn rows at 2048; a smaller cap truncates more, and
#                                   Lookback is literally a ratio of attention ON THE SOURCE.
#   * drop to bf16               -> bf16 has fp32's range so the softmax would not overflow, but
#                                   the Llama cache is fp32 and this would be a dtype difference
#                                   between the two populations, in the one feature read straight
#                                   off the attention weights.
#   * stream the layers          -> numerically identical, but it means editing features/lookback.py
#                                   mid-replication, and that file is verified to ~1e-6 against
#                                   Chuang's released code. Not worth touching for a hardware
#                                   problem hardware can solve.
#
# accelerate fills GPU 0 first, so --device-map auto ALONE can put all 59 GB on one card and leave
# us exactly where we started -- with a "sharding test" that proves nothing. --max-memory forces a
# real split and 01c asserts len(set(hf_device_map.values())) > 1 afterwards, failing at load time
# rather than eight hours in.
#
# ---- SIZING ----------------------------------------------------------------------------------
# Lookback cost goes as sum(min(seq,2048)^2), and Qwen's attention work per token is
# (48*40)/(32*32) = 1.875x Llama's. Llama measured 104 min for xsum and 251 min for cnn, so expect
# roughly 3-4 h and 8-9 h plus sharding overhead. Walltime is 24 h to leave room; if a job is
# killed anyway it restarts from scratch, so do not batch them.
#
# Host RAM: attentions move to CPU and are then torch.stack-ed, which copies -> ~2 * 32.2 = 64 GB
# peak, hence 160G. Note DoC applies a per-user memory QOS cap; this job plus a second large one
# can serialise. That is fine, it is not an error.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1

MODEL="Qwen/Qwen2.5-14B"
SLUG="Qwen_Qwen2.5-14B"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "usage: sbatch $0 <xsum|cnn_dailymail>"; exit 2
fi
for DS in "${DATASETS[@]}"; do
  case "$DS" in
    xsum|cnn_dailymail) ;;
    *) echo "REFUSING '$DS': it fits one card. Use slurm/qwen_lookback.sbatch -- this script would" \
            "hold two a100s for no reason." >&2; exit 2 ;;
  esac
done

echo "=== qwen lookback LONGSRC on $(hostname) at $(date)  datasets: ${DATASETS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in "${DATASETS[@]}"; do
  ROOT="${LUQ_CACHE_ROOT:-cache}"                 # both live in the top-level namespace
  OUT="$ROOT/features/${SLUG}__${DS}__ID__lookback.npz"
  REC="$ROOT/records/${SLUG}__${DS}__ID.jsonl"

  echo "================ $DS ================ $(date)"
  [ -f "$REC" ] || { echo "!!! no records at $REC -- skipping $DS"; continue; }
  echo "RECORDS $DS: $(wc -l < "$REC") rows"
  echo "RECORDS SHA256 $DS: $(sha256sum "$REC" | cut -d' ' -f1)"

  if [ -f "$OUT" ] && [ "${LUQ_OVERWRITE:-0}" != "1" ]; then
    echo "REFUSING $DS: $OUT exists and 01c would overwrite it. LUQ_OVERWRITE=1 to force."
    continue
  fi

  # 40 GiB per card: above half the 59 GB of weights, so accelerate cannot fit them on one device
  # and is forced to split, while still leaving ~40 GB per card for the attention tensor.
  python -u scripts/01c_lookback.py --model "$MODEL" --dataset "$DS" --ood ID \
      --device-map auto --max-memory "0=40GiB,1=40GiB" \
      || { echo "!!! lookback FAILED for $DS"; continue; }
  ls -la "$OUT"
done

echo "=== qwen lookback LONGSRC done at $(date) ==="
