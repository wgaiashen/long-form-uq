#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_lb
#SBATCH --output=logs/qwen_lookback_%j.out

# Lookback Lens attention features on Qwen2.5-14B, for the SIX SHORT-SOURCE evals.
#
#   sbatch slurm/qwen_lookback.sbatch pubmed_qa med_quad samsum asqa expertqa factscore
#
# ⛔ xsum and cnn_dailymail DO NOT BELONG HERE -- they need two cards. Use
# slurm/qwen_lookback_longsrc.sbatch. The allowlist below refuses them rather than letting the
# job OOM eight hours in.
#
# WHY THIS EXISTS. Same reason as slurm/qwen_ptrue.sbatch: prereg/M2_qwen14b_replication.md §4
# names Lookback in the scored baseline set and no Qwen cache existed, so the first ladder run
# left the column ABSENT. No registered claim reads it, so the replication verdict does not
# depend on this job -- it closes a stated gap against our own pre-registration.
#
# fp32 + eager, both forced inside 01c and both required, not chosen: SDPA returns no attention
# weights at all, and eager attention in fp16 overflows to NaN in the softmax. Llama's cache was
# built the same way, so this is a transfer, not a re-selection.
#
# ---- THE MEMORY ARITHMETIC, WHICH IS WHY THERE ARE TWO SCRIPTS -------------------------------
# `recompute_states` sets output_attentions=True, so the forward materialises EVERY block's
# attention at once before any of it moves to CPU:
#
#     GPU peak  =  fp32 weights  +  n_layers * n_heads * min(seq, 2048)^2 * 4 B
#
# Llama-3.1-8B:  32 GB + 32*32*2048^2*4 = 32.0 + 17.2 = 49 GB   -> one 80 GB card, always.
# Qwen2.5-14B:   59 GB + 48*40*2048^2*4 = 59.0 + 32.2 = 91 GB   -> PAST one 80 GB card.
#
# The 2048 term is `lookback_vector`'s max_seq cap, so it is the ceiling, not the typical cost.
# Measured max sequence (prompt + generation) on the Qwen records, which is what actually decides:
#
#     dataset         rows   med    p95    max   >2048   attn peak   GPU total
#     asqa             948   109    295    302       0     0.7 GB      60 GB
#     factscore        500   165    272    275       0     0.6 GB      60 GB
#     expertqa        2016   148    439    584       0     2.6 GB      62 GB
#     samsum          1800   173    390   1122       0     9.7 GB      69 GB
#     pubmed_qa       3800   436    629   1178       0    10.7 GB      70 GB
#     med_quad        1800  1338   1348   1365       0    14.3 GB      73 GB
#     ----------------------------------------- the six above fit ONE card -----------------------
#     cnn_dailymail   3800   847   1751   2537      61    32.2 GB      91 GB   <- two cards
#     xsum            3800   436   1210   5964      23    32.2 GB      91 GB   <- two cards
#
# The card follows the LONGEST SEQUENCE, never the row count -- factscore has 500 rows and asqa
# 948, and both are trivial, while med_quad's 1800 rows are the most expensive of the six because
# its sources are uniformly long (p5 to max is 1338-1365, essentially a fixed length).
#
# Host RAM: the attentions are moved to CPU and then torch.stack-ed, which COPIES, so the peak is
# about 2x the attention term. 2 * 14.3 = ~29 GB for the worst of the six, hence 96G.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1

MODEL="Qwen/Qwen2.5-14B"
SLUG="Qwen_Qwen2.5-14B"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "usage: sbatch $0 <dataset> [dataset ...]   (short-source sets only)"; exit 2
fi

regime_for() {
  case "$1" in
    pubmed_qa|med_quad|samsum) echo "" ;;
    asqa)      echo "asqa_rp12" ;;
    expertqa)  echo "expertqa_rp12" ;;
    factscore) echo "factscore_rp12" ;;
    xsum|cnn_dailymail)
      echo "REFUSING '$1': its attention peak is 32.2 GB on top of 59 GB of fp32 weights (~91 GB)," \
           "which does not fit one A100-80GB. Use slurm/qwen_lookback_longsrc.sbatch." >&2; return 1 ;;
    *) echo "REFUSING '$1': not a ProbeDriftLong long eval." >&2; return 1 ;;
  esac
}
for DS in "${DATASETS[@]}"; do regime_for "$DS" >/dev/null || exit 2; done

echo "=== qwen lookback on $(hostname) at $(date)  datasets: ${DATASETS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in "${DATASETS[@]}"; do
  RG=$(regime_for "$DS")
  ROOT="${LUQ_CACHE_ROOT:-cache}${RG:+/$RG}"
  OUT="$ROOT/features/${SLUG}__${DS}__ID__lookback.npz"
  REC="$ROOT/records/${SLUG}__${DS}__ID.jsonl"

  echo "================ $DS (regime '${RG:-<base>}') ================ $(date)"
  [ -f "$REC" ] || { echo "!!! no records at $REC -- skipping $DS"; continue; }
  echo "RECORDS $DS: $(wc -l < "$REC") rows"
  echo "RECORDS SHA256 $DS: $(sha256sum "$REC" | cut -d' ' -f1)"

  # 01c has no skip-if-exists and no resume: it overwrites whatever is at the output path.
  if [ -f "$OUT" ] && [ "${LUQ_OVERWRITE:-0}" != "1" ]; then
    echo "REFUSING $DS: $OUT exists and would be silently overwritten. LUQ_OVERWRITE=1 to force."
    continue
  fi

  python -u scripts/01c_lookback.py --model "$MODEL" --dataset "$DS" --ood ID \
      --prompt-regime "$RG" \
      || { echo "!!! lookback FAILED for $DS"; continue; }
  ls -la "$OUT"
done

echo "=== qwen lookback ALL DONE at $(date) ==="
