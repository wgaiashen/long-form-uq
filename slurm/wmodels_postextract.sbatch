#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=wm_post
#SBATCH --output=logs/wmodels_post_%j.out

# GENERAL W-Models post-extraction on DoC, one dataset per job: 01e_repool -> pertok consistency
# GUARD -> 01h_pertoken. Slurm twin of pbs/wmodels_postextract.pbs.
# (slurm/wmodels_qwen32_postextract.sbatch stays as the Qwen32-specific array.)
#
# ⚠️ 01e_repool IS MANDATORY, NOT OPTIONAL -- see pbs/wmodels_postextract.pbs for the full reasoning
# (inline vs teacher-forced pooling differ by ~8e-3 absolute, not precision).
#
# ⚠️ THE LAYER IS COMPUTED, NOT TRANSCRIBED, and cross-checked against the committed registry in
# scripts/checks/slice_caches_truncated.py.
#
#   sbatch --export=ALL,LUQ_MODEL=google/gemma-2-9b,LUQ_DTYPE=bf16,LUQ_DATASET=xsum \
#          slurm/wmodels_postextract.sbatch
#   (remember the DoC --export=ALL gotcha: export the vars in-shell instead and drop --export)
#
# CHAT-TEMPLATE MODE: set LUQ_CHAT=1 to override the regime to the chat namespace on EVERY dataset
# (including asqa/factscore, whose default regimes hold raw-few-shot records) -- must match whatever
# --chat-template extraction actually wrote to, or this reads the wrong (or a missing) cache silently.
#   sbatch --export=ALL,LUQ_MODEL=meta-llama/Llama-3.1-8B-Instruct,LUQ_DTYPE=fp32,LUQ_DATASET=xsum,LUQ_CHAT=1 \
#          slurm/wmodels_postextract.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  HF_HOME=$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${LUQ_MODEL:?set LUQ_MODEL}"
DTYPE="${LUQ_DTYPE:?set LUQ_DTYPE (must match extraction)}"
DS="${LUQ_DATASET:?set LUQ_DATASET}"
ATTN="${LUQ_ATTN:-eager}"

case "$DS" in
  cnn_dailymail|xsum|samsum) REGIME="" ;;
  pubmed_qa)                 REGIME="" ;;
  asqa)                      REGIME="asqa_rp12" ;;
  factscore)                 REGIME="factscore_rp12" ;;
  *) echo "!!! '$DS' is not in the six-dataset panel (prereg M5 §2)." >&2; exit 3 ;;
esac
if [ "${LUQ_CHAT:-0}" = "1" ]; then
  REGIME="${LUQ_CHAT_REGIME:-llama31i_chat}"
fi
RG=""; [ -n "$REGIME" ] && RG="--prompt-regime $REGIME"

# pubmed_qa was generated newline-stopped; the re-pool must reproduce the same token window.
TL=""; [ "$DS" = "pubmed_qa" ] && TL="--truncate-long"

LAYER=$(MODEL="$MODEL" python - <<'PY'
import math, os
from transformers import AutoConfig
n = AutoConfig.from_pretrained(os.environ["MODEL"]).num_hidden_layers
print(math.ceil(n / 2) - 1)
PY
)
[ -z "$LAYER" ] && { echo "!!! could not resolve layer for $MODEL" >&2; exit 4; }

MODEL="$MODEL" LAYER="$LAYER" python - <<'PY' || exit 5
import ast, os, sys
src = open("scripts/checks/slice_caches_truncated.py").read()
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LAYER":
        reg = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(n.value.keys, n.value.values)}
        break
m, got = os.environ["MODEL"], int(os.environ["LAYER"])
if m not in reg:
    sys.exit(f"!!! {m} is not in the LAYER registry -- add it before running")
if reg[m] != got:
    sys.exit(f"!!! layer disagreement for {m}: rule says {got}, registry says {reg[m]}")
print(f"  layer {got} agrees with the registry", flush=True)
PY

echo "=== W-Models post-extract (DoC) ===" >&2
echo "  model=$MODEL dtype=$DTYPE attn=$ATTN dataset=$DS layer=$LAYER regime='${REGIME:-<default>}'" >&2
echo "  host=$(hostname) at $(date)" >&2

echo "--- 1/3 re-pool (teacher-forced) ---" >&2
# shellcheck disable=SC2086
python -u scripts/01e_repool.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype "$DTYPE" --attn "$ATTN" $RG $TL || { echo "!!! repool failed" >&2; exit 6; }

echo "--- 2/3 per-token cache at layer $LAYER ---" >&2
# shellcheck disable=SC2086
python -u scripts/01h_pertoken.py --model "$MODEL" --dataset "$DS" --ood ID \
    --layer "$LAYER" --dtype "$DTYPE" --attn "$ATTN" $RG || { echo "!!! pertoken failed" >&2; exit 8; }

echo "--- 3/3 feature/pertok consistency guard (AFTER pertok, which it compares against) ---" >&2
# shellcheck disable=SC2086
python -u scripts/checks/feature_pertok_consistency.py --model "$MODEL" --datasets "$DS" \
    --layer "$LAYER" $RG --full --require-checked \
    || { echo "!!! CONSISTENCY GUARD FAILED OR SKIPPED -- the caches are not verified" >&2; exit 7; }

echo "=== $DS post-extract done at $(date) ===" >&2
