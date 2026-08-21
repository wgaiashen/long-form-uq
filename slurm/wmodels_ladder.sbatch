#!/bin/bash
#SBATCH --partition=t4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=wm_ladder
#SBATCH --output=logs/wmodels_ladder_%j.out

# GENERAL W-Models Stage-A ladder on DoC, ONE eval per job: ID + DiffTask-long + 1ds-Diff-long,
# 3 seeds. Slurm twin of pbs/wmodels_ladder.pbs.
# (slurm/wmodels_qwen32_ladder.sbatch stays as the Qwen32-specific array.)
#
# CPU ON PURPOSE, SAME AS THE PBS TWIN. probedriftlong.py falls back to device=cpu, and the
# ladder reads only cached pertok/features -- no model forward pass -- so it does not need a40/a100
# at all. Routed to t4 (usually the least contended DoC partition) rather than competing with
# extraction/postextract for the scarce big cards. DoC's QoS still force-assigns 1 GPU regardless of
# partition (see the project implementation notes), but that GPU is not needed and mostly goes unused.
#
# --wmsp-only wmsp_norm,wmsp_shrink2 IS THE PRE-REGISTRATION, NOT AN OPTIMISATION (the multi-model panel registration §4).
# --perex-dir IS REQUIRED -- the verification gate compares per-example vectors, not PRR.
# THE CELL COUNT IS ASSERTED AFTERWARDS -- a missing dataset silently shrinks the grid instead of
# erroring, so `probedriftlong` printing "<d>: no pertok cache -> skip" must be caught by hand.
#
#   sbatch --export=ALL,LUQ_MODEL=meta-llama/Llama-3.1-8B-Instruct,LUQ_EVAL=xsum slurm/wmodels_ladder.sbatch
#   (remember the DoC --export=ALL gotcha: export the vars in-shell instead and drop --export)

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  HF_HOME=$HF_HOME"

MODEL="${LUQ_MODEL:?set LUQ_MODEL}"
EVAL="${LUQ_EVAL:?set LUQ_EVAL}"
SEEDS="${LUQ_SEEDS:-1,2,3}"
RUNGS="${LUQ_RUNGS:-ID,DiffTask,1ds-Diff}"
# λ set. Default is the ORIGINAL pre-registered pair, so every existing invocation is unchanged.
# Override to add the λ=1.5 supplementary sensitivity (the multi-model panel registration, deviation 5, 2026-08-19):
#   LUQ_WMSP='wmsp_norm,wmsp_shrink1_5,wmsp_shrink2'
# The PRIMARY is always λ=2 vs λ=0. Adding λ=1.5 does not change that, and the better of 1.5 and 2
# is NEVER reported -- taking the max over two λ values is a two-point grid search on a population
# that exists precisely because nothing has been selected on it.
WMSP="${LUQ_WMSP:-wmsp_norm,wmsp_shrink2}"

case "$EVAL" in
  pubmed_qa|xsum|cnn_dailymail|samsum|asqa|factscore) ;;
  *) echo "!!! '$EVAL' is not in the six-dataset panel (the multi-model panel registration §2)." >&2; exit 3 ;;
esac

PANEL="pubmed_qa,xsum,cnn_dailymail,samsum,asqa,factscore"
SLUG="$(printf '%s' "$MODEL" | tr '/' '_')"
OUT="${LUQ_OUT:-results/wmodels_stageA__${SLUG}__${EVAL}.csv}"
PEREX="${LUQ_PEREX:-results/perex_wmodels/${SLUG}}"
mkdir -p "$PEREX"

LAYER=$(MODEL="$MODEL" python - <<'PY'
import math, os
from transformers import AutoConfig
print(math.ceil(AutoConfig.from_pretrained(os.environ["MODEL"]).num_hidden_layers / 2) - 1)
PY
)
[ -z "$LAYER" ] && { echo "!!! could not resolve layer for $MODEL" >&2; exit 4; }

echo "=== W-Models Stage-A ladder (DoC) ===" >&2
echo "  model=$MODEL layer=$LAYER eval=$EVAL rungs=$RUNGS seeds=$SEEDS" >&2
echo "  wmsp=$WMSP" >&2
echo "  out=$OUT perex=$PEREX host=$(hostname) at $(date)" >&2

python -u scripts/checks/probedriftlong.py \
    --model "$MODEL" --layer "$LAYER" \
    --evals "$EVAL" \
    --sources "$PANEL" \
    --rungs "$RUNGS" \
    --seeds "$SEEDS" \
    --wmsp-only "$WMSP" \
    --perex-dir "$PEREX" \
    --out "$OUT"
STATUS=$?

if [ "$MODEL" != "meta-llama/Meta-Llama-3.1-8B" ]; then
  echo "--- verifying special-token registration fired ---" >&2
  grep -q "special-token ids registered" "logs/wmodels_ladder_${SLURM_JOB_ID}.out" 2>/dev/null \
    || echo "!!! 'special-token ids registered' not found in the log -- verify before recording any number" >&2
fi

echo "=== $EVAL ladder done status=$STATUS at $(date) ===" >&2
exit $STATUS
