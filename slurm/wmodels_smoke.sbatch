#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=wm_smoke
#SBATCH --output=logs/wmodels_smoke_%j.out

# W-Models prompt-regime validity gate on DoC (prereg/multimodel_far_ood_panel.md §6).
# 40 DETERMINISTIC rows per dataset x 6 datasets, one GPU slot, then the report.
# Slurm twin of pbs/wmodels_smoke.pbs, plus --chat-template support (D5: raw few-shot
# failed the gate on 3/6 Llama-Instruct datasets on chatter -- see worklog 2026-08-19).
#
# THIS GATE IS UQ-PERFORMANCE-INDEPENDENT BY CONSTRUCTION. It reads degeneracy, empties,
# invented continuations, cap behaviour and generation length. It never computes or looks at PRR.
#
# --limit takes the FIRST N rows by index, so the smoke set is deterministic, not sampled.
# --prompt-regime <slug>_probe isolates every write, so a probe cache can never contaminate a
# real one.
#
#   sbatch --export=ALL,LUQ_MODEL=meta-llama/Llama-3.1-8B-Instruct,LUQ_DTYPE=fp32,LUQ_PROBE=llama31i_chat_probe,LUQ_CHAT=1 \
#          slurm/wmodels_smoke.sbatch
#   (remember the DoC --export=ALL gotcha: export the vars in-shell instead and drop --export)

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1

MODEL="${LUQ_MODEL:?set LUQ_MODEL}"
DTYPE="${LUQ_DTYPE:?set LUQ_DTYPE}"
PROBE="${LUQ_PROBE:?set LUQ_PROBE (isolated cache namespace)}"
ATTN="${LUQ_ATTN:-eager}"
N="${LUQ_LIMIT:-40}"
CHAT=""; [ "${LUQ_CHAT:-0}" = "1" ] && CHAT="--chat-template"
OUTDIR="${LUQ_OUTDIR:-results/wmodels_smoke}"
mkdir -p "$OUTDIR"

echo "=== W-Models validity smoke (DoC) ===" >&2
echo "  model=$MODEL dtype=$DTYPE attn=$ATTN probe_regime=$PROBE rows_per_dataset=$N chat_template=${LUQ_CHAT:-0}" >&2
echo "  host=$(hostname) at $(date)" >&2

FAILED=""
for DS in cnn_dailymail pubmed_qa xsum asqa samsum factscore; do
  case "$DS" in
    cnn_dailymail) BUD=128; EXTRA="" ;;
    pubmed_qa)     BUD=128; EXTRA="--truncate-long" ;;
    xsum)          BUD=56;  EXTRA="" ;;
    asqa)          BUD=256; EXTRA="--repetition-penalty 1.2" ;;
    samsum)        BUD=56;  EXTRA="" ;;
    factscore)     BUD=256; EXTRA="--repetition-penalty 1.2" ;;
  esac
  echo "--- smoke $DS (budget $BUD, $N rows) ---" >&2
  # shellcheck disable=SC2086
  python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
      --dtype "$DTYPE" --attn "$ATTN" \
      --prompt-regime "$PROBE" --max-new-tokens "$BUD" --max-new-tokens-cap "$BUD" \
      --limit "$N" $CHAT $EXTRA || FAILED="$FAILED $DS"
done

[ -n "$FAILED" ] && echo "!!! extraction failed for:$FAILED" >&2

echo "=== validity report (UQ-independent signals only) ===" >&2
python -u scripts/checks/generation_quality.py \
    --model "$MODEL" \
    --datasets cnn_dailymail,pubmed_qa,xsum,asqa,samsum,factscore \
    --regime "$PROBE" \
    --out "$OUTDIR/smoke__$(printf '%s' "$MODEL" | tr '/' '_')__${PROBE}.csv"

echo "--- gate verdict ---" >&2
python -u scripts/checks/wmodels_gate.py \
    --smoke-csv "$OUTDIR/smoke__$(printf '%s' "$MODEL" | tr '/' '_')__${PROBE}.csv"
GATE_STATUS=$?

echo "=== smoke done at $(date), gate exit=$GATE_STATUS ===" >&2
[ -n "$FAILED" ] && exit 9
exit $GATE_STATUS
