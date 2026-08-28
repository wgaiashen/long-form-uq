#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=wm_rpsmoke
#SBATCH --output=logs/wmodels_reppen_smoke_%j.out

# One-off diagnostic, generalizes wmodels_expertqa_reppen_smoke.sbatch to any --repetition-penalty
# dataset: is Qwen2.5-32B's degeneracy on ExpertQA (43.65% severe on the full 2016-row chat_qwen32
# population) and FActScore (7.6% severe, 3.0% fabricated) driven by --repetition-penalty 1.2, and
# does a lower value fix it? Direct text inspection of flagged rows showed genuine word-salad (a
# 134-content-word run with zero function words on ExpertQA) and hallucinated next-turn
# continuations (ExpertQA: "_user\nPlease write the code..."; FActScore idx142: a biography pivoting
# mid-sentence into "# Given Prompt:\n## A program to calculate area of circle...") -- not a detector
# false positive. Llama-Instruct's ExpertQA used the identical flag and stayed clean (1.04%), so this
# is model-specific (Qwen2.5-32B), not dataset-specific.
#
# Budgets/penalty transcribed from wmodels_extract.sbatch's own case block (asqa/factscore BUD=256,
# expertqa BUD=384; all three at repetition-penalty=1.2 in the real population) so the smoke exactly
# matches the real generation config apart from the swept penalty.
#
# LUQ_REPPEN sweeps the value under test; DETERMINISTIC first-N rows via --limit so repeat runs at
# different penalties are directly comparable. Isolated into its own prompt-regime,
# chat_qwen32_smoke_<dataset>_rp<value>, so nothing here can ever collide with or contaminate the
# real chat_qwen32 population.
#
#   sbatch --export=ALL,LUQ_DATASET=expertqa,LUQ_REPPEN=1.1 slurm/wmodels_reppen_smoke.sbatch
#   sbatch --export=ALL,LUQ_DATASET=factscore,LUQ_REPPEN=1.1 slurm/wmodels_reppen_smoke.sbatch
#   (remember the DoC --export=ALL gotcha: export the vars in-shell instead and drop --export)

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/vol/bitbucket/gs925/hf_cache
mkdir -p "$HF_HOME" logs
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-32B"
DS="${LUQ_DATASET:?set LUQ_DATASET (expertqa or factscore)}"
REPPEN="${LUQ_REPPEN:?set LUQ_REPPEN, e.g. 1.1}"
N="${LUQ_LIMIT:-150}"

case "$DS" in
  expertqa)  BUD=384 ;;
  factscore) BUD=256 ;;
  *) echo "!!! '$DS' is not one of the two --repetition-penalty datasets this smoke covers" >&2; exit 3 ;;
esac

TAG="rp$(printf '%s' "$REPPEN" | tr -d '.')"
REGIME="chat_qwen32_smoke_${DS}_${TAG}"

echo "=== repetition-penalty smoke (DoC) ===" >&2
echo "  model=$MODEL dataset=$DS budget=$BUD repetition_penalty=$REPPEN rows=$N regime=$REGIME" >&2
echo "  host=$(hostname) at $(date)" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --dtype bf16 --attn sdpa --chat-template \
    --prompt-regime "$REGIME" --max-new-tokens "$BUD" --max-new-tokens-cap "$BUD" \
    --repetition-penalty "$REPPEN" --limit "$N" \
    || { echo "!!! extraction FAILED"; exit 1; }

echo "--- quality report ---" >&2
python -u scripts/checks/generation_quality.py --model "$MODEL" --datasets "$DS" \
    --regime "$REGIME" \
    --out "results/wmodels_smoke/${DS}_reppen_${TAG}__Qwen_Qwen2.5-32B.csv"

echo "=== smoke done at $(date) ===" >&2
