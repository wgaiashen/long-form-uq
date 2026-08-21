#!/bin/bash
#SBATCH --job-name=luq-mq-span
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/med_quad_v2_span-%j.out

# A.2c -- med_quad v2, THE SHIPPING ARM: budget 768, v1 decoding, cut at the answer span.
#
# WHY THIS AND NOT THE OTHER TWO. Three arms have now failed, and for one reason:
#   v1 (128)          real answers truncated; 97.7% at the cap
#   n-gram only (768) median sits EXACTLY at the cap, 92.6% fabricated -- the same defect, bigger number
#   rep-pen 1.2 (768) 30% EMPTY (all emitting <|end_of_text|> alone), MORE looping, worse fabrication
#
# THE PROBLEM WAS NEVER REPETITION. Base Llama is not instruction-tuned: it finishes the answer and
# then CONTINUES THE FEW-SHOT FORMAT, writing a fresh "Question:" and inventing both question and
# answer, often on an unrelated topic. A repetition penalty cannot fix "invents a new question", which
# is why it perturbed the decoding without touching the mechanism. Lowering it to 1.05-1.1 would do
# less of the same wrong thing.
#
# So: STOP AT THE BOUNDARY -- this project's own documented convention (short-form cuts at the first
# newline AT EXTRACTION TIME so record, logprobs, features and label describe the same text).
# --truncate-answer-span applies luq.answer_span's per-dataset rules at exactly that site, BEFORE
# out.scores[:n_gen] and BEFORE the hidden-state pooling.
#
# EXPECTED, simulated on the existing 768 generations and verified against the real tokenizer:
#   median real answer 591 chars (v1: 498)   mean 948 (v1: 415)
#   %empty 0    %fabricated ~2.0-2.3 (from 92.6)    median NOT at the cap    true looping 0.06%
#   char->token mapping: overshoot <=1 char, ZERO undershoot, so no real answer text is dropped
#
# REPORT ON LANDING, in this order:
#   1. %empty  (<=2%, the hard gate)
#   2. p50/p95/p99 length -- 768 must not bind on the tail
#   3. %fabr and ansfrac -- expect ~2% and ~1.0
#   4. %severe WITH run>=25 SEPARATELY (most of med_quad's severe flags are the whitespace rule, which
#      its own gold references trip on 24.8% of rows)
#
#   LUQ_SMOKE=1 sbatch slurm/med_quad_v2_span_doc.sbatch   # 50-row smoke FIRST
#   sbatch slurm/med_quad_v2_span_doc.sbatch               # full, after the smoke passes
#
# Use the ENV-VAR PREFIX form above, NOT `sbatch --export=ALL,LUQ_SMOKE=1 …`. On DoC the explicit
# --export gets the job HELD with "user env retrieval failed" (reported from DoC, 2026-08-02). The
# prefix form works because sbatch already exports the submitting environment by default.
#
# This job writes RECORDS + FEATURES ONLY. No labels and no per-token cache: 01h_pertoken's sanity
# gate trains a probe on the label, so pertok cannot run until the judge has run, and the judge is an
# RCS step. Doing pertok here would cost three password-prompted syncs instead of one (and its ~6.4 GB
# exceeds DoC's free space anyway).

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

BUDGET=768
SMOKE="${LUQ_SMOKE:-0}"
LIMIT_ARG=""; REGIME="v2_med_quad_span"
if [ "$SMOKE" = "1" ]; then LIMIT_ARG="--limit 50"; REGIME="v2_med_quad_span_smoke"; fi

echo "med_quad span arm start $(hostname) $(date): budget=$BUDGET regime=$REGIME smoke=$SMOKE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --dataset med_quad --ood ID \
    --max-new-tokens "$BUDGET" --max-new-tokens-cap "$BUDGET" \
    --no-repeat-ngram-size 3 \
    --truncate-answer-span \
    --prompt-regime "$REGIME" $LIMIT_ARG \
    --dtype fp32 --attn eager

python -u scripts/checks/generation_quality.py --datasets med_quad --regime "$REGIME" \
    --budget "$BUDGET" --out "results/genqual_med_quad_${REGIME}.csv"

echo "DONE $(date). Read %empty, then p50/p95/p99, then %fabr/ansfrac, then %severe with run>=25 split."
