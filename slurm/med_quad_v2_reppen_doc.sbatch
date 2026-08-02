#!/bin/bash
#SBATCH --job-name=luq-mq-rp
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/med_quad_v2_reppen-%j.out

# A.2b -- med_quad v2, THE SHIPPING ARM: budget 768 + no_repeat_ngram 3 + repetition-penalty 1.2.
#
# WHY THIS ARM AND NOT THE n-GRAM-ONLY ONE. The n-gram arm's median is 768 -- EXACTLY the cap. That is
# the same signature that made the original 128 budget indefensible; shipping it would swap
# "97.7% truncated at 128" for "heavily truncated at 768", a bigger number and the same defect. The
# n-gram arm is kept as the DIAGNOSTIC: it is the evidence that a 128 cap concealed most of the severe
# repetition, and that finding is stronger with this arm beside it as the contrast than it would be as
# the shipped dataset.
#
# ⚠️ SMOKE FIRST. Set LUQ_SMOKE=1 for a 50-row run into its own regime. The FIRST gate is %EMPTY <= 2%,
# not the length distribution: a repetition penalty applied over a LONG few-shot context penalises
# context words a real answer legitimately repeats and can force immediate EOS. That produced ~35%
# empty generations on med_quad historically. samsum's 0.0% empty under rep-pen does NOT transfer --
# measured prompts are samsum 147 tokens vs med_quad 1184, eight times longer, and this failure mode
# scales with context length.
#
# AFTER THE FULL RUN, report in this order:
#   1. %empty            (<= 2%, the hard gate)
#   2. p50 / p95 / p99 of generation length -- confirm 768 is NOT binding on the TAIL. The premise was
#      that 128 could not fit a 591-token reference, so 768 must comfortably fit it.
#   3. %severe, read against prereg/A2 -- and note that ~96-98% of med_quad's "severe" flags come from
#      the WHITESPACE rule (ws>=8), which med_quad's own gold references trip on 24.8% of rows. Report
#      the run>=25 (true looping) count separately; it was 0 old / 3 new on the n-gram arm.
#
#   sbatch slurm/med_quad_v2_reppen_doc.sbatch                    # full
#   LUQ_SMOKE=1 sbatch --export=ALL,LUQ_SMOKE=1 slurm/med_quad_v2_reppen_doc.sbatch   # 50-row smoke

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

BUDGET=768
SMOKE="${LUQ_SMOKE:-0}"
LIMIT_ARG=""; REGIME="v2_med_quad_rp12"
if [ "$SMOKE" = "1" ]; then LIMIT_ARG="--limit 50"; REGIME="v2_med_quad_rp12_smoke"; fi

echo "med_quad rep-pen arm start $(hostname) $(date): budget=$BUDGET regime=$REGIME smoke=$SMOKE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --dataset med_quad --ood ID \
    --max-new-tokens "$BUDGET" --max-new-tokens-cap "$BUDGET" \
    --no-repeat-ngram-size 3 --repetition-penalty 1.2 \
    --prompt-regime "$REGIME" $LIMIT_ARG \
    --dtype fp32 --attn eager

python -u scripts/checks/generation_quality.py --datasets med_quad --regime "$REGIME" \
    --budget "$BUDGET" --out "results/genqual_med_quad_${REGIME}.csv"

echo "DONE $(date). GATE ORDER: %empty first, then p50/p95/p99, then %severe."
