#!/bin/bash
#SBATCH --partition=a100 --gres=gpu:1 --cpus-per-task=8 --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=luq_mq_v2
#SBATCH --output=luq_mq_v2_%j.out

# A.2 -- med_quad v2 regeneration. GENERATION ONLY, no judge call: the gates are read first.
#
# ⚠️ WHY THIS DOES NOT WAIT FOR THE samsum PILOT. The pilot decides whether the OTHER THREE regenerate,
# by testing whether their capping is truncation or over-generation. med_quad is not in that question:
# its budget is 128 against a gold reference p90 of 591, i.e. 0.22x, versus 1.25-1.7x for the others.
# It is the one dataset with a genuine budget defect, so it regenerates either way. Waiting would
# serialise two independent decisions.
#
# WHY RCS RATHER THAN DoC. v1 med_quad was RCS-generated (this file's sibling extract_neighbour.pbs has
# med_quad as its DEFAULT dataset). Most of med_quad's gates are v2-internal and would not care, but the
# pre-registered "med_quad should improve" prediction IS a v1-vs-v2 judge comparison and would inherit a
# hardware difference if v2 came from DoC. RCS avoids that entirely.
#
# CONFIG, confirmed and pre-registered in prereg/A2_med_quad_degeneracy_gate.md:
#   budget 128 -> 768   (gold p90 591 x ~1.3)
#   no_repeat_ngram_size = 3   -- what v1 med_quad actually used
#   NO repetition penalty      -- documented as producing ~35% EMPTY generations on med_quad, whose
#                                 few-shot prompt is the longest of the four (~1184 tokens), so the
#                                 penalty falls on legitimately repeated context words and forces EOS.
#   fp32 + eager, matching v1, so the pooling-consistency guard can pass at TOL=1e-4 later.
#
# ⚠️ EXPECTED GATE BEHAVIOUR, pre-registered: med_quad's DEGENERACY MAY RISE. At 97.7% capped, a cap at
# 128 truncates loops before they are long enough for the detector to see, so the 12.3% severe rate is a
# LOWER BOUND. A rise is the expected consequence of removing a loop-truncator, NOT evidence the fix
# failed; the response is to keep BOTH the larger budget and the anti-loop measure. This exemption is
# med_quad-only -- at <=0.7% severe there is nothing masked for samsum/xsum/cnn and a rise there IS a
# failure signal.
#
#   qsub pbs/med_quad_v2.pbs

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source pbs/_env.sh
luq_activate || exit 1
cd "$LUQ_REPO"
export HF_HOME
export HF_HUB_OFFLINE=1

BUDGET=768

echo "=== med_quad v2: budget $BUDGET, no_repeat_ngram 3, NO rep-pen, on $(hostname) $(date) ===" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --dataset med_quad --ood ID \
    --max-new-tokens "$BUDGET" --max-new-tokens-cap "$BUDGET" \
    --no-repeat-ngram-size 3 \
    --prompt-regime v2_med_quad \
    --dtype fp32 --attn eager

echo "=== generation-side gates: v1 vs v2 ===" >&2
python -u scripts/checks/generation_quality.py --datasets med_quad \
    --out results/genqual_med_quad_v1.csv
python -u scripts/checks/generation_quality.py --datasets med_quad --regime v2_med_quad \
    --budget "$BUDGET" --out results/genqual_med_quad_v2.csv

echo "=== med_quad v2 done $(date). NO judge spend yet. Degeneracy MAY rise -- see the prereg. ===" >&2
