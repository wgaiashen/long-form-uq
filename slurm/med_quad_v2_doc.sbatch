#!/bin/bash
#SBATCH --job-name=luq-mq-v2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/med_quad_v2-%j.out

# A.2 -- med_quad v2 regeneration on DoC. GENERATION ONLY: no judge call, the gates are read first.
#
# WHY med_quad REGENERATES AT ALL, when the samsum pilot said the others should not. med_quad is the
# only one of the four with a genuine budget defect: its budget is 128 against a gold reference p90 of
# 591, i.e. 0.22x, where samsum/xsum/cnn sit at 1.25-1.7x. The pilot decided the OTHER three; med_quad
# was never in that question.
#
# WHY DoC IS FINE HERE. v1 med_quad was RCS-generated, so in principle a DoC v2 mixes a hardware change
# into the v1-vs-v2 comparison. Two things make that acceptable: (1) v1 med_quad is 97.7% truncated, so
# there is barely a "v1 quality" to compare against and the real questions are v2-internal -- does
# uncapping produce coherent long answers; and (2) the v1 grid ALREADY mixes clusters, since xsum and
# cnn were both generated on DoC (slurm/llama_xsum.sbatch, slurm/cnn_dailymail_doc.sbatch). This adds
# no new class of problem.
#
# CONFIG, pre-registered in prereg/med_quad_degeneracy_gate.md:
#   budget 128 -> 768 (gold p90 591 x ~1.3); no_repeat_ngram_size 3; NO repetition penalty; fp32+eager.
#
# EXPECTED: DEGENERACY MAY RISE, and that is not a failure. At 97.7% capped, a cap of 128 truncates
# loops before the detector's run-length threshold can see them, so med_quad's 12.3% severe is a LOWER
# BOUND. A rise is the expected consequence of removing a loop-truncator; the response is to keep BOTH
# the budget and the anti-loop measure. med_quad ONLY -- at <=0.7% severe nothing is masked for the
# others and a rise there IS a failure.
#
# READ THE LENGTH DISTRIBUTION FIRST, NOT THE TRUNCATION RATE. The pilot showed the model writes to
# whatever cap it is given (samsum's median moved 56 -> 96 when the cap did). If med_quad's median lands
# at or near 768, the reading is that it needs BOTH the larger budget AND general drift control, and
# pbs/med_quad_v2_reppen.pbs (rep-pen 1.2, smoke-tests 50 rows first) is the prepared follow-up.
#
#   sbatch slurm/med_quad_v2_doc.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

BUDGET=768
echo "med_quad v2 start $(hostname) $(date): budget $BUDGET, ngram 3, NO rep-pen"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --dataset med_quad --ood ID \
    --max-new-tokens "$BUDGET" --max-new-tokens-cap "$BUDGET" \
    --no-repeat-ngram-size 3 \
    --prompt-regime v2_med_quad \
    --dtype fp32 --attn eager

python -u scripts/checks/generation_quality.py --datasets med_quad --regime v2_med_quad \
    --budget "$BUDGET" --out results/genqual_med_quad_v2.csv

echo "med_quad v2 DONE $(date). NO judge spend. Read the LENGTH DISTRIBUTION first."
