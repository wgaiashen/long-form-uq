#!/bin/bash
#SBATCH --job-name=luq-xsum-b56
#SBATCH --partition=a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xsum_probe_base56-%j.out

# THE MISSING CONTROL ARM for the xsum half of slurm/xsum_cnn_probe_doc.sbatch.
#
# WHY IT IS NEEDED. The probe compares xsum at budget 96 against the v1 xsum cache at budget 56 and
# calls that a one-variable change. For cnn that holds: the v1 and probe records match on prompt,
# prompt_token_ids and target for 400/400 examples, and 100% of the uncapped rows come back
# byte-identical, exactly as greedy decoding requires.
#
# For xsum it does NOT hold. Checked, not assumed:
#   - prompts, prompt_token_ids and targets are identical for 0/400 examples
#   - only 1.7% of v1-uncapped rows reproduce their v1 length (cnn: 100.0%)
#   - v1 xsum prompt : "Here's the text and it's short one-sentence summary."
#     current xsum   : "Here's the text and its short summary."
# ProbeDrift's xsum template changed after the v1 cache was generated (and the installed package is
# ProbeDriftUpdate/, not the ProbeDrift/ checkout), and the drawn articles changed with it. So the
# probe's xsum arm varies budget AND prompt AND example set at once, and the pre-registered reading
# cannot be applied to it.
#
# WHY A FRESH BASELINE IS THE RIGHT FIX RATHER THAN RE-DOING THE PROBE. Any actual regeneration would
# use the CURRENT prompt, so the question that decides it is "at the current prompt, is xsum's capped
# tail truncation or over-generation?" The v1 cache cannot answer that at any budget -- it is on a
# prompt that will never be generated again. This run supplies the missing half: the SAME prompt and
# the SAME 400 examples as the probe, at the OLD budget of 56.
#
# Pairing then works by construction: both runs call data.load('xsum','ID') against the same installed
# ProbeDrift with the same seeds, so (split, idx) identifies the same article in both, and the only
# difference between the two caches is the budget. That restores the one-variable comparison the
# pre-registered reading assumes.
#
# Identical to the probe's xsum call in every respect except --max-new-tokens. Generation only, no
# judge call, no £.
#
#   sbatch slurm/xsum_probe_base56_doc.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "xsum base-56 control start $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/01_extract.py --dataset xsum --ood ID \
    --max-new-tokens 56 --max-new-tokens-cap 56 --limit 200 \
    --prompt-regime v2probe_xsum_base56 --dtype fp32 --attn eager

python -u scripts/checks/generation_quality.py --datasets xsum --regime v2probe_xsum_base56 \
    --budget 56 --out results/genqual_xsum_base56.csv

echo "xsum base-56 control DONE $(date). No judge spend."
