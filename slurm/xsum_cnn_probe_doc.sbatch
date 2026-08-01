#!/bin/bash
#SBATCH --job-name=luq-xc-probe
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/xsum_cnn_probe-%j.out

# THE xsum + cnn PROBE -- does their capped TAIL behave like samsum's, or not?
#
# THE QUESTION, AND WHY THE PILOT DOES NOT ANSWER IT. The samsum pilot showed the model writing to
# whatever cap it is given: move the cap 56 -> 96 and the median moves with it, 56 -> 96, while the gold
# reference stays near 35 words. But samsum's MEDIAN was AT its cap. xsum and cnn are a different shape:
#
#     dataset   budget   median   %capped     the cap binds...
#     med_quad     128      128      97.7     for the typical example
#     samsum        56       56      65.9     for the typical example
#     xsum          56       33      30.0     only for a ~30% TAIL
#     cnn          128       39      28.4     only for a ~28% TAIL
#
# So for xsum and cnn, ~70% of generations already stop on their own and only a tail runs into the cap.
# Whether that tail is over-generation (the model would ramble on forever) or genuinely-long answers
# (it needed the room) is UNTESTED, and samsum cannot settle it because its whole distribution is
# pinned at the cap.
#
# PRE-REGISTERED READING, written before the run:
#   - If the median stays roughly put and only the capped tail extends -> the tail is where the model
#     wanted more room. That is closer to real truncation and these two deserve a regeneration decision.
#   - If the median RISES toward the new cap the way samsum's did -> over-generation, same as samsum, and
#     all three close on evidence rather than on analogy.
#
# DESIGN: budget-only, v1 decoding (NO rep-pen), so this is a ONE-VARIABLE change and is directly
# comparable to the pilot's arm A. Generation only, no judge call, no £. n=200 per split.
#   xsum 56 -> 96   (same ~1.7x the pilot used)   cnn 128 -> 256
#
# WHY DoC. Both were generated on DoC in v1 (slurm/llama_xsum.sbatch, slurm/cnn_dailymail_doc.sbatch),
# so this matches their origin cluster and introduces no hardware change. It also avoids RCS's
# documented OOM on fp32 8B over these long sources (implementation_notes: RCS tops out at a 48GB card).
#
#   sbatch slurm/xsum_cnn_probe_doc.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "xsum/cnn probe start $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "--- xsum: 56 -> 96, v1 decoding, n=200/split ---"
python -u scripts/01_extract.py --dataset xsum --ood ID \
    --max-new-tokens 96 --max-new-tokens-cap 96 --limit 200 \
    --prompt-regime v2probe_xsum --dtype fp32 --attn eager

echo "--- cnn: 128 -> 256, v1 decoding, n=200/split ---"
python -u scripts/01_extract.py --dataset cnn_dailymail --ood ID \
    --max-new-tokens 256 --max-new-tokens-cap 256 --limit 200 \
    --prompt-regime v2probe_cnn --dtype fp32 --attn eager

echo "=== gates: v1 vs probe. READ THE MEDIAN, not the truncation rate. ==="
python -u scripts/checks/generation_quality.py --datasets xsum,cnn_dailymail \
    --out results/genqual_xsum_cnn_v1.csv
python -u scripts/checks/generation_quality.py --datasets xsum --regime v2probe_xsum \
    --budget 96  --out results/genqual_xsum_probe.csv
python -u scripts/checks/generation_quality.py --datasets cnn_dailymail --regime v2probe_cnn \
    --budget 256 --out results/genqual_cnn_probe.csv

echo "xsum/cnn probe DONE $(date). No judge spend."
