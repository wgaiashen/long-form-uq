#!/bin/bash
#SBATCH --job-name=eqa-regen
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/expertqa_regen-%j.out

# ExpertQA FULL-SET REGENERATION with the sweep-winning decoding config (job 257204):
#   rp1.2_only = repetition_penalty=1.2, NO no_repeat_ngram_size.
# The earlier reppen set used repetition_penalty=1.3 + no_repeat_ngram_size=3, which INDUCED
# word-salad (~26% degraded). The sweep showed dropping no_repeat_ngram + a gentler 1.2 penalty
# fixes 27/35 hard cases and stays on-topic (relevance-checked). It does NOT fully clear the
# precommitted SEVERE<=2% bar (~5% residual on the hardest open-ended questions) -> the residual
# is distrust-labelled downstream (degenerate = untrustworthy), not dropped. New namespace
# expertqa_rp12 (records don't stamp decoding params, so the namespace IS the provenance).

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "expertqa_regen start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
timeout 300 python -u -c "import torch, transformers; print('imports OK | cuda', torch.cuda.is_available(), flush=True)" \
  || { echo "PREFLIGHT FAILED at $(date)"; exit 1; }

# rp1.2_only: --repetition-penalty 1.2 and NO --no-repeat-ngram-size (omitted => HF default 0 = off).
COMMON="--model meta-llama/Meta-Llama-3.1-8B --dataset expertqa --ood ID --prompt-regime expertqa_rp12 --max-new-tokens-cap 384 --dtype fp32 --attn eager --repetition-penalty 1.2"
echo "---- smoke (--limit 5) ----"
python -u scripts/01_extract.py $COMMON --limit 5
echo "---- full run (resumes from smoke) ----"
python -u scripts/01_extract.py $COMMON
echo "expertqa_regen DONE at $(date)"
