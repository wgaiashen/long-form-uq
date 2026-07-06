#!/bin/bash
#SBATCH --job-name=eqa-full
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/expertqa_full-%j.out

# ExpertQA FULL-SET Stage 1 (GPU): generate all 2016 factual-core records with frozen
# Llama-3.1-8B + cache Tier-1 records and Tier-2 SAPLMA pooled features.
#   --ood ID            -> the whole usable pool (2016), not the 200-example pilot.
#   --prompt-regime expertqa_reppen -> isolated namespace cache/expertqa_reppen/ + results/;
#                          signals these generations used the repetition fix (records do NOT
#                          stamp the decoding params, so the namespace is the provenance).
#   --repetition-penalty 1.3 --no-repeat-ngram-size 3 -> the decoding fix that took the
#                          re-pilot from 48.5% -> 0% repetition-degenerate (worklog 2026-07-05).
#   --dtype fp32 --attn eager -> matches the other Llama datasets for feature comparability.
#                          This is why the job runs on DoC's A100-80GB: fp32 8B + hidden-states
#                          over a 384-token generation OOMs RCS's 48GB L40S/A40 cards.
#   --max-new-tokens-cap 384 -> Stage-0 locked budget (REQUIRED; default 128 would clip it).
#   NO --truncate-long: ExpertQA is long-form; newlines are legitimate.
# The judge (login-node API) and per-token L15 (needs labels first) are SEPARATE follow-up steps.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1   # Llama-3.1-8B is fully cached; avoid the online xet fetch
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "expertqa_full start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "---- preflight: imports + cuda (isolates a bad node) ----"
timeout 300 python -u -c "import torch, transformers; print('imports OK | cuda', torch.cuda.is_available(), flush=True)" \
  || { echo "PREFLIGHT FAILED (node/fs problem) at $(date)"; exit 1; }

COMMON="--model meta-llama/Meta-Llama-3.1-8B --dataset expertqa --ood ID --prompt-regime expertqa_reppen --max-new-tokens-cap 384 --dtype fp32 --attn eager --repetition-penalty 1.3 --no-repeat-ngram-size 3"
echo "---- smoke (--limit 5) ----"
python -u scripts/01_extract.py $COMMON --limit 5
echo "---- full run (resumes from smoke checkpoint) ----"
python -u scripts/01_extract.py $COMMON
echo "expertqa_full DONE at $(date)"
echo "NEXT (login node): scripts/02_label.py --dataset expertqa --ood ID --prompt-regime expertqa_reppen --judge gpt-5-mini"
