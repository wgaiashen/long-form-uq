#!/bin/bash
#SBATCH --job-name=eqa-pilot
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/expertqa_pilot-%j.out

# ExpertQA Stage 1 (GPU): generate the 200-example pilot with frozen Llama-3.1-8B + cache the
# Tier-1 records and Tier-2 SAPLMA pooled features, into the isolated `expertqa` namespace.
# fp32 + eager to match the other Llama runs (feature comparability for the later full run).
# max-new-tokens-cap 384 is REQUIRED (else the default-128 ceiling clips the 384 budget).
# NO --truncate-long: ExpertQA is long-form; newlines are legitimate.
# The judge (login-node API) and per-token L15 (needs labels first) are SEPARATE follow-up steps.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1   # Llama-3.1-8B is fully cached; avoid the online xet fetch that hit the quota
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "expertqa_pilot start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "---- preflight: imports + cuda (isolates a bad node) ----"
timeout 300 python -u -c "import torch, transformers; print('imports OK | cuda', torch.cuda.is_available(), flush=True)" \
  || { echo "PREFLIGHT FAILED (node/fs problem) at $(date)"; exit 1; }

# bf16 (not fp32): fp32 8B (32GB) + output_hidden_states over a 384-token generation OOM'd the a40.
# bf16 halves the weights (16GB) with comfortable headroom. Trade-off: the PILOT SAPLMA features are
# bf16-derived (fine for the go/no-go, which turns on the JUDGE LABELS + generation quality, not the
# probe); if the pilot passes we regenerate fp32 for the full run to match the other Llama datasets.
# python -u so a hard failure/OOM can't be swallowed by buffered stdout. --limit 5 smoke first (fast
# failure); the full run then resumes from the checkpoint.
COMMON="--model meta-llama/Meta-Llama-3.1-8B --dataset expertqa --ood pilot --prompt-regime expertqa --max-new-tokens-cap 384 --dtype fp32 --attn eager"
echo "---- smoke (--limit 5) ----"
python -u scripts/01_extract.py $COMMON --limit 5
echo "---- full run (resumes from smoke) ----"
python -u scripts/01_extract.py $COMMON
echo "expertqa_pilot DONE at $(date)"
echo "NEXT (login node): scripts/02_label.py --dataset expertqa --ood pilot --prompt-regime expertqa --judge gpt-5-mini"
