#!/bin/bash
#SBATCH --job-name=asqa-full
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/asqa_full-%j.out

# ASQA FULL-SET Stage 1 (GPU): generate all 948 dev records CLOSED-BOOK with frozen Llama-3.1-8B and
# cache Tier-1 records + Tier-2 SAPLMA pooled features. ASQA is a second FACTUALITY eval target.
#   --dataset asqa --ood ID   -> the full dev split (948), eval-only (no train split).
#   --prompt-regime asqa_rp12 -> isolated namespace cache/asqa_rp12/ + results/ (records do NOT stamp
#                                the decoding params, so the namespace IS the provenance).
#   --repetition-penalty 1.2  -> the ExpertQA sweep-winning fix (rp1.2_only, NO no_repeat_ngram_size;
#                                the earlier 1.3+ngram induced word-salad). Holds base-Llama looping down
#                                (pilot showed 10% repeat at 200 tok with plain greedy).
#   --dtype fp32 --attn eager -> matches the other Llama datasets for feature comparability. fp32 8B +
#                                hidden states over a 256-token generation -> A100-80GB on DoC (safe).
#   --max-new-tokens-cap 256  -> asqa budget (gold p90=252); default 128 would clip it, so REQUIRED.
#   NO --truncate-long: ASQA is long-form; newlines are legitimate.
# The judge (login/VM API, gpt-5-mini) and per-token L15 are SEPARATE follow-up steps.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "asqa_full start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "---- preflight: imports + cuda + asqa dataset loads (isolates a bad node) ----"
timeout 300 python -u -c "import torch, transformers; import sys; sys.path.insert(0,'src'); from luq import data; tr,ev=data.load('asqa','ID'); print('imports OK | cuda', torch.cuda.is_available(), '| asqa eval', len(ev.x), flush=True)" \
  || { echo "PREFLIGHT FAILED (node/fs/dataset problem) at $(date)"; exit 1; }

COMMON="--model meta-llama/Meta-Llama-3.1-8B --dataset asqa --ood ID --prompt-regime asqa_rp12 --max-new-tokens-cap 256 --dtype fp32 --attn eager --repetition-penalty 1.2"
echo "---- smoke (--limit 5) ----"
python -u scripts/01_extract.py $COMMON --limit 5
echo "---- full run (resumes from smoke checkpoint) ----"
python -u scripts/01_extract.py $COMMON
echo "asqa_full DONE at $(date)"
echo "NEXT (login/VM): source .openai_key; python scripts/02_label.py --dataset asqa --ood ID --prompt-regime asqa_rp12 --judge gpt-5-mini"
