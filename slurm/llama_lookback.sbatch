#!/bin/bash
#SBATCH --job-name=llama-lookback
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/llama_lookback-%j.out

# Lookback Lens features on Llama-3.1-8B (the attention lookback ratio, Chuang et al.).
# GPU-only: 01c replays the cached token ids with the EAGER attention backend in fp32
# (eager fp16 overflows to NaN in the softmax) and writes cache/features/...__lookback.npz.
# Our features/lookback.py is verified ~1e-6 against Chuang's released code; the luh
# lookback orients the ratio the other way (1 - ours), which is PRR-identical under the
# linear probe. Independent of the ptrue/uhead jobs (different feature file), so run in parallel.
#
# After this finishes, on the LOGIN NODE (CPU, no GPU) score it -- lookback is a single
# combined layer so use --layer 0:
#   for DS in sciq trivia_qa pubmed_qa xsum; do
#     python scripts/03_probe.py --model meta-llama/Meta-Llama-3.1-8B --dataset $DS --ood ID \
#         --method lookback --layer 0
#     python scripts/04_eval.py  --model meta-llama/Meta-Llama-3.1-8B --dataset $DS --ood ID
#   done

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
echo "llama_lookback start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# sciq/trivia (short, fast) first, then the long-form summarisation/QA sets.
for DS in sciq trivia_qa pubmed_qa xsum; do
  echo "================ $DS : 01c_lookback (eager fp32) ================ $(date)"
  python scripts/01c_lookback.py --dataset "$DS" --ood ID --model "$MODEL" \
    || echo "!!! lookback extract $DS FAILED"
done
echo "llama_lookback ALL DONE at $(date)"
