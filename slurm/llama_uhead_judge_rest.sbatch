#!/bin/bash
#SBATCH --job-name=uhead-judge-rest
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/llama_uhead_judge_rest-%j.out

# Finish the judge-trained uhead run: only trivia_qa + xsum (pubmed + sciq already completed and
# verified before the filesystem-inode crash killed the original job mid-trivia). Writes
# uhead_judge / uhead_judge_v2 (the frozen AlignScore-trained uhead / uhead_v2 are preserved).
# python -u so a hard failure can't be swallowed; xsum is the heavy one (long articles -> all-layer
# attentions), a100 128G handles it.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
echo "uhead_judge_rest start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "---- preflight ----"
timeout 300 python -u -c "import torch, transformers; print('imports OK | cuda', torch.cuda.is_available(), flush=True)" \
  || { echo "PREFLIGHT FAILED (bad node) at $(date)"; exit 1; }

for DS in trivia_qa xsum; do
  echo "================ $DS : uhead_judge v1 (JUDGE label) ================ $(date)"
  python -u scripts/01j_uhead_joe.py --dataset "$DS" --ood ID --model "$MODEL" \
      --variant v1 --name uhead_judge --label-field correctness || echo "!!! v1 $DS FAILED"
  echo "================ $DS : uhead_judge v2 (JUDGE label) ================ $(date)"
  python -u scripts/01j_uhead_joe.py --dataset "$DS" --ood ID --model "$MODEL" \
      --variant v2 --name uhead_judge_v2 --label-field correctness || echo "!!! v2 $DS FAILED"
done
echo "uhead_judge_rest ALL DONE at $(date)"
