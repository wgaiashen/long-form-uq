#!/bin/bash
#SBATCH --job-name=overnight-gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/overnight_gpu-%j.out

# All GPU-heavy AlignScore + pubmed-D1 work in ONE allocation (submitted by overnight.sh via
# sbatch --wait). pubmed: truncate to Joe's newline-stop, re-pool with corrected pooling,
# AlignScore-RE-SCORE the truncated text. Then AlignScore-label sciq/trivia/xsum (free). The
# env vars point torch.hub / nltk at the big volume (ckpt + punkt_tab pre-cached there).
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export TORCH_HOME=/vol/gpudata/gs925-msc_project/hf_cache/torch
export NLTK_DATA=/vol/gpudata/gs925-msc_project/hf_cache/nltk_data
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
M=meta-llama/Meta-Llama-3.1-8B; OOD=ID
echo "overnight_gpu start $(hostname) $(date)"

# pubmed D1 (truncate+clear labels) -> re-pool truncated (corrected pooling) -> AlignScore-label truncated
python scripts/checks/make_pubmed_d1.py --model $M --dataset pubmed_qa --ood $OOD || echo "!!! FAIL make_pubmed_d1"
python scripts/01e_repool.py --model $M --dataset pubmed_qa --ood $OOD --dtype fp32 --attn eager --cache-per-token || echo "!!! FAIL repool pubmed"
python scripts/01f_alignscore.py --model $M --dataset pubmed_qa --ood $OOD || echo "!!! FAIL alignscore pubmed"

# AlignScore-label the rest (free); xsum only if its extraction has finished
for DS in sciq trivia_qa xsum; do
  if [ -f "cache/records/meta-llama_Meta-Llama-3.1-8B__${DS}__${OOD}.jsonl" ]; then
    echo "==== alignscore-label $DS $(date) ===="
    python scripts/01f_alignscore.py --model $M --dataset "$DS" --ood $OOD || echo "!!! FAIL alignscore $DS"
  else
    echo "skip alignscore $DS (no records yet)"
  fi
done
echo "overnight_gpu DONE $(date)"
