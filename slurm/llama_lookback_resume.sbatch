#!/bin/bash
#SBATCH --job-name=llama-lookback-r
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/llama_lookback_resume-%j.out

# RESUME of llama_lookback.sbatch: sciq + trivia_qa lookback features were saved before the
# first run died on the project quota; only pubmed_qa + xsum remain. Same Chuang-faithful
# feature (eager fp32), writes tiny (~15MB) files.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B
echo "llama_lookback RESUME start on $(hostname) at $(date)"
for DS in pubmed_qa xsum; do
  echo "================ $DS : 01c_lookback (eager fp32) ================ $(date)"
  python scripts/01c_lookback.py --dataset "$DS" --ood ID --model "$MODEL" \
    || echo "!!! lookback extract $DS FAILED"
done
echo "llama_lookback RESUME ALL DONE at $(date)"
