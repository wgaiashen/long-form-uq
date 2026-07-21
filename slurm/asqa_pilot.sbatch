#!/bin/bash
#SBATCH --job-name=asqa-pilot
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/asqa-pilot-%j.out
# ASQA closed-book PILOT (W8 gate) on DoC. Generates ~100 closed-book Llama-3.1-8B answers on ASQA and
# reports the Str-EM (coverage) label distribution -> GO (healthy spread, build ASQA) / NO-GO (degenerate,
# switch to WildHallucinations). Closed-book fp16 8B fits any DoC GPU (a40 chosen for availability; a100 is
# fine too and faster). Needs internet on the node for the din0s/asqa download (DoC salloc nodes have it).
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
echo "asqa-pilot start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u scripts/asqa_pilot.py --n 100 --out results/asqa/asqa_pilot.jsonl
echo "asqa-pilot DONE at $(date)"
