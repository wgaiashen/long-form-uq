#!/bin/bash
#SBATCH --job-name=viz_attn
#SBATCH --partition=a40           # 48GB is ample for sciq/trivia/pubmed (short/medium seqs).
#SBATCH --gres=gpu:1              # For xsum (long source) override: sbatch --partition=a100 ...
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/viz_attn-%j.out

# Attention visualiser chain (01h_pertoken -> dump_viz_attention -> visualise_attention) for
# one or more datasets, on DoC. Generalises slurm/viz_attention_xsum.sbatch. fp32 + eager
# throughout to stay dtype-consistent with the cached SAPLMA feature and across datasets.
#
#   sbatch slurm/viz_attention.sbatch                          # default: the 3 short/medium sets
#   DS="pubmed_qa" sbatch slurm/viz_attention.sbatch           # one dataset
#   DS="xsum" sbatch --partition=a100 slurm/viz_attention.sbatch   # long source needs the 80GB card
set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
OOD=ID
LAYER=15
DS="${DS:-sciq trivia_qa pubmed_qa}"   # space-separated; override at submit
NSELF="${NSELF:-120}"
MAXSEQ="${MAXSEQ:-2048}"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for D in $DS; do
  echo; echo "############### $D : 01h_pertoken (L$LAYER) ############### $(date)"
  python scripts/01h_pertoken.py --model "$MODEL" --dataset "$D" --ood "$OOD" \
    --layer "$LAYER" --attn eager --dtype fp32 \
    || { echo "!!! 01h_pertoken FAILED for $D"; continue; }

  echo; echo "############### $D : dump_viz_attention ############### $(date)"
  python scripts/tools/dump_viz_attention.py --model "$MODEL" --dataset "$D" --ood "$OOD" \
    --layer "$LAYER" --n-self "$NSELF" --max-seq "$MAXSEQ" \
    || { echo "!!! dump_viz_attention FAILED for $D"; continue; }

  echo; echo "############### $D : visualise_attention (HTML) ############### $(date)"
  python scripts/tools/visualise_attention.py --model "$MODEL" --dataset "$D" --ood "$OOD" \
    --max-examples 60 \
    || echo "!!! visualise_attention FAILED for $D"
done
echo; echo "=== ALL DONE $(date) ==="
