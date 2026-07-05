#!/bin/bash
#SBATCH --job-name=viz_xsum
#SBATCH --partition=a100          # need the 80GB card: fp32 8B + all-layer attn at seq 2048 ~66GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G                 # CPU-side stack of all-layer attentions (fp32) is ~34GB at 2048
#SBATCH --time=05:00:00
#SBATCH --output=logs/viz_xsum-%j.out

# xsum attention visualiser on the DoC cluster (RCS has NO 80GB GPU, so fp32 xsum OOMs there).
# Three steps, mirroring pbs/viz_attention.pbs but slurm + fp32/eager throughout to stay
# dtype-consistent with the sciq/trivia/pubmed runs:
#   01h_pertoken        per-token hidden states L15 (trains + applies the learned AttnPool).
#                       fp32 + eager, want_attentions=False, so eager only holds one layer's
#                       scores transiently -- the full ~5,600-tok source fits on the 80GB A100.
#   dump_viz_attention  train pooler + read per-token weights (CPU), then a small eager+fp32
#                       self-attention forward over the 120 most-interesting test examples,
#                       bounded to --max-seq 2048. Writes cache/viz/<key>__attn.npz.
#   visualise_attention CPU renderer; picks up the sidecar and writes the HTML.
set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
# Reclaim reserved-but-unallocated GPU memory (fragmentation guard on the long xsum forward).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B
DS=xsum
OOD=ID
LAYER=15

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo; echo "########## 1. 01h_pertoken (per-token L$LAYER) ##########"; date
python scripts/01h_pertoken.py --model "$MODEL" --dataset "$DS" --ood "$OOD" --layer "$LAYER" \
  --attn eager --dtype fp32 \
  || { echo "!!! 01h_pertoken FAILED"; exit 1; }

echo; echo "########## 2. dump_viz_attention ##########"; date
python scripts/tools/dump_viz_attention.py --model "$MODEL" --dataset "$DS" --ood "$OOD" \
  --layer "$LAYER" --n-self 120 --max-seq 2048 \
  || { echo "!!! dump_viz_attention FAILED"; exit 1; }

echo; echo "########## 3. visualise_attention (HTML) ##########"; date
python scripts/tools/visualise_attention.py --model "$MODEL" --dataset "$DS" --ood "$OOD" \
  --max-examples 60 \
  || echo "!!! visualise_attention FAILED"

echo; echo "=== ALL DONE $(date) ==="
