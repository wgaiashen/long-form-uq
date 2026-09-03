#!/bin/bash
#SBATCH --job-name=m11-bg
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/m11_bg-%j.out
# Background statistics for a CHUNK of layers, at all four generation budgets.
# Registered in prereg/M11_alllayer_published_distance_baselines.md, amendment A1.5.
#
# ALL THIRTY-TWO LAYERS ARE COMPUTED ON THIS MACHINE, so the background does not carry a per-layer
# accelerator difference of its own into the very quantity that is subtracted. The other cluster
# computes none of them and consumes what this produces.
#
# CHUNKED FOR MEMORY, not for correctness. One forward per background row yields every layer in the
# chunk, so a chunk costs one pass over the corpus however many layers it covers; but the states are
# held until the statistics are fitted, at roughly 2.3 GB per layer. Eight layers is about 18 GB.
# The chunk boundaries change no value.
#
#   for C in "0,1,2,3,4,5,6,7" "8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23" \
#            "24,25,26,27,28,29,30,32"; do
#     export LUQ_M11_LAYERS="$C"; sbatch slurm/m11_background_doc.sbatch
#   done
#
# Layer 15 is deliberately INCLUDED: it is the one layer whose statistics can be checked against a
# path that recomputes nothing, and that check is what licenses the other thirty-one.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

LAYERS="${LUQ_M11_LAYERS:?set LUQ_M11_LAYERS in the submitting shell, e.g. export LUQ_M11_LAYERS=0,1,2,3}"
echo "=== background statistics on $(hostname) at $(date) ==="
echo "layers: $LAYERS"
echo "commit: $(git rev-parse HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

python -u scripts/01q_background_stats.py --model meta-llama/Meta-Llama-3.1-8B --from-records \
    --layers "$LAYERS" --budgets 56,128,256,384 --window ref \
    --dtype fp32 --attn eager --device-map cuda \
    || { echo "!!! FAILED"; exit 3; }

echo; echo "=== done at $(date) ==="
