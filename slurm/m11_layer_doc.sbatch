#!/bin/bash
#SBATCH --job-name=m11-layer
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/m11_layer-%j.out
# One layer of the full-layer reproduction, end to end, on one machine.
# Registered in prereg/M11_alllayer_published_distance_baselines.md.
#
# WHY THIS IS ONE JOB AND NOT THREE. Per-token hidden states for one layer over the eight long-form
# datasets are about 19 GB, against a hard 50 GB storage cap here. Extracting every layer first and
# scanning afterwards would need about 400 GB. So each layer is extracted, reduced to a few megabytes
# of per-example distances, and deleted inside a single job. Nothing large ever accumulates.
#
# WHAT IT NEEDS ALREADY PRESENT
#   the model in HF_HOME
#   cache/records and the four namespaced record directories (about 150 MB in total)
#   cache/background_c4/<slug>__records__b384.npz (about 40 MB), the background token corpus
#
# THE LAYER COMES FROM THE ENVIRONMENT, NOT FROM --export. Passing --export=ALL,VAR=val leaves the
# job held rather than queued on this scheduler. Export the variable in the submitting shell:
#
#   export LUQ_M11_LAYER=16; sbatch slurm/m11_layer_doc.sbatch
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL="meta-llama/Meta-Llama-3.1-8B"
SLUG="meta-llama_Meta-Llama-3.1-8B"
export LUQ_REGIME="med_quad=cleanv2"
L="${LUQ_M11_LAYER:?set LUQ_M11_LAYER in the submitting shell, e.g. export LUQ_M11_LAYER=16}"

echo "=== layer $L on $(hostname) at $(date) ==="
echo "commit: $(git rev-parse HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# SPACE GATE. One layer needs about 19 GB and the cap here is 50 GB. Finding that out after six
# hours of extraction wastes the whole job, so it is checked before the model is loaded.
AVAIL_GB=$(df -BG --output=avail /vol/gpudata/gs925-msc_project | tail -1 | tr -dc '0-9')
echo "free space: ${AVAIL_GB} GB"
if [ "${AVAIL_GB:-0}" -lt 25 ]; then
  echo "!!! under 25 GB free. One layer needs about 19 GB. Clear space before resubmitting."
  exit 2
fi

# 1. BACKGROUND STATISTICS for this layer, at all four generation budgets. Cheap: one teacher-forced
#    forward per background row, and the output is four files of about 67 MB.
echo; echo "---- background statistics, layer $L, at $(date)"
python -u scripts/01q_background_stats.py --model "$MODEL" --from-records \
    --layers "$L" --budgets 56,128,256,384 --window ref \
    --dtype fp32 --attn eager --device-map cuda \
    || { echo "!!! background statistics FAILED"; exit 3; }

# 2. EXTRACT this layer for the eight long-form datasets. One forward per record yields the layer.
echo; echo "---- extraction, layer $L, at $(date)"
for DS in pubmed_qa xsum cnn_dailymail samsum med_quad asqa expertqa factscore; do
  case "$DS" in
    med_quad)  REGIME="cleanv2" ;;
    asqa)      REGIME="asqa_rp12" ;;
    expertqa)  REGIME="expertqa_rp12" ;;
    factscore) REGIME="factscore_rp12" ;;
    *)         REGIME="" ;;
  esac
  echo "  $DS (regime '${REGIME:-canonical}') at $(date)"
  python -u scripts/01p_pertoken_multi.py --model "$MODEL" --dataset "$DS" --ood ID \
      --layers "$L" --prompt-regime "$REGIME" --dtype fp32 --attn eager \
      --device-map cuda --skip-lm-head \
      || { echo "!!! extraction FAILED for $DS layer $L"; exit 4; }
done
df -h /vol/gpudata/gs925-msc_project | tail -1

# 3. SCAN: reduce the layer to per-example distances. --require-background makes a missing background
#    a failure rather than a quietly incomplete file.
echo; echo "---- scan, layer $L, at $(date)"
python -u scripts/checks/md_layer_scan.py --model "$MODEL" --layer "$L" --seeds 1,2,3 \
    --window ref --require-background \
    || { echo "!!! scan FAILED for layer $L"; exit 5; }

# 4. DELETE the states. They are regenerable and nobody needs to keep them. This runs only after the
#    scan has succeeded, so a failed scan leaves its inputs in place to be retried.
echo; echo "---- deleting the states for layer $L"
for D in cache cache/cleanv2 cache/asqa_rp12 cache/expertqa_rp12 cache/factscore_rp12; do
  rm -fv "$D"/pertok/"${SLUG}"__*__ID__L"${L}".npz 2>/dev/null
done
df -h /vol/gpudata/gs925-msc_project | tail -1

echo; echo "=== layer $L DONE at $(date) ==="
echo "scan files now in results/hybrids/mdscan_refwin__${SLUG}/L${L}__*.npz"
