#!/bin/bash
#SBATCH --job-name=ptrue-wording
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ptrue_wording-%j.out

# P(True) wording change, gated end-to-end:
#   1. validate the NEW wording ("Is the above response accurate?") -- does yes/no still
#      dominate the verdict next-token mass (old wording ~0.96)?
#   2. ONLY if the min mass across datasets >= GATE, re-extract ptrue features under the
#      name 'ptrue_accurate' (the old 'ptrue' true-wording features are left untouched).
# Gemma-2 needs bf16 + eager attention; generate.load_model handles that.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=google/gemma-2-9b-it
GATE=0.80
DSETS="sciq pubmed_qa"   # xsum added once its labels finish

echo "ptrue wording job on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

min_mass=1.0
for DS in $DSETS; do
  echo "================ verdict check: $DS ================"
  tmp="logs/_verdict_${DS}.txt"
  python scripts/checks/check_ptrue_verdict.py --dataset "$DS" --ood ID --model "$MODEL" | tee "$tmp"
  m=$(grep -i "mean yes/no probability mass" "$tmp" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  echo "  -> $DS mean yes/no mass = ${m:-NA}"
  # track the minimum mass (awk for float compare)
  if [ -n "${m:-}" ]; then
    min_mass=$(awk -v a="$min_mass" -v b="$m" 'BEGIN{print (b<a)?b:a}')
  else
    min_mass=0.0
  fi
done

echo "================ GATE: min mass=$min_mass vs threshold=$GATE ================"
pass=$(awk -v a="$min_mass" -v b="$GATE" 'BEGIN{print (a>=b)?1:0}')
if [ "$pass" != "1" ]; then
  echo "GATE FAILED -- new wording's yes/no mass too low; NOT re-extracting. Inspect above."
  echo "PTRUE WORDING JOB DONE (gate failed) at $(date)"
  exit 0
fi

echo "GATE PASSED -- re-extracting ptrue features under name 'ptrue_accurate'"
for DS in $DSETS; do
  echo "================ re-extract (ptrue_accurate): $DS ================ $(date)"
  python scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" --name ptrue_accurate \
    || echo "!!! ptrue_accurate extract $DS FAILED"
done
echo "PTRUE WORDING JOB DONE (gate passed, features extracted) at $(date)"
