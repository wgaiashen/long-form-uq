#!/bin/bash
#SBATCH --job-name=keystone_pdnew
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/keystone_pdnew-%j.out

# M2 -- re-validate the keystone on the UPDATED ProbeDrift prompts, as a CHECK not a replacement.
# Regenerates sciq / trivia_qa / pubmed_qa under the new prompts in the SEPARATE "pdnew" namespace
# (cache/pdnew), labels them with the FREE yardsticks (string-match for short-form, AlignScore for
# all three -- no API, no judge confound), trains SAPLMA at L15, and prints the new-prompt PRR beside
# the frozen old-prompt PRR on the SAME yardstick so the delta is the prompt change alone.
#
# PREREQUISITE (consequential, do once, NOT in this job): install the updated ProbeDrift into the env
#   pip install -e /vol/gpudata/gs925-msc_project/ProbeDriftUpdate
# The frozen old-prompt cache stays valid (03/04 read cached features), so replacing the package does
# not disturb the reproduction, we just never regenerate old-prompt data again.
#
#   sbatch slurm/keystone_newprompts.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
REG=pdnew
SHORT="sciq trivia_qa"
ALL="sciq trivia_qa pubmed_qa"

echo "=== node ==="; hostname; date
python -c "import probe_drift, inspect; print('probe_drift from', probe_drift.__file__)"

echo; echo "########## 1. Generate on NEW prompts (keystone config: fp32 + eager) ##########"
for d in $ALL; do
  EXTRA=""; [ "$d" = "pubmed_qa" ] && EXTRA="--truncate-long"   # pubmed D1, matches the keystone
  echo "---- extract $d ----"
  python scripts/01_extract.py --model "$MODEL" --dataset "$d" --ood ID \
    --dtype fp32 --attn eager --prompt-regime "$REG" $EXTRA
done

echo; echo "########## 2. FREE labels: string-match (short) + AlignScore (all) ##########"
for d in $SHORT; do
  python scripts/02_label.py --model "$MODEL" --dataset "$d" --ood ID --prompt-regime "$REG"
done
for d in $ALL; do
  python scripts/01f_alignscore.py --model "$MODEL" --dataset "$d" --ood ID --prompt-regime "$REG"
done

echo; echo "########## 3. SAPLMA L15 PRR delta, NEW vs OLD (in-process; frozen scores untouched) ##########"
# keystone_delta trains the probe in memory and prints OLD-vs-NEW per yardstick, writing nothing,
# so the frozen old-prompt cache/scores stay exactly as they are.
python scripts/checks/keystone_delta.py --model "$MODEL" --layer 15 --new-regime "$REG"

echo; echo "=== ALL DONE $(date) ==="
