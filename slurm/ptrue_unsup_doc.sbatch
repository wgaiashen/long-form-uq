#!/bin/bash
#SBATCH --job-name=luq-ptunsup
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ptrue_unsup-%j.out
#
# UNSUPERVISED P(True) for ONE dataset. The original Kadavath P(True): append the verification
# question, read the model's OWN yes/no next-token probability, renormalise. No training, no labels --
# an unsupervised baseline like MSP, and the CONTROL for our supervised P(True) probe (01b), which
# reads the hidden state at the very same slot. the project framing notes calls that distinction load-bearing and we
# have never shown it.
#
#   DS=cnn_dailymail sbatch slurm/ptrue_unsup_doc.sbatch
#   DS=asqa REG=asqa_rp12 sbatch slurm/ptrue_unsup_doc.sbatch
#
# USE THE CALLER-ENV FORM ABOVE, NOT `--export=ALL,DS=...` (corrected 2026-08-04 after running all
# six on DoC). On this cluster the `--export=ALL,VAR=` form gets jobs HELD with "user env retrieval
# failed" — the same failure we hit with LUQ_SMOKE. The comment previously documented that form, so
# anyone following it got six held jobs and no error explaining why. Setting the variable in the
# caller's environment and letting sbatch inherit it works and was what actually ran.
#
# ONE DATASET PER JOB, deliberately. The baseline-feature job (269813) finished with 25 SECONDS of
# its 8h wall left because it did six datasets in one go. 01g IS resumable (it skips records already
# scored, unlike 01b/01c), but a per-dataset job means a walltime overrun costs one dataset, not six.
#
# a40 / 48G, NOT the a100 the lookback job needed. Verified in src/luq/generate.py: load_model
# auto-selects fp16 for anything that is not Gemma, so this is a ~16GB model, and 01g needs only the
# next-token logits -- no output_attentions, no all-layer hidden states. Lookback needed the a100
# because it forces fp32 AND materialises 32x32x2048^2 attention (17.2GB); none of that applies here.
#
# IT WRITES INTO THE TIER-1 RECORDS, so the results DO NOT go back as record files. The export step
# below emits a small SIDECAR CSV instead; RCS merges that into its OWN canonical records. Shipping
# records back would overwrite the files carrying the £-paid judge labels, the ones hashed in
# scripts/checks/CANONICAL_v1_MANIFEST.md, and rsync guards against deletion, not against overwriting.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export HF_HUB_OFFLINE=1
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
mkdir -p logs

DS="${DS:?set DS=<dataset>}"
REG="${REG:-}"                       # namespace tag: asqa_rp12 / expertqa_rp12 / factscore_rp12
MODEL="meta-llama/Meta-Llama-3.1-8B"

echo "=== unsupervised P(True): $DS ${REG:+(regime $REG)} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no GPU visible)"

python -u scripts/01g_ptrue_unsup.py --model "$MODEL" --dataset "$DS" --ood ID \
    ${REG:+--prompt-regime "$REG"}
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "!!! 01g FAILED for $DS (exit $STATUS) — NOT exporting a sidecar from a partial run"
  exit $STATUS
fi

# Export the sidecar. Runs ONLY on success, so a partial run cannot produce a file that looks complete.
echo "=== export sidecar $(date) ==="
python -u scripts/checks/ptrue_unsup_sidecar.py --export --datasets "$DS"

echo "=== SIZE + SHA256 for the sync back ==="
F="results/sidecar_ptrue_unsup/ptrue_unsup__${DS}.csv"
[ -e "$F" ] && printf "%12s  %s  %s\n" "$(stat -c%s "$F")" "$(sha256sum "$F" | cut -d' ' -f1)" "$F" \
            || echo "MISSING  $F"
echo "=== DONE $(date) ==="
