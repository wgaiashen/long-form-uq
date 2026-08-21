#!/bin/bash
#SBATCH --partition=a30
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_api
#SBATCH --output=logs/qwen_api_%j.out

# CPU-only: neither job touches a GPU, and requesting one had it queue behind GPU work.
# The two API jobs -- relabelling the truncated population and the step-4 judge-agreement test.
#
# WHY THESE RUN AS A BATCH JOB AND NOT ON THE SHELL VM. They were started on cloud-vm-40-244 and
# both were KILLED mid-run: that box terminates long processes. Both are resumable (02_label skips
# rows already stamped on the field; rejudge writes one JSON per dataset), so nothing was lost and
# nothing is re-paid for -- but they belong on a compute node. A GPU is requested only because every
# DoC partition is a GPU partition; neither job touches it.
#
# The key is SOURCED inside the job. Passing it via --export=ALL,VAR=val gets jobs held on this
# cluster ("user env retrieval failed"), which is a documented trap here.

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
source /vol/gpudata/gs925-msc_project/.openai_key

echo "=== API jobs on $(hostname) at $(date) ==="
python - <<'PY' || { echo "NO OUTBOUND INTERNET ON THIS NODE -- run from the login/jump box instead"; exit 3; }
import urllib.request
urllib.request.urlopen("https://api.openai.com/v1/models", timeout=15)
PY
echo "connectivity OK"

echo "########## 1. RELABEL the truncated population (resumes; skips stamped rows) ##########"
for SPEC in "Qwen/Qwen2.5-14B:med_quad:trunc_v1" "Qwen/Qwen2.5-14B:samsum:trunc_v1" \
            "Qwen/Qwen2.5-14B:cnn_dailymail:trunc_v1" "Qwen/Qwen2.5-14B:xsum:trunc_v1" \
            "Qwen/Qwen2.5-14B:expertqa:expertqa_rp12_trunc_v1" \
            "Qwen/Qwen2.5-14B:factscore:factscore_rp12_trunc_v1" \
            "meta-llama/Meta-Llama-3.1-8B:samsum:trunc_v1" \
            "meta-llama/Meta-Llama-3.1-8B:cnn_dailymail:trunc_v1" \
            "meta-llama/Meta-Llama-3.1-8B:xsum:trunc_v1"; do
  IFS=':' read -r M DS NS <<< "$SPEC"
  echo "---------- $M / $DS / $NS ----------"
  python -u scripts/02_label.py --model "$M" --dataset "$DS" --ood ID --prompt-regime "$NS" \
    || echo "!!! relabel FAILED for $M/$DS"
done

echo "########## 2. STEP 3 judge-agreement test (med_quad already written; the rest resume) ##########"
for DS in samsum expertqa factscore cnn_dailymail xsum; do
  python -u scripts/checks/rejudge_truncated.py --model Qwen/Qwen2.5-14B \
      --dataset $DS --n 80 --judge gpt-5-mini --also-gpt5 --cut-source template_restart \
    || echo "!!! rejudge FAILED for $DS"
done
echo "=== API jobs done at $(date) ==="
