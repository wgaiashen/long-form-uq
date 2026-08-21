#!/bin/bash
#SBATCH --job-name=llama-ptrue-acc
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/llama_ptrue_acc-%j.out

# SUPERVISED P(accurate) features on Llama-3.1-8B: the verdict-position hidden state after
# appending "Is the above response accurate?" (the task-agnostic wording). 01b replays the
# cached token ids + the suffix and reads the last-position state at every layer. This is the
# SUPERVISED probe input; the UNSUPERVISED P(accurate) (the model's own yes/no probability) is
# already in the Llama records as ptrue_unsup and is surfaced by 04_eval automatically.
#
# Uses the default forward (fp16/SDPA) to match how the unsupervised ptrue and the Gemma ptrue
# features were extracted -- the ptrue family's own convention, evaluated on its own PRR.
# Independent of the lookback job, so run in parallel.
#
# After this finishes, on the LOGIN NODE (CPU) score it at the middle layer 15:
#   for DS in sciq trivia_qa pubmed_qa xsum; do
#     python scripts/03_probe.py --model meta-llama/Meta-Llama-3.1-8B --dataset $DS --ood ID \
#         --method ptrue_accurate --layer 15
#     python scripts/04_eval.py  --model meta-llama/Meta-Llama-3.1-8B --dataset $DS --ood ID
#   done

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
echo "llama_ptrue_accurate start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in sciq trivia_qa pubmed_qa xsum; do
  echo "================ $DS : 01b_ptrue --name ptrue_accurate ================ $(date)"
  python scripts/01b_ptrue.py --dataset "$DS" --ood ID --model "$MODEL" --name ptrue_accurate \
    || echo "!!! ptrue_accurate extract $DS FAILED"
done
echo "llama_ptrue_accurate ALL DONE at $(date)"
