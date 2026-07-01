#!/bin/bash
#SBATCH --job-name=agg-table
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/aggregation_table-%j.out

# The consolidated aggregation table (scripts/checks/aggregation_table.py): the pooler ladder
# (mean-pool / last-token / per-sentence #5 / per-token #4 / uniform / attention variants) on the
# cached per-token L15 states, over a UNIFORM seed set, on EXPLICIT stamped labels, with a paired
# significance verdict per dataset and length-normalised weight diagnostics. Emits one CSV.
#
# GPU (small: no base model is loaded, only the tokenizer + cached states; the attention-family
# training over the temperature grid x seeds is what wants the GPU -- it is NOT cheap on the login
# node, which is also against cluster etiquette). Reads the per-token caches only, so it does not
# touch the running baseline jobs.
#
# sciq runs first, so a bug (cache gate, label, shapes) surfaces within minutes -- watch the log.

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "aggregation_table start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# -u = unbuffered so per-dataset progress is visible live. 5 seeds gives a stable seed-AVERAGED
# point estimate; significance does NOT come from the seeds (a paired t-test over near-identical
# re-inits has a ~0 denominator and calls any gap significant) but from a PAIRED BOOTSTRAP over the
# test set inside the driver (real data variance). HEADLINE verdict = attention vs mean-pool+MLP
# (the strong baseline a reader compares to), reported with a 95% CI. correctness (judge) runs first.
python -u scripts/checks/aggregation_table.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --datasets sciq,trivia_qa,pubmed_qa,xsum \
    --label-fields correctness,correctness_alignscore \
    --layer 15 --seeds 1,2,3,4,5

echo "aggregation_table DONE at $(date)"
