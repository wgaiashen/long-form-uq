#!/bin/bash
#SBATCH --job-name=aux_preflight
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aux_preflight-%j.out

# BLOCKING pre-flight for the B.1 auxiliary-loss re-run. Must pass before slurm/aux_attention_rerun.sbatch.
#
# Checks (a) the normalised auxiliary penalty against its closed form and against the real targets, and
# (b) that val_split_held_out_source really holds out a whole source dataset -- the single most important
# change in the re-run, and one that had never been functionally tested. If (b) is wrong every lambda is
# selected on the wrong criterion and the 24h run looks completely normal while being worthless.
#
# Also prints peak RSS, which is the sizing evidence for the ladder job (it loads the same caches).
#
#   sbatch slurm/aux_preflight.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== node ==="; hostname; date
echo "git: $(git log --oneline -1)"
echo "HF_HOME: $HF_HOME"

python scripts/checks/aux_preflight.py \
  --evals pubmed_qa,cnn_dailymail,xsum,factscore \
  --targets nll,content_mass \
  --seeds 1,2,3 --layer 15
rc=$?

echo; echo "=== preflight exit code: $rc  $(date) ==="
exit $rc
