#!/bin/bash
#SBATCH --job-name=s6_multihead
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=logs/multihead_ladder-%j.out

# S6 — multi-head (design note 2), SELF-CONTAINED: MH (K queries+heads) + ABLATION (1 query, K heads),
# both trained in-run at T=1. --defer-baseline leaves the K=1 columns EMPTY + baseline_deferred=True;
# the MH-vs-K=1 reference is filled POST-HOC by join_arma_baseline.py from S3's arm A. So S6 needs NO
# dependency on S3 and runs NOW on a free A100.
# factscore is EXCLUDED from S6 by design (see worklog 2026-07-28): S3's ~77 min/cell rate over 21
# cells vs 24h makes its OOD cells unlikely to exist in time, and it is the thinnest eval (n=455, base
# rate 0.386) with an rp12 decoding confound. -> 12 paired cells (6 evals x 2 OOD rungs) + 6 ID.
set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== node ==="; hostname; date; echo "commit $(git rev-parse HEAD)"

# NOTE: the driver has no --baseline-csv/K=1@T=1 config; per STEP 4 fallback we run 2 configs (MH,
# ABLATION) and label the MH-vs-K=1 temperature asymmetry. Head-diversity -> CSV (query_cosine/attn_corr).
python -u scripts/checks/multihead_ladder.py \
  --evals pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa \
  --rungs DiffTask,LOO \
  --K 4 --seeds 1,2,3 --defer-baseline

echo "=== ALL DONE $(date) ==="
