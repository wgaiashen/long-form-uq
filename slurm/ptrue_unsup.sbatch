#!/bin/bash
#SBATCH --job-name=ptrue-unsup
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ptrue_unsup-%j.out

# One GPU allocation, two jobs:
#   1. alignscore_vs_authors.py  -- numerical check that our AlignScore matches Joe's
#      literal scorer (<1e-3). Needs a GPU because Joe's scorer calls torch.cuda
#      unconditionally. Quick (~20 records, small roberta model).
#   2. 01g_ptrue_unsup.py  -- unsupervised P(True) for the four datasets: append the
#      verification question, read the model's own yes/no probability at the verdict
#      slot. One forward pass per example; writes ptrue_unsup into the records.
# Then a read-only PRR summary of the new score.
#
# Llama-3.1-8B loads fp16/sdpa by default (16GB), so a40 is plenty.
#
#   sbatch slurm/ptrue_unsup.sbatch

# No `set -e`: one dataset failing must not abort the rest.
set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
export TORCH_HOME=/vol/gpudata/gs925-msc_project/hf_cache/torch
export NLTK_DATA=/vol/gpudata/gs925-msc_project/hf_cache/nltk_data
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B
OOD=ID

echo "=== node ==="; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date

echo; echo "########## 1. AlignScore vs Joe (numerical check) ##########"
python scripts/checks/alignscore_vs_authors.py --model "$MODEL" --n 20

echo; echo "########## 2. Unsupervised P(True) extraction ##########"
for d in sciq trivia_qa pubmed_qa xsum; do
  echo "---- $d ----"
  python scripts/01g_ptrue_unsup.py --model "$MODEL" --dataset "$d" --ood "$OOD"
done

echo; echo "########## 3. Unsupervised P(True) PRR (read-only) ##########"
python - <<'PY'
import sys; sys.path.insert(0, "src")
from luq import cache, results
from luq.config import Config
m = "meta-llama/Meta-Llama-3.1-8B"
for d in ["sciq", "trivia_qa", "pubmed_qa", "xsum"]:
    cd = Config(model_name=m, dataset=d, ood_setting="ID").cache_dir
    recs = [r for r in cache.load_records(cd, cache.run_key(m, d, "ID")) if r["split"] == "test"]
    if not all(isinstance(r.get("ptrue_unsup"), (int, float)) for r in recs):
        print(f"{d:10s} ptrue_unsup not present"); continue
    mass = sum(r.get("ptrue_unsup_mass", 0.0) for r in recs) / len(recs)
    for lab in ["correctness", "correctness_alignscore"]:
        if all(isinstance(r.get(lab), (int, float)) for r in recs):
            y = [r[lab] for r in recs]; u = [r["ptrue_unsup"] for r in recs]
            print(f"{d:10s} {lab:22s} P(True)-unsup PRR {results.prr(y, u):+.3f}   meanmass {mass:.2f}")
PY

echo; echo "=== ALL DONE $(date) ==="
