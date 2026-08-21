#!/bin/bash
#SBATCH --job-name=pertok_cnn
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/pertok_cnn-%j.out

# cnn_dailymail per-token L15 states on DoC's A100-80GB. cnn's source articles are long, so fp32 8B + the
# article prompt OOMs RCS 48GB (same wall as xsum's per-token pass) -> DoC. Run ONLY AFTER the judge labels
# are in the records (01h's SANITY gate needs r["correctness"] to compare mean-pool PRR vs the cached SAPLMA
# feature). So first rsync the LABELLED records RCS->DoC:
#   rsync -avP <user>@login.cx3.hpc.imperial.ac.uk:/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925/cache/records/meta-llama_Meta-Llama-3.1-8B__cnn_dailymail__ID.jsonl \
#     /vol/gpudata/gs925-msc_project/msc-project-gs925/cache/records/
# then:  sbatch slurm/cnn_dailymail_pertoken_doc.sbatch
# after it finishes, rsync the pertok cache DoC->RCS (the ladder runs on RCS where the other caches live):
#   cache/pertok/meta-llama_Meta-Llama-3.1-8B__cnn_dailymail__ID__L15.npz  (verify size + that it np.load()s)

set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925
MODEL=meta-llama/Meta-Llama-3.1-8B

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo; echo "########## cnn_dailymail per-token L15 (fp32) ##########"
python scripts/01h_pertoken.py --model "$MODEL" --dataset cnn_dailymail --layer 15

echo; echo "=== DONE $(date) ==="
