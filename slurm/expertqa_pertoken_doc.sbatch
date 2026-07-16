#!/bin/bash
#SBATCH --job-name=pertok_eqa
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/pertok_eqa-%j.out

# ExpertQA per-token L15 on DoC's A100-80GB. Chosen because the RCS L40S queue was jammed (job 3269851
# stuck >3 days). ExpertQA has NO long source (question-only prompt, gen <=384), so it fits ANY GPU -- the
# A100 is just to skip the RCS queue, not for memory. Teacher-forced fp32+eager over cache/expertqa_rp12
# (records already judge-labelled: faithfulness + consistency), writes cache/expertqa_rp12/pertok/...__L15.npz.
# Command is BYTE-IDENTICAL to pbs/expertqa_pertoken.pbs (the RCS version) so the two are interchangeable.
#
# PREREQ on DoC: cache/expertqa_rp12/ must be present (records + the SAPLMA feature npz the reload gate
# checks). If not on DoC, rsync it from RCS first:
#   rsync -avP gs925@login.cx3.hpc.imperial.ac.uk:/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925/cache/expertqa_rp12/ \
#     /vol/gpudata/gs925-msc_project/msc-project-gs925/cache/expertqa_rp12/
# AFTER it finishes: rsync the pertok back DoC->RCS (the ladder runs on RCS):
#   cache/expertqa_rp12/pertok/...__L15.npz   (verify size + that it np.load()s)
#
#   sbatch slurm/expertqa_pertoken_doc.sbatch

set -uo pipefail

export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq

cd /vol/gpudata/gs925-msc_project/msc-project-gs925

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo; echo "########## ExpertQA per-token L15 (fp32 + eager) ##########"
python -u scripts/01h_pertoken.py --model meta-llama/Meta-Llama-3.1-8B \
    --dataset expertqa --ood ID --prompt-regime expertqa_rp12 \
    --layer 15 --dtype fp32 --attn eager

echo; echo "=== DONE $(date) ==="
