#!/bin/bash
#SBATCH --job-name=llama-uhead
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/llama_uhead-%j.out

# UHead baseline on Llama-3.1-8B, reproducing the paper's Table-1 Uhead by running the reference
# `luh` code: his feature extractors (token probabilities top-4 + all-layer attention, history 2)
# + his FullSeqHead transformer, imported from Temp_robust_UQ_probes behind a TF stub, driven by
# our cached records via a teacher-forced forward. Trained on AlignScore (the paper's primary
# Table-1 correctness signal, which we have on Llama and verified ~1e-7 vs the reference implementation). Writes the
# `uhead` (v1) / `uhead_v2` (v2) scores; the head trains in-job, so no separate 03 probe step.
#
# fp32 + eager base (matches the reference implementation; eager is REQUIRED for the attention features). The all-layer
# attention is memory-heavy: if XSum OOMs, lower --max-seq or add --dtype bf16 to the xsum calls.
# Independent of the lookback/ptrue jobs -> run in parallel.
#
# Target (paper Table 1, Llama, AlignScore): short-form ID ~0.63, long-form ID ~0.31 (1-layer) /
# ~0.27 (2-layer). These are 3-dataset averages, so per-dataset numbers won't match a cell exactly;
# the reproduction is validated by running HIS code, and sanity-checked against that ballpark.
#
# After it finishes, evaluate against AlignScore to line up with Table 1:
#   for DS in sciq trivia_qa pubmed_qa xsum; do
#     python scripts/04_eval.py --model meta-llama/Meta-Llama-3.1-8B --dataset $DS --ood ID \
#         --label-field correctness_alignscore
#   done

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

MODEL=meta-llama/Meta-Llama-3.1-8B
echo "llama_uhead (the luh) start on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# sciq/trivia (short, small seq) first, then long-form. xsum last (heaviest attention).
for DS in sciq trivia_qa pubmed_qa xsum; do
  echo "================ $DS : uhead v1 (his luh, AlignScore) ================ $(date)"
  python scripts/01j_uhead_reference.py --dataset "$DS" --ood ID --model "$MODEL" \
      --variant v1 --name uhead --label-field correctness_alignscore \
    || echo "!!! uhead v1 $DS FAILED"
  echo "================ $DS : uhead v2 (his luh, AlignScore) ================ $(date)"
  python scripts/01j_uhead_reference.py --dataset "$DS" --ood ID --model "$MODEL" \
      --variant v2 --name uhead_v2 --label-field correctness_alignscore \
    || echo "!!! uhead v2 $DS FAILED"
done
echo "llama_uhead ALL DONE at $(date)"
