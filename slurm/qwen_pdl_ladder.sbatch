#!/bin/bash
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_pdl
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_pdl_%A_%a.out

# The ProbeDriftLong ladder on Qwen2.5-14B: one eval per array task, 3 seeds, the COMPLETE grid
# (every rung cells_long emits -- no --rungs subset, per the standing evaluation rule).
#
# NOT SUBMITTABLE YET. Two things must land first, in this order:
#
#   1. prereg/M2_qwen14b_replication.md committed. The thresholds are already pre-committed in
#      the project's planning notes (commit 9731fea, 7 August, before any Qwen record existed), but
#      the transcription into prereg/ never happened. Once a single ladder number exists it can
#      never honestly be written, so it is a hard predecessor to this job, not a tidy-up after it.
#   2. scripts/checks/probedriftlong.py ported to --model. Line 43 hardcodes
#      MODEL = "meta-llama/Meta-Llama-3.1-8B" and there are 10 usages, all of them path or Config
#      construction (run_key, probe pkl name, per-example npz name, AutoTokenizer, load_per_token,
#      the output CSV name). The port is a --model flag DEFAULTING to that exact string, so every
#      existing invocation stays byte-identical. That file sits on both workstreams' collision
#      lists, so the change is coordinated with RCS and followed by a no-op control: re-run one
#      Llama cell and diff against the published CSV. DoC holds Llama pertok for 5 of 8 datasets
#      (cnn_dailymail, med_quad, pubmed_qa, samsum, xsum), which is enough for that control.
#
# WHY a30 RATHER THAN a100. The ladder is RAM-bound, not GPU-bound: it holds a whole dataset's
# per-token cache in memory (Qwen's are 3-5 GB each, larger than Llama's -- hidden 5120 vs 4096) and
# the only GPU work is training the attention pooler. a30 nodes carry 515 GB and 8 cards; a100 is
# where the pertok extraction array lives, so keeping the ladder off it avoids self-contention.
#
# WHY LAYER 23. the middle-layer rule ceil(N/2)-1, fixed protocol on both models, decided by the
# author 2026-08-09 -- the same rule the Llama grid actually consumed (every Llama pertok on disk is
# __L15). See slurm/qwen_pertoken.sbatch for the artifact evidence.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
export HF_HOME HF_HUB_OFFLINE=1
echo "LUQ_CACHE_ROOT=${LUQ_CACHE_ROOT:-<unset>}  LUQ_CARVE=${LUQ_CARVE:-legacy}"

MODEL="Qwen/Qwen2.5-14B"
LAYER=23

EVALS=(pubmed_qa med_quad asqa xsum cnn_dailymail samsum expertqa factscore)
EV="${EVALS[$SLURM_ARRAY_TASK_ID]}"

OUT="results/probedriftlong__$(echo "$MODEL" | tr '/' '_')__${EV}.csv"

echo "=== ladder $EV (L$LAYER) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

python -u scripts/checks/probedriftlong.py --model "$MODEL" --layer "$LAYER" \
    --evals "$EV" --seeds 1,2,3 --out "$OUT" \
    || { echo "!!! ladder FAILED for $EV"; exit 1; }

echo "=== ladder $EV done at $(date) ==="
