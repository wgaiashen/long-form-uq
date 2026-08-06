#!/bin/bash
#SBATCH --job-name=aux_headarm
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --output=logs/aux_headarm-%j.out

# Section 3 SECONDARY: subset-of-heads supervision (K heads, only J supervised, the rest left free).
#
# ⚠️ SELECTION IS "arbitrary_J", NOT the paper's greedy top-J, and the CSV records it that way. The
# paper ranks heads by supervising each individually, which presupposes heads that DIFFER: its 12 heads
# sit inside a fine-tuned BERT and have specialised. Ours sit on a frozen Llama, are initialised
# identically, and B.2 measured them collapsing to 0.996-1.000 correlation under a shared training
# condition. A ranking would have nothing to sort, so the honest label beats performing the ritual.
#
# TWO LAMBDA MODES, because the guard needs supervision to actually be ON to mean anything:
#   sel    -- the full lambda grid, lambda chosen by the protocol. Faithful, but if selection picks
#             lambda=0 the aux term never fires and the heads are trivially identical, which says
#             nothing about whether supervision COULD separate them.
#   forced -- lambda pinned to 1.0. NOT a result to quote as PRR; it is the diagnostic that answers
#             "does supervising a subset make those heads diverge from the free ones AT ALL?"
#             The first run of this check returned sup|free = 0.9996 at lambda=0 -- i.e. with the loss
#             switched off -- which cannot distinguish "supervision does not separate heads" from
#             "supervision was never applied". This mode removes that ambiguity.
#
# PRE-REGISTERED READING RULE: if head_corr_sup_free is ~0.99 in the FORCED mode, the arm did not test
# the paper's structural claim -- it re-confirmed that these heads will not diverge. Write it up that
# way, NOT as "subset-of-heads does not help".
#
#   sbatch slurm/aux_head_arm.sbatch pubmed_qa sel
#   sbatch slurm/aux_head_arm.sbatch pubmed_qa forced

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

EVALS=${1:-pubmed_qa}
MODE=${2:-sel}
K=${3:-4}
TAG=$(echo "$EVALS" | tr ',' '_')

case "$MODE" in
  sel)    LAMBDAS=0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8 ;;
  forced) LAMBDAS=1.0 ;;
  *)      echo "unknown mode '$MODE' (want: sel|forced)" >&2; exit 2 ;;
esac

OUT=results/aux_headarm_K${K}_${MODE}_${TAG}.csv
if [ -e "$OUT" ]; then
  echo "REFUSING TO RUN: $OUT already exists -- move it aside first." >&2
  exit 3
fi

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "git: $(git log --oneline -1)"
echo "evals=$EVALS  K=$K  J=0,1,2,$K  mode=$MODE  lambdas=$LAMBDAS"
echo "out: $OUT"

# J=0 is the unsupervised multi-head baseline; J=K supervises every head (the paper found that WORSE
# than a subset). J=1,2 are the subsets. head_corr_sup_free is blank at J=0 and J=K by construction --
# there is no supervised/free split to measure, and a blank must not read as an uncorrelated zero.
python scripts/checks/aux_attention_ladder.py \
  --evals "$EVALS" \
  --targets nll \
  --lambdas "$LAMBDAS" \
  --seeds 1,2,3 \
  --layer 15 \
  --aux-normalise \
  --n-query "$K" \
  --aux-heads "0,1,2,$K" \
  --out "$OUT"
rc=$?

echo; echo "=== exit code: $rc  $(date) ==="
exit $rc
