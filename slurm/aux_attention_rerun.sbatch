#!/bin/bash
#SBATCH --job-name=aux_rerun
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH --output=logs/aux_rerun-%j.out

# B.1 RE-RUN -- auxiliary-loss attention supervision, done properly (DOC_INSTRUCTIONS.md §3).
#
# The original screen concluded the mechanism was null. That was unsupported: it ran ID-only (where the
# claimed OOD benefit is zero by construction), lambda was tuned on a same-dataset slice (which drives
# lambda->0 regardless of merit), and the dense targets made the summed penalty scale as 1/n. All three
# are fixed here: all 5 rungs, lambda selected by holding out a whole source dataset, and the penalty
# referenced to uniform attention so one lambda means the same thing at 56 and 768 tokens.
#
# WALLTIME IS 72h. Nine RCS jobs died at an 8h wall on 2026-08-05; those were killed by WALLTIME, not
# OOM -- 64G ran them for 8h with no memory trouble -- so the headroom goes into TIME, not RAM.
# ⚠️ 72h, not 24h, because EVERY DoC partition allows 3 days (a30/a40/a100 = 3-00:00:00, long = 30 days)
# and `scontrol update TimeLimit` is DENIED once a job is RUNNING -- it only works while PENDING. So a
# too-short wall cannot be fixed later; it can only be fixed by throwing the job away and restarting.
# Asking for the partition maximum costs nothing here and removes that trap.
#
# MEMORY IS 40G, NOT 64G. Measured peak RSS is 21.7GB (from slurm/aux_preflight.sbatch). 64G was not
# merely generous, it was HARMFUL: long-partition nodes gpuvm14-19 have RealMemory=64000MB, so a 64G
# (65536MB) request could never schedule there and silently hid six idle GPUs.
#
# The driver flushes the CSV PER CELL, so a walltime kill costs the cell in progress, not the run.
#
# PREREQUISITE: slurm/aux_preflight.sbatch must have passed.
#
#   sbatch slurm/aux_attention_rerun.sbatch

set -uo pipefail
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
cd /vol/gpudata/gs925-msc_project/msc-project-gs925

# One per regime of the taxonomy: pubmed concentrated, cnn spread, xsum no signal in the probabilities,
# factscore factuality.
#
# ⚠️ RUN ONE EVAL PER JOB. Measured 2026-08-05: a cell costs ~2.5h (10-point lambda curve x 3 seeds x 3
# targets), so all four evals in ONE job is ~50h against a 24h wall and truncates at roughly half the grid.
# Split by eval -- 5 cells each, ~12.5h -- and combine the four CSVs afterwards. Nothing about the method
# changes; this is parallelism, not a narrower run.
EVALS=${1:-pubmed_qa,cnn_dailymail,xsum,factscore}
TAG=$(echo "$EVALS" | tr ',' '_')

# REALISED TARGET SET -- state this in every table caption. §3 names five targets; four have builders:
#   nll          = every-token surprisal
#   content_mass = content words
#   topk:0.1     = top-k surprisal
#   orgad        = Orgad spans -- NOT in the default set here, see below
# "rank-normalised surprisal" has NO builder in prior_builders.py and is therefore NOT run. Not
# substituted with a near-neighbour: a missing target is reported as missing.
#
# ⚠️ ORGAD IS RUN SEPARATELY, NOT HERE. build_soft_orgad requires EVERY source in a cell's pool to have a
# broad mask cache, and those exist for pubmed_qa, med_quad and expertqa ONLY. Across this grid that leaves
# exactly ONE runnable cell -- ID/pubmed_qa, whose pool is {pubmed_qa}. Every other cell's pool pulls in
# asqa, xsum, cnn, samsum or factscore, none of which are covered, so cnn/xsum/factscore yield NO orgad
# cells at all. Putting orgad in the default list would therefore print a loud skip on nearly every cell
# and produce three rows; it is cleaner to run that one cell as its own job:
#
#   sbatch slurm/aux_attention_rerun.sbatch pubmed_qa orgad     # after cache/orgad_llm/ is synced from RCS
#
TARGETS=${2:-nll,content_mass,topk:0.1}

# LOSS ARM (3rd arg). Section 3 names three losses; lambda=0 is reachable inside the grid below and is
# also the reported prr_baseline, so the two ARMS that need separate passes are:
#   norm  -- uniform-referenced squared error (primary)
#   raw   -- the paper's raw sum (paper-faithful check)
ARM=${3:-norm}
case "$ARM" in
  norm) NORMFLAG="--aux-normalise" ;;
  raw)  NORMFLAG="" ;;
  *)    echo "unknown loss arm '$ARM' (want: norm|raw)" >&2; exit 2 ;;
esac

# OUTPUT NAME -- must encode targets AND arm, not just the eval.
# ⚠️ WHY: with the eval alone, `sbatch ... pubmed_qa orgad` and `sbatch ... pubmed_qa` both resolved to
# results/aux_rerun_pubmed_qa.csv, so the Orgad run would have SILENTLY OVERWRITTEN the primary run's
# results. Caught before either job ran.
# The default primary invocation keeps its original bare name so the four in-flight jobs stay consistent
# with each other; anything non-default is tagged.
TGT_SLUG=$(echo "$TARGETS" | tr ',:' '_-')
if [ "$TARGETS" = "nll,content_mass,topk:0.1" ] && [ "$ARM" = "norm" ]; then
  OUT=results/aux_rerun_${TAG}.csv                      # legacy name, matches job 270751 already running
else
  OUT=results/aux_rerun_${TGT_SLUG}_${ARM}_${TAG}.csv
fi

# lambda=0 MUST be in the grid or selection cannot choose "none" -- without it the control is unreachable
# and a forced-positive lambda would be read as the method helping.
LAMBDAS=0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8

echo "=== node ==="; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "git: $(git log --oneline -1)"
echo "evals:   $EVALS"
echo "targets: $TARGETS"
echo "arm:     $ARM  (flag: '${NORMFLAG:-<none, raw sum>}')"
echo "lambdas: $LAMBDAS"
echo "out:     $OUT"

if [ -e "$OUT" ]; then
  echo "REFUSING TO RUN: $OUT already exists -- move it aside first." >&2
  exit 3
fi

python scripts/checks/aux_attention_ladder.py \
  --evals "$EVALS" \
  --targets "$TARGETS" \
  --lambdas "$LAMBDAS" \
  --seeds 1,2,3 \
  --layer 15 \
  $NORMFLAG \
  --out "$OUT"
rc=$?

echo; echo "=== exit code: $rc  $(date) ==="
exit $rc
