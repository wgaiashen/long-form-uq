#!/bin/bash
# Shared body for the relative-distance grid runs. Sourced by the per-budget wrappers, which differ
# only in which evaluation datasets they cover and which background budget they slice to.
#
# WHY THIS IS SPLIT BY BUDGET AT ALL. The reference regenerates its background corpus at the
# generation budget of the evaluation dataset in hand, so the background a response is compared
# against always has the same generated length as the response population. Our grid spans eight
# datasets with four different budgets, and the driver takes one budget per invocation. Running the
# whole grid at a single budget would compare five of the eight against a background of the wrong
# length. Splitting the run by budget keeps each dataset matched to a background of its own length,
# which is what the reference does, at no extra compute: each invocation covers only its own
# datasets, so the eight are partitioned rather than repeated.
#
# ONE GPU PASS SERVES ALL FOUR. The background was generated once at 384 generated tokens, the
# largest budget here. Greedy decoding makes a shorter generation an exact prefix of a longer one, so
# every shorter budget is recovered by slicing, and the generation job verified that rather than
# assuming it: eight of eight rows exact at budget 56.
#
# WHAT ELSE THIS PRODUCES, AND WHY IT IS USEFUL. Each invocation recomputes the non-relative methods
# too. Those already exist from the corrected-regression run, so they are a free invariance control:
# msp, saplma, md_mean_mid, satmd_mid, huq_satmd_mid and hbo must come back equal to the existing
# rows. A relative-distance run that moves them has changed something it was not supposed to touch.
#
# THE TWO METHODS THAT NEED PREDICTIVE ENTROPY, msp_satmd_mid and msp_satrmd_mid, will be left blank
# wherever the entropy cache is absent. The driver prints that decision per cell rather than
# substituting a value. Entropy is currently available for three of the eight datasets, so those two
# will be partial and must be reported as such.
#
# Parameters set by the wrapper: LUQ_RMD_EVALS, LUQ_RMD_BUDGET, LUQ_RMD_TAG

set -uo pipefail
cd "$PBS_O_WORKDIR"; source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

mkdir -p logs results/hybrids
LIVE="logs/md_hybrids_rmd_${LUQ_RMD_TAG}__${PBS_JOBID%%.*}.log"
exec > >(tee -a "$LIVE") 2>&1
echo "live log: $LIVE"

MODEL="meta-llama/Meta-Llama-3.1-8B"
LAYER=15
export LUQ_REGIME="med_quad=cleanv2"
BG="cache/background_c4/meta-llama_Meta-Llama-3.1-8B__L15__b384.npz"

echo "=== relative-distance grid, $MODEL === host=$(hostname) at $(date)"
echo "commit: $(git rev-parse HEAD 2>/dev/null)"
echo "working tree clean: $(test -z "$(git status --porcelain --untracked-files=no)" && echo yes || echo NO)"
echo "evals: $LUQ_RMD_EVALS   background budget: $LUQ_RMD_BUDGET"

if [ ! -f "$BG" ]; then echo "!!! background absent at $BG"; exit 2; fi

echo; echo "################ port equivalence against the reference implementation ################"
python -u scripts/checks/md_port_equivalence.py || { echo "!!! port equivalence FAILED -- refusing to run"; exit 2; }

echo; echo "################ grid ################ $(date)"
python -u scripts/checks/md_hybrids.py --model "$MODEL" --layer "$LAYER" \
    --seeds 1,2,3 --evals "$LUQ_RMD_EVALS" \
    --background "$BG" --bg-budget "$LUQ_RMD_BUDGET" \
    --out "results/hybrids/pdl_hybrids_rmd_bg${LUQ_RMD_BUDGET}__meta-llama_Meta-Llama-3.1-8B.csv" \
    || { echo "!!! grid FAILED"; exit 4; }

echo; echo "=== done at $(date) ==="
