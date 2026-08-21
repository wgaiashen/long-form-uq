#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_probe
#SBATCH --array=0-7%3
#SBATCH --output=logs/qwen_budget_probe_%A_%a.out

# BUDGET PROBE for Qwen2.5-14B -- the GATE that decides each dataset's token budget.
# 200 rows per dataset at a RAISED cap, so the model's NATURAL generation length is visible. A probe
# run at the old cap can only ever re-measure the old cap.
#
# WHY THIS IS A GATE, NOT A FORMALITY. On Llama the capping had TWO different causes needing
# different responses, and only the gold-vs-budget ratio separates them:
#   med_quad  ratio 0.21x, 97.7% capped -> DEFECT: the reference answer cannot fit the budget.
#   samsum/xsum/cnn  ratio 1.22-1.65x, 28-66% capped -> the model merely OVER-GENERATES.
# (Those four figures were re-measured on this cache on 2026-08-08 and reproduce the recorded table.)
#
# DECISION RULE, deliberately conservative because this is a REPLICATION, not an exploration:
# keep Llama's budgets unless Qwen shows a genuine defect (ratio < 1.0). Over-generation alone is NOT
# a reason to change a budget -- it is a property of the model, which is exactly what we are measuring.
# Changing a budget changes the generated population and weakens the cross-model comparison.
#
# ── TWO DELIBERATE DEPARTURES FROM THE RCS REFERENCE RECIPE (pbs/qwen_budget_probe.pbs) ─────────────
# 1. fp32 + eager, NOT bf16. The reference chose bf16 for reasons that are specific to RCS: fp32
#    Qwen-14B (~59GB) needs two cards there, and its GPU queue was saturated. Neither holds on DoC --
#    the A100 is 80GB (CONFIRMED on-card: 81920 MiB) so fp32 fits on ONE card, and the queue is idle.
#    Running the probe at the real run's precision means it characterises the exact population the
#    full run will produce, instead of a bf16 proxy for it.
# 2. A job array (3 concurrent, the DoC limit) instead of one sequential job, because the datasets are
#    independent and a failure should cost one dataset rather than all eight.
#
# Isolated under --prompt-regime qwen14b_probe, so it writes to cache/qwen14b_probe/ and CANNOT
# pollute the real Qwen cache. Nothing from this namespace may enter a results table.
#
#   sbatch slurm/qwen_budget_probe.sbatch
#   # then, once all 8 array tasks are done (CPU-only, read-only):
#   python scripts/checks/generation_quality.py --model Qwen/Qwen2.5-14B \
#       --gold-tokenizer Qwen/Qwen2.5-14B --regime qwen14b_probe \
#       --datasets pubmed_qa,xsum,cnn_dailymail,med_quad,samsum,expertqa,asqa,factscore

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"
# Offline: the weights are already local, and a network stall mid-run would waste an A100 allocation.
export HF_HOME HF_HUB_OFFLINE=1

MODEL="Qwen/Qwen2.5-14B"
REGIME="qwen14b_probe"
N=200

# dataset : probe cap : extra flags.  The caps are deliberately well above the Llama budget.
# The extra flags are NOT optional decoration -- they are what makes this population comparable to
# Llama's: pubmed_qa is the one long dataset cut at the first newline, and expertqa/asqa/factscore
# were all generated under a 1.2 repetition penalty.
TASKS=(
  "pubmed_qa:512:--truncate-long"
  "xsum:256:"
  "cnn_dailymail:512:"
  "med_quad:768:"
  "samsum:256:"
  "expertqa:768:--repetition-penalty 1.2"
  "asqa:512:--repetition-penalty 1.2"
  "factscore:512:--repetition-penalty 1.2"
)

IFS=':' read -r DS CAP EXTRA <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "=== $DS (probe cap $CAP) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --dtype fp32 is passed EXPLICITLY. `--dtype auto` resolves to fp16 for any non-Gemma model, and
# token probabilities ARE the msp_min signal, so an auto-dtype cache is not comparable to the fp32
# Llama caches. 01_extract refuses `auto` for a non-Llama model; do not work around that guard.
# --max-new-tokens-cap MUST be raised alongside --max-new-tokens: the effective budget is
# min(table_value, cap) and the cap defaults to 128, so without it the probe silently runs at 128.
# shellcheck disable=SC2086
python -u scripts/01_extract.py --model "$MODEL" --dataset "$DS" --ood ID \
    --prompt-regime "$REGIME" --limit "$N" \
    --dtype fp32 --attn eager \
    --max-new-tokens "$CAP" --max-new-tokens-cap "$CAP" $EXTRA

# Quota headroom. The ceph xattrs are not exposed on every compute node (they read back empty on
# vm-he-a100), so this reports UNAVAILABLE rather than throwing -- but it must never silently print a
# reassuring number it did not measure, so the unavailable case says so out loud.
echo "--- quota after $DS ---"
python -c "
import subprocess
def g(a):
    r = subprocess.run(['getfattr','--only-values','-n',a,'/vol/gpudata/gs925-msc_project'],
                       capture_output=True, text=True)
    s = r.stdout.strip()
    return int(s) if r.returncode == 0 and s.isdigit() else None
u, q = g('ceph.dir.rbytes'), g('ceph.quota.max_bytes')
if u is None or q is None:
    print('  quota UNAVAILABLE on this node (ceph xattrs not exposed) -- check from the submit host')
else:
    print(f'  used {u/2**30:.2f} GiB / {q/2**30:.2f} GiB -> FREE {(q-u)/2**30:.2f} GiB')"

echo "=== $DS done at $(date) ==="
