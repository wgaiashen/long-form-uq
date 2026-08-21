#!/bin/bash
# Label each Qwen dataset with the LLM judge as soon as ITS generation job has finished, so that
# labelling (API-only, no GPU) overlaps generation (GPU-bound) instead of queueing behind all of it.
#
# Runs on the jump box on purpose: it needs outbound internet and no GPU, and DoC caps SUBMITTED jobs
# at 8, which the generation array already consumes in full. 02_label checkpoints every 25 rows, so a
# jump-box kill costs at most 25 judge calls -- rerun this script and it continues.
#
# WAIT FOR THE GENERATION JOB TO LEAVE THE QUEUE. Do NOT label a dataset that is still generating.
# 01_extract holds `records` in memory and rewrites the WHOLE file at every checkpoint, so any label
# written into that file while it is still running is silently destroyed by the next checkpoint -- and
# those labels cost money. Row count alone is not sufficient evidence: a resumed job re-reads the file
# it is about to overwrite. So this gates on (job gone from squeue) AND (row count == expected).
#
# --judge gpt-5-mini IS PASSED EXPLICITLY AND MUST BE. 02_label's default is "whichever judge
# already labelled this dataset", and for a fresh Qwen cache NOTHING has, so it silently falls back to
# the pinned gpt-5-2025-08-07. Every Llama dataset is stamped gpt-5-mini (verified on the records, all
# eight). A probe can learn a judge's biases and PRR needs one yardstick, so mixing judges across the
# two models would invalidate the comparison -- and it would do so after the money was spent.
#
#   source /vol/gpudata/gs925-msc_project/.openai_key
#   bash scripts/02_label_when_ready.sh            # polls until all 8 are labelled

set -uo pipefail
cd "$(dirname "$0")/.."
source pbs/_env.sh; luq_activate || exit 1; cd "$LUQ_REPO"

MODEL="Qwen/Qwen2.5-14B"
JUDGE="gpt-5-mini"
SLUG="Qwen_Qwen2.5-14B"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set. Run: source /vol/gpudata/gs925-msc_project/.openai_key" >&2
  exit 1
fi

# NOT EVERY DATASET IS LABELLED BY 02_label.py. expertqa and factscore have their own labellers,
# and using the generic one on them is silently wrong rather than an error: it writes the shared
# `correctness` field, which is last-labeller-wins, while those two datasets are scored on EXPLICIT
# fields the ladder selects by name (--label-field factuality). 02_label_expertqa.py says so in as
# many words: it writes `factuality` "never the shared `correctness` field".
# It also produces `uncovered` and `coherent`, which are what the M3 coverage rule and the M4
# prediction are computed from -- so the generic labeller does not merely mislabel, it fails to
# produce the numbers the replication is being judged on. Llama's expertqa records have no
# `correctness` key at all, which is the check that catches this.
#
# dataset : expected rows : prompt-regime ("" = default namespace) : labeller script
TARGETS=(
  "pubmed_qa:3800::02_label.py"
  "xsum:3800::02_label.py"
  "cnn_dailymail:3800::02_label.py"
  "med_quad:1800::02_label.py"
  "samsum:1800::02_label.py"
  "asqa:948:asqa_rp12:02_label.py"
  "expertqa:2016:expertqa_rp12:02_label_expertqa.py"
  "factscore:500:factscore_rp12:02_label_factscore.py"
)

done_marker () { echo "logs/.labelled_${1}"; }

while true; do
  remaining=0
  for t in "${TARGETS[@]}"; do
    IFS=':' read -r DS EXPECT REG LABELLER <<< "$t"
    [ -f "$(done_marker "$DS")" ] && continue
    remaining=$((remaining + 1))

    # (a) has this dataset's generation job actually FINISHED? The job prints "=== <ds> done at ..."
    # only after 01_extract returns, so that line is direct evidence the writer has stopped. Anything
    # weaker (rows present, job absent from squeue) can be true mid-run or between a kill and a resume.
    LOG=$(grep -l "^=== $DS (budget" logs/qwen_generate_full_*.out 2>/dev/null | head -1)
    [ -n "$LOG" ] || continue                        # not started yet
    grep -q "^=== $DS done at" "$LOG" || continue    # still generating

    # (b) does the record file hold the full expected row count?
    RECDIR="${LUQ_CACHE_ROOT:-cache}${REG:+/$REG}/records"
    F="$RECDIR/${SLUG}__${DS}__ID.jsonl"
    [ -f "$F" ] || continue
    N=$(wc -l < "$F")
    if [ "$N" -ne "$EXPECT" ]; then
      echo "[$(date +%H:%M)] $DS: $N/$EXPECT rows — still generating, not labelling yet"
      continue
    fi

    echo "[$(date +%H:%M)] $DS: $N/$EXPECT rows complete -> $LABELLER with $JUDGE"
    # The two dedicated labellers take no --dataset (each handles exactly one) and default to the
    # right regime, but the regime is passed anyway so the call is self-describing in the log.
    if [ "$LABELLER" = "02_label.py" ]; then
      python -u "scripts/$LABELLER" --model "$MODEL" --dataset "$DS" --ood ID \
          --judge "$JUDGE" ${REG:+--prompt-regime "$REG"} 2>&1 | tee -a "logs/label_${DS}.log"
    else
      python -u "scripts/$LABELLER" --model "$MODEL" --ood ID \
          --judge "$JUDGE" --prompt-regime "$REG" 2>&1 | tee -a "logs/label_${DS}.log"
    fi
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
      touch "$(done_marker "$DS")"
      echo "[$(date +%H:%M)] $DS: LABELLED"
    else
      echo "[$(date +%H:%M)] $DS: labelling exited non-zero — will retry on the next pass" >&2
    fi
  done

  [ "$remaining" -eq 0 ] && { echo "all datasets labelled"; break; }
  sleep 300
done
