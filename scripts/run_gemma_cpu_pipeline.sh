#!/bin/bash
# CPU/judge half of the Gemma migration, run on the login VM (has internet for the
# judge). Waits for the GPU sbatch (gemma_gpu.sbatch) to drop a per-dataset sentinel,
# then labels -> probes -> evals that dataset. Designed to run unattended in the
# background overnight; every action is appended to logs/gemma_migration.md so the
# morning reader sees exactly what happened.
#
#   nohup bash scripts/run_gemma_cpu_pipeline.sh >/dev/null 2>&1 &
#
# sciq needs no judge (string match), so it completes with or without a key. pubmed_qa
# and xsum need OPENAI_API_KEY; if it is missing, their generation+features are still
# saved and labelling is simply left for the morning.

set -uo pipefail
cd /vol/gpudata/gs925-msc_project/msc-project-gs925
source /vol/gpudata/gs925-msc_project/miniconda/etc/profile.d/conda.sh
conda activate luq
export HF_HOME=/vol/gpudata/gs925-msc_project/hf_cache

MODEL=google/gemma-2-9b-it
OOD=ID
LOG=logs/gemma_migration.md
RAW=logs/gemma_migration.raw.log
PY=python
MAX_WAIT_MIN=900     # give up waiting on a dataset's GPU features after ~15h
# A running process freezes its environment at launch, so a key exported into .bashrc
# AFTER this started would never reach us. Instead we read the key from this file at
# judge-time, so it can be dropped in any time before the long-form datasets are reached.
KEYFILE=/vol/gpudata/gs925-msc_project/.openai_key

log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOG"; }
section() { printf '\n### %s\n' "$1" >> "$LOG"; }

# Load the OpenAI key from the env or the keyfile (whichever is available right now).
# The keyfile is a sourceable shell snippet (`export OPENAI_API_KEY=sk-...`), so we
# source it rather than read it raw.
load_key() {
  if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$KEYFILE" ]; then
    # shellcheck disable=SC1090
    source "$KEYFILE"
  fi
}

section "Orchestrator run started (pid $$)"
load_key
if [ -z "${OPENAI_API_KEY:-}" ]; then
  log "No OpenAI key yet (env unset, no $KEYFILE). sciq will complete regardless. For pubmed_qa/xsum, drop the key into $KEYFILE any time before they are reached and the judge will pick it up automatically."
else
  log "OpenAI key present -> long-form judge enabled (will validate gpt-5-mini first)."
fi

MINI_STATUS="unchecked"   # unchecked | adopt | reject

for DS in sciq pubmed_qa xsum; do
  section "Dataset: $DS"
  SENT="cache/.gpu_done__${DS}__${OOD}"

  # ---- wait for GPU features ----
  log "$DS: waiting for GPU features ($SENT)"
  waited=0
  while [ ! -f "$SENT" ]; do
    sleep 60; waited=$((waited+1))
    if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then break; fi
  done
  if [ ! -f "$SENT" ]; then log "$DS: TIMEOUT waiting for GPU features after ${MAX_WAIT_MIN}m -- skipping"; continue; fi
  log "$DS: GPU features present (waited ${waited}m)"

  # ---- label ----
  if [ "$DS" = "sciq" ]; then
    if $PY scripts/02_label.py --dataset "$DS" --ood "$OOD" --model "$MODEL" >>"$RAW" 2>&1; then
      log "$DS: labelled via string match"
    else
      log "$DS: LABEL FAILED (see $RAW) -- skipping probe/eval"; continue
    fi
  else
    load_key   # re-check the keyfile now, in case it was dropped in after launch
    if [ -z "${OPENAI_API_KEY:-}" ]; then
      log "$DS: no key -> features saved, labelling left for morning. Command: python scripts/02_label.py --dataset $DS --ood $OOD --model $MODEL --judge gpt-5-mini"
      continue
    fi
    # validate gpt-5-mini once, on the first long-form dataset reached
    if [ "$MINI_STATUS" = "unchecked" ]; then
      log "$DS: validating gpt-5-mini vs GPT-5 on 100 records (committing test)"
      $PY scripts/checks/validate_mini_judge.py --dataset "$DS" --ood "$OOD" --model "$MODEL" --n 100 >>"$RAW" 2>&1
      vc=$?
      if [ "$vc" -eq 0 ]; then MINI_STATUS="adopt"; log "$DS: mini validation = ADOPT (agreement healthy; see $RAW)";
      elif [ "$vc" -eq 2 ]; then MINI_STATUS="reject"; log "$DS: mini validation = REJECT (judges disagree; NOT mass-labelling long-form -- left for morning)";
      else MINI_STATUS="reject"; log "$DS: mini validation = ERROR/inconclusive (exit $vc); treating as REJECT to be safe -- left for morning"; fi
    fi
    if [ "$MINI_STATUS" != "adopt" ]; then
      log "$DS: skipping labelling (gpt-5-mini not adopted)"; continue
    fi
    if $PY scripts/02_label.py --dataset "$DS" --ood "$OOD" --model "$MODEL" --judge gpt-5-mini >>"$RAW" 2>&1; then
      log "$DS: labelled via gpt-5-mini"
    else
      log "$DS: LABEL FAILED (see $RAW) -- skipping probe/eval"; continue
    fi
  fi

  # ---- probe (saplma/ptrue/lookback) ----
  for M in saplma ptrue lookback; do
    if $PY scripts/03_probe.py --dataset "$DS" --ood "$OOD" --model "$MODEL" --method "$M" >>"$RAW" 2>&1; then
      log "$DS: probe $M ok"
    else
      log "$DS: probe $M FAILED (see $RAW)"
    fi
  done

  # ---- eval (writes results CSV + prints the PRR table) ----
  if $PY scripts/04_eval.py --dataset "$DS" --ood "$OOD" --model "$MODEL" >>"$RAW" 2>&1; then
    log "$DS: eval done -- PRR table:"
    # pull the PRR block this eval just printed into the markdown log
    awk '/^'"$DS"' '"$OOD"' \| test/{f=1} f{print "    "$0}' "$RAW" | tail -20 >> "$LOG"
  else
    log "$DS: EVAL FAILED (see $RAW)"
  fi
  log "$DS: COMPLETE"
done

section "Orchestrator finished at $(date '+%Y-%m-%d %H:%M:%S')"
