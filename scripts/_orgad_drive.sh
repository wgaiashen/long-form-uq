#!/bin/bash
# Detached overnight driver for the Orgad important-token method as ONE method across ALL datasets.
# Extracts (resumable, concurrent) the model's own important tokens for every dataset -- QA exact answer
# for sciq/trivia/pubmed/med_quad, key fact-bearing terms for xsum/samsum -- then auto-submits the
# ID + OOD ladder eval and the token-weight visualiser to the scheduler. Runs under setsid+nohup so it
# survives the interactive session ending. Fully resumable (each dataset's extraction skips cached rows).
set -uo pipefail
cd /rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925
source pbs/_env.sh
luq_activate || exit 1
set -a; source /rds/general/user/gs925/home/gs925-msc_project/.openai_key; set +a
echo "=== orgad drive start $(date) ===" >&2
# eval datasets first (they carry the headline ID+OOD cells), then the source-only datasets
for d in sciq trivia_qa pubmed_qa xsum med_quad samsum; do
    echo "=== extracting $d $(date) ===" >&2
    python -u scripts/01o_orgad_llm_extract.py --dataset "$d" --workers 12
done
echo "=== all extractions done; submitting ladder + viz $(date) ===" >&2
qsub pbs/orgad_ladder.pbs
qsub pbs/viz_token_weights.pbs
echo "=== orgad drive done $(date) ===" >&2
