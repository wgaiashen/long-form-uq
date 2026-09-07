#!/bin/bash
# Label one W-Models replication population across the six-dataset panel.
#
# RUNS ON A LOGIN NODE ON PURPOSE. The judge calls the OpenAI API and compute nodes are not
# guaranteed a route out. This is API-bound and light on CPU, which is the one class of long-running
# work a login node is allowed to carry.
#
# --judge gpt-5-mini IS PASSED EXPLICITLY, EVERY TIME. `_resolve_judge()` finds nothing stamped on
# a new model slug and falls through to gpt-5 -- roughly 15x the price, and worse, a DIFFERENT
# YARDSTICK. Never mix judges within one comparison.
#
# THE TWO REGIME DATASETS ARE NOT OPTIONAL EXTRAS. asqa and factscore were generated with
# repetition_penalty 1.2 into their own cache namespaces; labelling them without --prompt-regime
# reads a different (or empty) population.
#
# --out IS MANDATORY ON THE COST ESTIMATE. judge_cost_estimate.py and generation_quality.py both
# default to paths that are TRACKED FILES (results/judge_cost_estimate.csv,
# results/generation_quality.csv). Writing them here would dirty the repo with a side effect.
#
# Expected spend: ~12,648 rows over the six datasets, about £2-4 at the measured gpt-5-mini rate
# ($0.64 / 2,560 rows). Approved before the run.
#
#   bash scripts/checks/wmodels_label.sh meta-llama/Llama-3.1-8B-Instruct
#   bash scripts/checks/wmodels_label.sh google/gemma-2-9b-it --estimate-only

set -uo pipefail
MODEL="${1:?usage: wmodels_label.sh <model-id> [--estimate-only]}"
MODE="${2:-}"
SCRATCH="${LUQ_SCRATCH:-${TMPDIR:-/tmp}}/wmodels_label"
mkdir -p "$SCRATCH"

# The key lives in the PRIVATE PARENT as a sourceable `export OPENAI_API_KEY=...` file, gitignored
# and mode 600. Sourcing it here means an unattended overnight run does not die on a missing key.
# Never echo the value, and never move this file into the public repo.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  KEYFILE="${LUQ_OPENAI_KEYFILE:-$(cd "$(dirname "$0")/../.." && pwd)/../.openai_key}"
  if [ -f "$KEYFILE" ]; then
    # shellcheck disable=SC1090
    . "$KEYFILE"
    echo "OPENAI_API_KEY loaded from the private parent (value not shown)" >&2
  fi
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set and no key file was found -- the judge cannot run}"

echo "=== cost estimate BEFORE spending (model=$MODEL) ===" >&2
python -u scripts/checks/judge_cost_estimate.py \
    --datasets pubmed_qa,xsum,cnn_dailymail,samsum \
    --out "$SCRATCH/judge_cost_estimate__$(printf '%s' "$MODEL" | tr '/' '_').csv" 2>&1 | tail -20

if [ "$MODE" = "--estimate-only" ]; then
  echo "--estimate-only: stopping before any spend." >&2
  exit 0
fi

echo >&2
echo "=== labelling, gpt-5-mini, explicit ===" >&2
FAILED=""
for DS in pubmed_qa xsum cnn_dailymail samsum; do
  echo "--- $DS (default regime) ---" >&2
  python -u scripts/02_label.py --model "$MODEL" --dataset "$DS" --ood ID \
      --judge gpt-5-mini || FAILED="$FAILED $DS"
done

echo "--- asqa (asqa_rp12) ---" >&2
python -u scripts/02_label.py --model "$MODEL" --dataset asqa --ood ID \
    --prompt-regime asqa_rp12 --judge gpt-5-mini || FAILED="$FAILED asqa"

echo "--- factscore (factscore_rp12, three-state factuality judge) ---" >&2
python -u scripts/02_label_factscore.py --model "$MODEL" \
    --prompt-regime factscore_rp12 --judge gpt-5-mini || FAILED="$FAILED factscore"

echo >&2
echo "=== judge coverage (declines are DATA, not gaps) ===" >&2
# On Qwen2.5-14B, factscore came back 466/500: the judge RAN and declined where the reference could
# adjudicate nothing. That is a property of the task, and it must be reported, never written as 0.
MODEL="$MODEL" python - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "src")
from luq import cache
from luq.config import Config

MODEL = os.environ["MODEL"]
REGIME = {"asqa": "asqa_rp12", "factscore": "factscore_rp12"}
FIELD = {"factscore": "factuality"}
print(f"{'dataset':16s} {'rows':>6s} {'judged':>7s} {'coverage':>9s}")
for ds in ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "asqa", "factscore"]:
    cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID",
                 prompt_regime=REGIME.get(ds, ""))
    try:
        recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"))
    except Exception as e:
        print(f"{ds:16s} {'-':>6s} {'-':>7s} {'NOT FOUND':>9s}  ({type(e).__name__})")
        continue
    f = FIELD.get(ds, "correctness")
    n = len(recs)
    j = sum(1 for r in recs if r.get(f) is not None)
    print(f"{ds:16s} {n:6d} {j:7d} {100 * j / max(n, 1):8.1f}%")
PY

[ -n "$FAILED" ] && { echo "!!! labelling failed for:$FAILED" >&2; exit 9; }
echo "=== labelling complete ===" >&2
