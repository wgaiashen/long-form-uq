#!/bin/bash
# Label the two datasets the six-dataset panel does not cover, so a replication population can be
# scored on the full eight-dataset long-form grid registered as deviation D6.
#
# scripts/checks/wmodels_label.sh already labels pubmed_qa, xsum, cnn_dailymail, samsum, asqa and
# factscore. The eight-dataset grid adds med_quad and expertqa, and those are what this covers. Run
# the six-dataset driver first; this one is additive and neither repeats nor overwrites its work.
#
# RUNS ON A LOGIN NODE ON PURPOSE. The judge calls the OpenAI API and compute nodes are not
# guaranteed a route out. This is API-bound and light on CPU, which is the one class of long-running
# work a login node is allowed to carry.
#
# --judge gpt-5-mini IS PASSED EXPLICITLY, EVERY TIME. _resolve_judge() finds nothing stamped on a
# new model slug and falls through to gpt-5 -- roughly 15x the price, and worse, a DIFFERENT
# YARDSTICK. A probe can learn one judge's biases, so the judge is fixed across a comparison.
#
# --prompt-regime expertqa_rp12 IS NOT OPTIONAL. expertqa was generated with repetition_penalty 1.2
# into its own cache namespace; labelling it without the regime reads a different or empty
# population. med_quad uses the default namespace and takes no regime flag.
#
# THE TWO DATASETS DO NOT SHARE A LABEL FIELD, and that is deliberate rather than untidy.
# med_quad writes `correctness` through the standard judge. expertqa writes `factuality` (and
# `consistency`) through the three-state judge, because an instance whose claims are all uncovered
# has no factuality signal at all -- it is recorded as None, never as 0, so a coverage failure can
# never be read as a confident zero.
#
# ON THE ANSWER SPAN, which matters here more than on the other six. med_quad generations often
# continue past the answer with a fabricated `Question:` / `Answer:` block. Measured on
# google/gemma-2-9b, the promoted span rule cuts 2.67% of rows under its literal form and 18.89%
# under the whitespace-tolerant form, because this model tends to write `Question :` with a space.
# THIS SCRIPT LABELS THE RAW GENERATION, matching how every other dataset in the panel was labelled
# and matching the per-token features, which are also computed over the raw generation. Labelling a
# retained span while the features read the full text would put the label and the uncertainty score
# on two different pieces of text, which is the asymmetry this project has already had to correct
# once. A clean-span variant remains available later by slicing, and would require the features to
# be sliced in the same pass.
#
# --out IS MANDATORY ON THE COST ESTIMATE. judge_cost_estimate.py defaults to a TRACKED path
# (results/judge_cost_estimate.csv). Writing it here would dirty the repo as a side effect, and a
# dirty tree aborts any ladder job submitted afterwards.
#
#   bash scripts/checks/wmodels_label_sens8.sh google/gemma-2-9b --estimate-only
#   bash scripts/checks/wmodels_label_sens8.sh google/gemma-2-9b

set -uo pipefail
MODEL="${1:?usage: wmodels_label_sens8.sh <model-id> [--estimate-only]}"
MODE="${2:-}"
SLUG="$(printf '%s' "$MODEL" | tr '/' '_')"
SCRATCH="${LUQ_SCRATCH:-${TMPDIR:-/tmp}}/wmodels_label"
mkdir -p "$SCRATCH"

# The key lives outside this repository as a sourceable `export OPENAI_API_KEY=...` file, mode 600,
# so an unattended overnight run does not die on a missing key. Never echo the value.
if [ -z "${OPENAI_API_KEY:-}" ]; then
  KEYFILE="${LUQ_OPENAI_KEYFILE:-$(cd "$(dirname "$0")/../.." && pwd)/../.openai_key}"
  if [ -f "$KEYFILE" ]; then
    # shellcheck disable=SC1090
    . "$KEYFILE"
    echo "OPENAI_API_KEY loaded from the key file (value not shown)" >&2
  fi
fi
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set and no key file was found -- the judge cannot run}"

echo "=== cost estimate BEFORE spending (model=$MODEL) ===" >&2
python -u scripts/checks/judge_cost_estimate.py \
    --datasets med_quad,expertqa \
    --out "$SCRATCH/judge_cost_estimate_sens8__${SLUG}.csv" 2>&1 | tail -25

if [ "$MODE" = "--estimate-only" ]; then
  echo "--estimate-only: stopping before any spend." >&2
  exit 0
fi

echo >&2
echo "=== labelling, gpt-5-mini, explicit ===" >&2
FAILED=""

echo "--- med_quad (default regime, standard judge -> correctness) ---" >&2
python -u scripts/02_label.py --model "$MODEL" --dataset med_quad --ood ID \
    --judge gpt-5-mini || FAILED="$FAILED med_quad"

echo "--- expertqa (expertqa_rp12, three-state judge -> factuality) ---" >&2
python -u scripts/02_label_expertqa.py --model "$MODEL" \
    --prompt-regime expertqa_rp12 --judge gpt-5-mini || FAILED="$FAILED expertqa"

echo >&2
echo "=== judge coverage (declines are DATA, not gaps) ===" >&2
# An instance whose claims are all uncovered has no factuality signal. The judge RAN and declined to
# score it. That is a property of the task and must be reported as such, never written as 0.
MODEL="$MODEL" python - <<'PY'
import os, sys
sys.path.insert(0, "src")
from luq import cache
from luq.config import Config

MODEL = os.environ["MODEL"]
REGIME = {"expertqa": "expertqa_rp12"}
FIELD = {"expertqa": "factuality"}
print(f"{'dataset':16s} {'rows':>6s} {'judged':>7s} {'coverage':>9s}")
for ds in ["med_quad", "expertqa"]:
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
    print(f"{ds:16s} {n:6d} {j:7d} {j / n if n else 0:9.3f}   field={f}")
PY

if [ -n "$FAILED" ]; then
  echo >&2
  echo "!!! FAILED:$FAILED -- rerun; both labellers are resumable and skip stamped rows." >&2
  exit 1
fi
echo >&2
echo "=== done ===" >&2
