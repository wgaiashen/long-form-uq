"""ExpertQA loader for the long-form UQ pipeline (a second FACTUALITY task).

ExpertQA (Malaviya et al., github.com/chaitanyamalaviya/ExpertQA, data/r2_compiled_anon.jsonl):
expert-curated questions across 32 fields, each with exactly one expert-REVISED answer. We use the
revised answer as the instance-level gold. Open-domain factuality: the model sees only a question
(no source document), so this patterns with biographies, not with Xsum/PubMedQA.

Schema (verified against the real file): 2177 questions; exactly 1 answer/question;
gold = answers[*].revised_answer_string (2169/2177 populated, drop the 8 empty);
domain = metadata['field']; filter = metadata['question_type'] ('|'-separated full-text labels).
We emit `field` + `cluster` per record from day one so the Role-C within-ExpertQA domain-shift OOD
split is a free re-slice later.

Prompt + gold-length decisions are Stage-0 locked; see EXPERTQA_PLAN.md. Vendors the field->cluster
map and the factual-core filter from the project's expertqa_loader.py starter.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

# ExpertQA's 32 fields -> the paper's ~6 superclusters (Role C shifts on clusters, not raw fields,
# since 12 fields have <30 examples).
CLUSTER = {
    "Healthcare / Medicine": "professional", "Law": "professional", "Business": "professional",
    "Military or Law Enforcement": "professional",
    "Psychology": "humanities_social", "Education": "humanities_social", "History": "humanities_social",
    "Linguistics": "humanities_social", "Economics": "humanities_social",
    "Political Science": "humanities_social", "Sociology": "humanities_social",
    "Philosophy": "humanities_social", "Anthropology": "humanities_social",
    "Journalism": "humanities_social", "Criminology": "humanities_social",
    "Classical Studies": "humanities_social", "Theology": "humanities_social",
    "Literature": "humanities_social",
    "Chemistry": "natural_physical", "Biology": "natural_physical",
    "Environmental Science": "natural_physical", "Physics and Astronomy": "natural_physical",
    "Mathematics": "natural_physical", "Climate Science": "natural_physical",
    "Geography": "natural_physical",
    "Engineering and Technology": "engineering", "Aviation": "engineering",
    "Visual Arts": "arts", "Architecture": "arts", "Music": "arts", "Culinary Arts": "arts",
    "Other": "other",
}

# Question types where a factual "correct answer" is well defined. Keep a question if it carries at
# least one of these; drop pure opinion / resource-list questions where correctness is ill-defined.
FACTUAL_CORE = {
    "Directed question that has a single unambiguous answer",
    "Open-ended question that is potentially ambiguous",
    "Summarization of information on a topic",
    "Question that describes a hypothetical scenario and asks a question based on this scenario",
    "Advice or suggestions on how to approach a problem",
}

# FROZEN prompt wording (design-once; freezing it stabilises the prompt_hash cache guard). Long-form
# QA style with the Question:/Answer: markers the judge parser reads. No context block (ExpertQA is
# question->answer only). Base-model priming ("detailed, factual answer") to elicit long-form.
PROMPT = ("The following is a question and a detailed, factual answer.\n\n"
          "Question: {question}\nAnswer:")

# Stage-0 locked generation budget: gold revised-answer Llama-tokenizer p90=348, p95=403. 384 covers
# ~92% of natural answer length while staying genuinely long-form (a cap, not a target).
MAX_NEW_TOKENS = 384

# Default data location (downloaded into the project parent dir).
DEFAULT_JSONL = Path(__file__).resolve().parents[3] / "ExpertQA" / "data" / "r2_compiled_anon.jsonl"

# Pilot size + seed (Stage 1). data.load and the Stage-0 manifest use the SAME values, so records
# (enumerate idx) align positionally with the manifest's field/cluster metadata.
PILOT_N = 200
PILOT_SEED = 1


def load_records(path=None, keep_factual_only=True):
    """One dict per usable ExpertQA question: {question, gold, field, cluster, question_types}."""
    path = Path(path) if path else DEFAULT_JSONL
    out = []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        answer = next(iter(rec["answers"].values()))          # exactly one answer per question
        gold = (answer.get("revised_answer_string") or "").strip()
        if not gold:
            continue                                          # drop the 8 with no revision
        types = {t.strip() for t in rec["metadata"]["question_type"].split("|") if t.strip()}
        if keep_factual_only and not (types & FACTUAL_CORE):
            continue                                          # drop pure opinion / resource-list
        field = rec["metadata"]["field"]
        out.append({"question": rec["question"], "gold": gold, "field": field,
                    "cluster": CLUSTER.get(field, "other"), "question_types": sorted(types)})
    return out


def pilot_sample(records, n=200, seed=1):
    """Stratified sample of ~n across (field x question-type-set), deterministic under `seed`.

    Round-robins over strata so small fields/types are represented rather than swamped by
    Healthcare/Medicine. Returns records in a fixed shuffled order."""
    strata = defaultdict(list)
    for r in records:
        strata[(r["field"], tuple(r["question_types"]))].append(r)
    rng = random.Random(seed)
    for k in strata:
        rng.shuffle(strata[k])
    keys = sorted(strata)                                     # deterministic stratum order
    rng.shuffle(keys)
    picked, i = [], 0
    while len(picked) < min(n, len(records)):
        progressed = False
        for k in keys:
            if i < len(strata[k]):
                picked.append(strata[k][i]); progressed = True
                if len(picked) >= min(n, len(records)):
                    break
        if not progressed:
            break
        i += 1
    rng.shuffle(picked)
    return picked
