"""Stage 0: data. A thin wrapper over ProbeDrift.

ProbeDrift hands us formatted prompts (.x) and gold targets (.y) only. We run the
model on .x ourselves and compute correctness ourselves. This module is the loader
plus two small lookup tables the rest of the pipeline needs.
"""
from probe_drift import get_datasets
from probe_drift.dataset import Dataset as PDDataset

from . import expertqa

# ProbeDrift key -> the name Joe's llm_as_a_judge script expects.
# The judge asserts the dataset name appears in the JSONL filename, so map before use.
JUDGE_NAME_MAP = {
    "sciq": "sciq",
    "trivia_qa": "triviaqa",
    "qa": "coqa",
    "pubmed_qa": "pubmed",
    "xsum": "xsum",
    "cnn_dailymail": "cnn_dailymail",
    "expertqa": "expertqa",   # long-form FACTUALITY QA (not a ProbeDrift dataset)
}

# Datasets whose correctness comes from string match (the rest use the LLM judge).
SHORT_FORM = {"sciq", "trivia_qa", "qa"}

# `task` column for the results CSV (read by robust-uq-eval).
# NOTE: expertqa's OOD placement (factuality vs faithfulness property tag) is DEFERRED — see
# EXPERTQA_PLAN.md "Task tag / OOD placement". This coarse "long_qa" tag is a PLACEHOLDER used only
# for the results CSV column; it is NOT wired into any OOD splitter (the pilot runs no OOD), so it is
# safe until the property-tag decision is made.
TASK_OF = {
    "sciq": "short_qa",
    "trivia_qa": "short_qa",
    "qa": "short_qa",
    "pubmed_qa": "long_qa",
    "xsum": "summarisation",
    "cnn_dailymail": "summarisation",
    "expertqa": "long_qa",   # PLACEHOLDER (property tag deferred)
}

# Per-dataset generation budget. The updated ProbeDrift no longer ships max_new_tokens on
# the Dataset, so we own these now. Values are the originals from ProbeDrift's own configs
# (dataset_configs.py), kept identical so generations match the frozen runs.
MAX_NEW_TOKENS = {
    "sciq": 20,
    "trivia_qa": 20,
    "qa": 20,
    "pubmed_qa": 128,
    "xsum": 56,
    "cnn_dailymail": 128,
    # expertqa: gold revised-answer p90=348 / p95=403 tokens (Llama tokenizer). 384 covers ~92%.
    # GOTCHA: config.max_new_tokens_cap defaults to 128, so ExpertQA extraction MUST pass a higher
    # --max-new-tokens-cap (>=384) or the budget is silently clipped (budget = min(384, cap)).
    "expertqa": expertqa.MAX_NEW_TOKENS,  # 384
}


def load(dataset: str, ood_setting: str = "ID"):
    """Return (train_ds, eval_ds) for one dataset + OOD setting.

    Each Dataset exposes .x (prompts) and .y (targets) and iterates as `for xb, yb in ds`
    with batch_size=1. ProbeDrift datasets come from get_datasets; ExpertQA (not a ProbeDrift
    dataset) is built here from its own loader (src/luq/expertqa.py).
    """
    if dataset == "expertqa":
        return _load_expertqa(ood_setting)
    return get_datasets(
        eval_dataset=dataset,
        ood_setting=ood_setting,
        instruct=False,
        batch_size=1,
    )


def _load_expertqa(ood_setting: str):
    """ExpertQA as (train_ds, eval_ds). For the Stage-1 pilot (ood_setting='pilot') the whole
    stratified 150-200 sample is the eval/test split (no training); the frozen prompt is applied
    here so `01_extract`'s prompt_hash guard is stable. `field`/`cluster` metadata is carried in
    the eval Dataset's source_ids ('field||cluster') and mirrored to the Stage-0 manifest, so it
    joins to records by idx for the later Role-C domain-shift re-slice."""
    recs = expertqa.load_records()
    if ood_setting == "pilot":
        recs = expertqa.pilot_sample(recs, n=expertqa.PILOT_N, seed=expertqa.PILOT_SEED)
    elif ood_setting != "ID":
        raise ValueError(f"expertqa: unsupported ood_setting {ood_setting!r} "
                         "(pilot is Stage 1; full ID/OOD splits are a later phase)")
    x = [expertqa.PROMPT.format(question=r["question"]) for r in recs]
    y = [r["gold"] for r in recs]
    meta = [f"{r['field']}||{r['cluster']}" for r in recs]
    train_ds = PDDataset([], [], batch_size=1)                   # pilot has no training split
    eval_ds = PDDataset(x, y, batch_size=1, source_ids=meta)
    return train_ds, eval_ds
