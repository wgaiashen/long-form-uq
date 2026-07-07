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
    "med_quad": "med_quad",   # medical QA; a same-task OOD NEIGHBOUR for pubmed_qa (training source)
    "samsum": "samsum",       # dialogue summarisation; xsum's same-task neighbour + a 2nd summ set for QA DiffTask
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
    "med_quad": "long_qa",   # same task family as pubmed_qa (medical QA) -> the SameTask OOD rung
    "samsum": "summarisation",   # dialogue summarisation -> a 2nd summ set (de-degenerates QA DiffTask)
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
    # med_quad answers are 1-3 free-text sentences on one line; 128 (matching its pubmed sibling) is
    # a safe budget and the few-shot format is truncated at the first newline anyway (see load()).
    "med_quad": 128,
    # samsum reference summaries are one sentence (prompt: "Summary (one sentence):"); 56 matches its
    # summarisation sibling xsum so the two are budget-consistent in the QA DiffTask pool.
    "samsum": 56,
}


def load(dataset: str, ood_setting: str = "ID"):
    """Return (train_ds, eval_ds) for one dataset + OOD setting.

    Each Dataset exposes .x (prompts) and .y (targets) and iterates as `for xb, yb in ds`
    with batch_size=1. ProbeDrift datasets come from get_datasets; ExpertQA (not a ProbeDrift
    dataset) is built here from its own loader (src/luq/expertqa.py).
    """
    if dataset == "expertqa":
        return _load_expertqa(ood_setting)
    if dataset == "med_quad":
        return _load_med_quad(ood_setting)
    if dataset == "samsum":
        return _load_samsum(ood_setting)
    return get_datasets(
        eval_dataset=dataset,
        ood_setting=ood_setting,
        instruct=False,
        batch_size=1,
    )


def _load_med_quad(ood_setting: str):
    """med_quad as a first-class dataset, for use as a same-task OOD NEIGHBOUR of pubmed_qa.

    med_quad is train-only in ProbeDrift (no eval split), so `get_datasets(eval_dataset='med_quad')`
    is rejected. Its examples are only reachable as the TRAINING pool of pubmed_qa's
    OOD_ONE_DATASET_SAME_TASK setting (1800 med_quad examples, already few-shot-formatted). We pull
    that pool and present it as the whole dataset under the key `med_quad__ID`, all in the TRAIN
    split (there is no held-out med_quad eval; it is a training source for the pubmed SameTask rung,
    never an eval target). The empty eval split means 01_extract generates only the 1800 train rows.

    The few-shot 'Question: ... Answer: ... Question: ...' format is the same one trivia_qa uses, so
    the model invents a next question after its answer -- extract with --truncate-long to cut at the
    first newline (kept judge-labelled, not string-matched, since medical answers are free text)."""
    if ood_setting != "ID":
        raise ValueError(f"med_quad: only 'ID' is supported (it is a training source, not an eval "
                         f"target); got {ood_setting!r}")
    train_ds, _ = get_datasets(eval_dataset="pubmed_qa", ood_setting="OOD_ONE_DATASET_SAME_TASK",
                               instruct=False, batch_size=1)
    empty_eval = PDDataset([], [], batch_size=1)
    return train_ds, empty_eval


def _load_samsum(ood_setting: str):
    """samsum (dialogue summarisation) as a first-class TRAINING source, exactly like med_quad.

    Purpose: samsum is the summarisation family's second cached set. It (a) de-degenerates the QA
    evals' OOD_DIFF_TASK rung -- their summarisation pool is {samsum, xsum, cnn_dailymail} but only
    xsum was cached, so DiffTask collapsed to a single dataset already inside LOO -- and (b) is xsum's
    OOD_ONE_DATASET_SAME_TASK neighbour, so it also unlocks a summarisation-eval SameTask rung.

    samsum is train-only in ProbeDrift (eval_split='validation', not a keystone eval), so like med_quad
    it is only reachable as a TRAINING pool: it is xsum's same-task neighbour, so we pull it via
    get_datasets(eval_dataset='xsum', OOD_ONE_DATASET_SAME_TASK) -> 1800 samsum rows, already
    few-shot-formatted, and present them under the key `samsum__ID`, all in the TRAIN split (no held-out
    samsum eval; it is a training source only). It is long-form summarisation, so NOT string-matched --
    the summary judge labels it (JUDGE_NAME_MAP['samsum'] -> 'samsum', routed to the summary prompt),
    and it is never truncated."""
    if ood_setting != "ID":
        raise ValueError(f"samsum: only 'ID' is supported (it is a training source, not an eval "
                         f"target); got {ood_setting!r}")
    train_ds, _ = get_datasets(eval_dataset="xsum", ood_setting="OOD_ONE_DATASET_SAME_TASK",
                               instruct=False, batch_size=1)
    empty_eval = PDDataset([], [], batch_size=1)
    return train_ds, empty_eval


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
