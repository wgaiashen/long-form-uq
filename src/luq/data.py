"""Stage 0: data. A thin wrapper over ProbeDrift.

ProbeDrift hands us formatted prompts (.x) and gold targets (.y) only. We run the
model on .x ourselves and compute correctness ourselves. This module is the loader
plus two small lookup tables the rest of the pipeline needs.
"""
from probe_drift import get_datasets

# ProbeDrift key -> the name Joe's llm_as_a_judge script expects.
# The judge asserts the dataset name appears in the JSONL filename, so map before use.
JUDGE_NAME_MAP = {
    "sciq": "sciq",
    "trivia_qa": "triviaqa",
    "qa": "coqa",
    "pubmed_qa": "pubmed",
    "xsum": "xsum",
    "cnn_dailymail": "cnn_dailymail",
}

# Datasets whose correctness comes from string match (the rest use the LLM judge).
SHORT_FORM = {"sciq", "trivia_qa", "qa"}

# `task` column for the results CSV (read by robust-uq-eval).
TASK_OF = {
    "sciq": "short_qa",
    "trivia_qa": "short_qa",
    "qa": "short_qa",
    "pubmed_qa": "long_qa",
    "xsum": "summarisation",
    "cnn_dailymail": "summarisation",
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
}


def load(dataset: str, ood_setting: str = "ID"):
    """Return (train_ds, eval_ds) for one ProbeDrift dataset + OOD setting.

    Each Dataset exposes .x (prompts) and .y (targets) and iterates as `for xb, yb in ds`
    with batch_size=1. The updated ProbeDrift fixes the splits at build time (no `seed`
    argument, no per-example max_new_tokens), so the train order is deterministic and the
    generation budget comes from MAX_NEW_TOKENS above, not the Dataset.
    """
    return get_datasets(
        eval_dataset=dataset,
        ood_setting=ood_setting,
        instruct=False,
        batch_size=1,
    )
