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


def load(dataset: str, ood_setting: str = "ID", seed: int = 1):
    """Return (train_ds, eval_ds) for one ProbeDrift dataset + OOD setting.

    Each Dataset exposes .x (prompts), .y (targets), .max_new_tokens (per example),
    and iterates as `for xb, yb, mnt in ds` with batch_size=1. The eval split is
    subsampled with a fixed seed for reproducibility; the train split uses `seed`.
    """
    return get_datasets(
        eval_dataset=dataset,
        ood_setting=ood_setting,
        instruct=False,
        seed=seed,
        batch_size=1,
    )
