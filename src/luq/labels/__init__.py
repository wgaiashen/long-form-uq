"""Correctness labels. Two jobs: the probe's training target and the PRR ground truth.

    string_match  -> short-form QA, no model (use first, so the label is never the bug)
    llm_judge     -> long-form, the GPT-5 judge, run post-hoc on the login node
"""
