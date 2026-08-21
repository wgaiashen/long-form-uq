"""FROZEN candidate answer-span boundaries for trailing next-template continuation.

READ THIS FIRST — WHAT DECIDES A BOUNDARY HERE, AND WHAT NEVER DOES.
A boundary in this module is justified by the dataset's OWN PROMPT TEMPLATE (or, where stated, by
the suffix being an unambiguously different task), and by nothing else. It is **never** chosen or
tuned by whether cutting there improves a PRR, a judge correlation, or any method's score. That
would be selection on the outcome, and this project has already measured methods latching onto junk
signal (on Llama, clean med_quad labels HELPED SAPLMA and HURT wMSP). Every rule below was written
and committed BEFORE its effect on any UQ score was computed — the git history is the audit trail.

This module is DELIBERATELY SEPARATE from `luq.answer_span`:
  * `answer_span` is live on the Llama path via `01_extract --truncate-answer-span` and its rules
    are already promoted; changing it would change an existing, published behaviour.
  * this module is a CANDIDATE rule set for a sensitivity analysis. Nothing consumes it in the
    canonical pipeline. If the author later approves a boundary, it gets wired into `answer_span`
    then, as an explicit promotion.
It is also strictly MODEL-AGNOSTIC. The same rule must be testable on Llama and Qwen; firing often
on one and rarely on the other is a finding, not a reason to retune.

────────────────────────────────────────────────────────────────────────────────────────────────
TWO KINDS OF BOUNDARY, WITH DIFFERENT EVIDENTIAL STANDING. Do not conflate them in a write-up.

TEMPLATE_RESTART — the generation reproduces the prompt's OWN opening verbatim and begins a fresh
example. Provenance is exact and checkable:

  xsum            ProbeDriftUpdate/probe_drift/dataset_configs.py:97
                  "Here's the text and its short summary.\\n\\nText:\\n{text}\\n\\nSummary (one sentence):\\n"
  cnn_dailymail   ProbeDriftUpdate/probe_drift/dataset_configs.py:106   (identical template)
  samsum          ProbeDriftUpdate/probe_drift/dataset_configs.py:132
                  "Here's the dialogue and its short summary.\\n\\nText:\\n{text}\\n\\nSummary (one sentence):\\n"
  med_quad        5-shot "Question: …\\nAnswer: …" — ALREADY covered by luq.answer_span; included
                  here only so the sensitivity can use one code path across datasets.

  For these the model emits the template's first line again, then `Text:` / `Question:`. There is no
  reading on which that is part of the requested summary or answer.

PRETRAINING_ARTEFACT — the suffix is a different task, but it is NOT in the prompt. It does not
appear anywhere in the template:

  expertqa        src/luq/expertqa.py:56  "The following is a question and a detailed, factual
                  answer.\\n\\nQuestion: {question}\\nAnswer:"
  factscore       src/luq/factscore.py:30 "Question: Tell me a bio of {entity}.\\nAnswer:"

  Observed suffix: "A single-select problem: Is the question answered in a satisfactory fashion?
  Available choices: (1). yes (2). no" — an instruction-tuning format the base model reproduces from
  pretraining. It is unambiguously a NEW task (a meta-question ABOUT the answer just produced), so
  the boundary is defensible, but it is NOT derivable from the prompt and that weaker standing is
  recorded on every row this rule fires on.

NEITHER KIND FIXES INTRINSIC DEGENERATION. ExpertQA's severe rows mostly have no marker at all
(21.1% severe against 12.3% bleed) and they start clean and rot — 25% prefix is 2.1% severe, 75%
prefix 62.4%. Cutting at a marker cannot repair text that decays continuously. Template restart and
word-salad degeneration are different phenomena and must stay separate in any conclusion.
────────────────────────────────────────────────────────────────────────────────────────────────
"""
import re

TEMPLATE_RESTART = "template_restart"
PRETRAINING_ARTEFACT = "pretraining_artefact"

# Rules are (compiled pattern, label, standing). Ordered; the EARLIEST match in the text wins, so
# adding a rule can only ever cut earlier, never later. Patterns are anchored to a line start
# (MULTILINE) except where the marker is unambiguous mid-line.
_RULES = {
    "xsum": [
        (re.compile(r"Here'?s the text and its short summary\.", re.I), "prompt-first-line", TEMPLATE_RESTART),
        (re.compile(r"^\s*Text:\s*$", re.M), "Text:-block", TEMPLATE_RESTART),
        (re.compile(r"^\s*Summary \(one sentence\):", re.M), "Summary-cue", TEMPLATE_RESTART),
    ],
    "cnn_dailymail": [
        (re.compile(r"Here'?s the text and its short summary\.", re.I), "prompt-first-line", TEMPLATE_RESTART),
        (re.compile(r"^\s*Text:\s*$", re.M), "Text:-block", TEMPLATE_RESTART),
        (re.compile(r"^\s*Summary \(one sentence\):", re.M), "Summary-cue", TEMPLATE_RESTART),
    ],
    "samsum": [
        (re.compile(r"Here'?s the dialogue and its short summary\.", re.I), "prompt-first-line", TEMPLATE_RESTART),
        (re.compile(r"^\s*Text:\s*$", re.M), "Text:-block", TEMPLATE_RESTART),
        (re.compile(r"^\s*Summary \(one sentence\):", re.M), "Summary-cue", TEMPLATE_RESTART),
    ],
    "med_quad": [
        (re.compile(r"^\s*Question:", re.M), "next-Question:", TEMPLATE_RESTART),
    ],
    "pubmed_qa": [
        (re.compile(r"^\s*Question:", re.M), "next-Question:", TEMPLATE_RESTART),
        (re.compile(r"^\s*Abstract:", re.M), "next-Abstract:", TEMPLATE_RESTART),
    ],
    "expertqa": [
        (re.compile(r"A single-select problem", re.I), "single-select", PRETRAINING_ARTEFACT),
        (re.compile(r"Is the question answered", re.I), "meta-question", PRETRAINING_ARTEFACT),
        (re.compile(r"^\s*Available choices:", re.M | re.I), "available-choices", PRETRAINING_ARTEFACT),
        (re.compile(r"^\s*(?:Pick|Choose|Select) (?:from|your answer)", re.M | re.I), "pick-from", PRETRAINING_ARTEFACT),
    ],
    "factscore": [
        (re.compile(r"A single-select problem", re.I), "single-select", PRETRAINING_ARTEFACT),
        (re.compile(r"Is the question answered", re.I), "meta-question", PRETRAINING_ARTEFACT),
        (re.compile(r"^\s*Available choices:", re.M | re.I), "available-choices", PRETRAINING_ARTEFACT),
        (re.compile(r"^\s*(?:Pick|Choose|Select) (?:from|your answer)", re.M | re.I), "pick-from", PRETRAINING_ARTEFACT),
        (re.compile(r"^\s*Question:\s*Tell me a bio", re.M | re.I), "next-bio-prompt", TEMPLATE_RESTART),
    ],
}

DATASETS = frozenset(_RULES)


def restart_cut(text, dataset):
    """(clean_text, cut_char, reason, standing). cut_char is None when nothing fires.

    `standing` is TEMPLATE_RESTART or PRETRAINING_ARTEFACT — carried through so a table can never
    silently merge a boundary read off the prompt with one that is not in the prompt.
    """
    if not text or dataset not in _RULES:
        return text, None, "no-rule", None
    best = None
    for pat, label, standing in _RULES[dataset]:
        m = pat.search(text)
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), label, standing)
    if best is None:
        return text, None, "no-match", None
    cut, label, standing = best
    return text[:cut].rstrip(), cut, f"{dataset}:{label}@{cut}", standing


def n_tokens_removed(record, dataset, tokenizer=None):
    """How many GENERATED tokens the cut drops, by re-tokenising the retained prefix.

    Returned as (n_keep, n_total). Needs the tokenizer that produced the record; without one it
    falls back to a character-proportional estimate, which is flagged by returning a float.
    """
    text = record.get("gen_text", "") or ""
    keep, cut, _, _ = restart_cut(text, dataset)
    total = len(record.get("gen_token_ids", []))
    if cut is None:
        return total, total
    if tokenizer is None:
        return total * (len(keep) / max(len(text), 1)), total
    return len(tokenizer(keep, add_special_tokens=False).input_ids), total
