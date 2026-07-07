"""Long-form correctness label: Joe's LLM-as-a-judge (GPT-5), run post-hoc.

Ported faithfully from reference/llm_as_a_judge_scoring_prompt.py so our pipeline is
self-contained and the judge stays fixed and reproducible (keep the model and prompt
pinned, since a probe can learn one judge's biases). The judge scores the model
output against the GOLD reference on a 0.0-1.0 scale, with the source as context.

Two prompt variants, exactly as Joe's script:
  * summarisation (xsum, cnn_dailymail): no in-context examples.
  * QA (everything else): four in-context examples.

Runs on the LOGIN NODE (needs internet + OPENAI_API_KEY) over the cached records.
Each call costs money, so 02_label drives this resumably (skip already-scored records,
checkpoint periodically).
"""
import os
import re
import time

from openai import OpenAI

from ..data import JUDGE_NAME_MAP

MODEL = "gpt-5-2025-08-07"  # pinned: keep the judge model fixed and recorded

# Lazily created so importing this module needs no API key; only judge() does.
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. The judge runs on the login node; "
                "`export OPENAI_API_KEY=sk-...` in your shell first."
            )
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


# ---- prompt templates (verbatim from Joe's script) ----------------------------

def _qa_prompt(question: str, label, answer: str, caveat: str) -> str:
    return f"""

Evaluate the following answers to questions. For each question you will be given a model answer and the correct answer.
You must rate how correct the model answer is on a scale from 0.0 to 1.0, where:
- 1.0 means the model answer is completely correct
- 0.0 means the model answer is completely incorrect or wrong
- Values in between reflect answers that are partially correct but incomplete or imprecise

Only respond with a number between 0.0 to 1.0. Do not write any explanation.

For example:

Question: who is the young guitarist who played with buddy guy?
Ground Truth: Quinn Sullivan
Model Answer: Ronnie Earl
Score: 0.0

Question: name of the first episode of stranger things
Ground Truth: Chapter One: The Vanishing of Will Byers
Model Answer: The disappearance of Will Byers
Score: 0.0

Question: What are the symptoms of diabetes?
Ground Truth: Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.
Model Answer: Symptoms of diabetes include increased thirst and frequent urination.
Score: 0.6

Question: What is the capital of Australia?
Ground Truth: Canberra
Model Answer: Canberra, which is located in the Australian Capital Territory
Score: 1.0

{question}
Ground Truth{caveat} {label}
Model Answer: {answer}
Score:
"""


def _summary_prompt(question: str, label, answer: str) -> str:
    return f"""
Only respond with a number between 0.0 to 1.0. Do not write any explanation.

The task below is a text summarisation task. You will see a Text, a Ground Truth Summary, and a Model Summary. Score how well the Model Summary matches the Ground Truth Summary from 0.0 to 1.0, where 1.0 means it conveys the same information, 0.0 means it is completely different or irrelevant, and scores in between reflect partial overlap in the key points covered.

{question}
Ground Truth Summary: {label}
Model Summary: {answer}
Score:
"""


# ---- per-dataset prompt trimming (verbatim from Joe's __main__) ----------------

def _extract_question(prompt: str, judge_name: str):
    """Trim the full prompt down to the question/context the judge should see, and
    return (question, caveat). Mirrors Joe's per-dataset slicing exactly so the judge
    sees the same text he intends."""
    if judge_name == "sciq":
        prompt = prompt[: prompt.rfind("Answer:")].strip("\n").strip()
        prompt = prompt[prompt.rfind("Context"):]
        prompt = prompt.replace("Context: \n", "")  # drop empty-context mention
        return prompt, ":"
    if judge_name == "triviaqa":
        prompt = prompt[: prompt.rfind("Answer:")]
        prompt = prompt[prompt.rfind("Question:"):].strip("\n").strip()
        return prompt, " (any of the following are correct):"
    if judge_name == "coqa":
        story = prompt[prompt.find("Story:"): prompt.find("Question:")].strip().strip("\n")
        prompt = prompt[: prompt.rfind("Answer:")]
        prompt = prompt[prompt.rfind("Question:"):].strip("\n").strip()
        return story + "\n" + prompt, ":"
    if judge_name == "pubmed":
        prompt = prompt[: prompt.rfind("Answer:")].strip("\n").strip()
        prompt = prompt[prompt.rfind("Abstract:"):]
        prompt = prompt.replace("Abstract: \n", "")  # drop empty-abstract mention
        return prompt, ":"
    if judge_name in ("xsum", "cnn_dailymail", "samsum"):
        # samsum's dialogue-summary prompt uses the identical "Text:\n...\nSummary (one sentence):"
        # markers as xsum, so the same trim + summary template apply verbatim.
        prompt = prompt[: prompt.rfind("Summary")].strip("\n").strip()
        prompt = prompt[prompt.rfind("Text:"):]
        return prompt, None  # caveat unused for the summary template
    if judge_name == "expertqa":
        # ExpertQA prompt = "...\n\nQuestion: {q}\nAnswer:" (no context block). Trim to the question;
        # routes to the QA judge (scores factual correctness of the response vs the expert-revised
        # gold, with partial credit — same QA template as pubmed).
        prompt = prompt[: prompt.rfind("Answer:")]
        prompt = prompt[prompt.rfind("Question:"):].strip("\n").strip()
        return prompt, ":"
    if judge_name == "med_quad":
        # med_quad = few-shot 'Question:/Answer:' medical QA (same prompt shape as triviaqa/expertqa).
        # Trim to the LAST question; QA judge scores the response vs the free-text gold answer with
        # partial credit. Used as a same-task OOD neighbour of pubmed_qa (a training source).
        prompt = prompt[: prompt.rfind("Answer:")]
        prompt = prompt[prompt.rfind("Question:"):].strip("\n").strip()
        return prompt, ":"
    raise ValueError(f"no prompt-trimming rule for judge dataset {judge_name!r}")


# ---- the GPT call + scoring ----------------------------------------------------

def _is_valid_score(text: str) -> bool:
    try:
        return 0.0 <= float(text.strip()) <= 1.0
    except (ValueError, AttributeError):
        return False


_SCORE_RE = re.compile(r"\d*\.?\d+")


def parse_score(text):
    """Lenient score parse: return the first number in [0, 1], else None. Open-source
    instruct judges sometimes add a word despite 'only respond with a number', so the
    local judge uses this; the GPT-5 path keeps the strict _is_valid_score."""
    if text is None:
        return None
    for tok in _SCORE_RE.findall(text):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v
    return None


def _gpt_response(user_prompt: str, model: str) -> str:
    """One judge call. Retries a few times on transient API errors (rate limits,
    network) so a long run survives the occasional hiccup."""
    client = _get_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=1,
                top_p=1,
                logprobs=False,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.choices[0].message.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def build_prompt(record: dict, dataset: str, strip_newlines: bool = False) -> str:
    """Assemble the exact judge user-prompt for one record. Backend-agnostic: the
    GPT-5 judge and any cheaper / open-source judge both call this, so the ONLY thing
    that varies across judges is the model, never the prompt (the fairness rule for
    the judge-agreement check). `dataset` is the ProbeDrift key (e.g. "pubmed_qa"); we
    map it to the judge name ("pubmed") for prompt trimming and template choice.

    strip_newlines: collapse all newlines out of the model answer before judging,
    matching Joe's Hidden Failures judge input (collect_llm_judge_inputs.py:92). OFF
    by default; turn on only to reproduce his labels faithfully.
    """
    judge_name = JUDGE_NAME_MAP[dataset]
    question, caveat = _extract_question(record["prompt"], judge_name)
    label, answer = record["target"], record["gen_text"]
    if strip_newlines:
        answer = answer.replace("\n", "").strip()
    if judge_name in ("xsum", "cnn_dailymail", "samsum"):
        return _summary_prompt(question, label, answer)
    return _qa_prompt(question, label, answer, caveat)


def judge(record: dict, dataset: str, model: str = MODEL, max_retries: int = 10,
          strip_newlines: bool = False):
    """Score one record's gen_text against its gold target with the OpenAI judge.
    Returns a float in [0, 1], or None if it never returned a valid number after
    max_retries. `model` defaults to the pinned GPT-5; pass a cheaper sibling
    (e.g. "gpt-5-mini") to validate it against GPT-5 in scripts/checks.
    `strip_newlines` reproduces Joe's newline-collapsed answer (see build_prompt).
    """
    user_prompt = build_prompt(record, dataset, strip_newlines=strip_newlines)
    for _ in range(max_retries):
        response = _gpt_response(user_prompt, model)
        if _is_valid_score(response):
            return float(response.strip())
    return None  # judge never produced a valid score; 02_label flags these
