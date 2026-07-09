"""Orgad exact-answer location the RIGHT way (Joe's way): extract the MODEL'S OWN short answer with an
LLM, then locate that span -- instead of string-matching the GOLD answer.

WHY (the audit fix): locating the GOLD answer makes "located" ~ the correctness label (located|correct
99%, located|incorrect 0.2%), so masked (located) rows are ~all correct and unmasked (unlocated) rows are
~all wrong -> the +/-mask comparison is confounded along the label axis (see the STEP-2 leak). Joe's code
instead asks the model to extract ITS OWN short answer from ITS OWN generation (right or wrong), so a span
is found for correct AND incorrect rows and "located" no longer tracks correctness.

This is a faithful port of `temp_idea_1_msp_probe/extract_exact_answer/get_exact_answer.py`
(`extract_exact_answer` + `get_indices_of_exact_answer`), with two deliberate changes, both documented in
the report's method section:
  * we use an API LLM (gpt-5-mini) for the extraction instead of a local Llama generate() -- cheaper and
    what the author asked for; the prompt is Joe's verbatim (2 few-shot examples, "NO ANSWER" escape).
  * SHORT-FORM only, exactly like Joe (his code asserts dataset in {truthfulqa, sciq, medquad}); a long
    multi-sentence answer has no single short answer.
Validity rule is Joe's: the extracted string must be a non-empty substring of the model answer and not
"NO ANSWER"; else the row is unlocated.
"""
import os
import time

# Joe's extraction prompt, verbatim (extract_exact_answer, the f-string body).
EXTRACT_PROMPT = """
Extract from the following long answer the short answer, only the relevant tokens. If the long answer does not answer the question, output NO ANSWER.

The short answer ('Exact answer') must be a subset of the tokens from the short answer. If required, they can be exactly the same.

Q: Which musical featured the song The Street Where You Live?
A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady." It is one of the most famous songs from the show, and it is sung by Professor Henry Higgins as he reflects on the transformation of Eliza Doolittle and the memories they have shared together.
Exact answer: My Fair Lady

Q: Which Swedish actress won the Best Supporting Actress Oscar for Murder on the Orient Express?
A: I'm glad you asked about a Swedish actress who won an Oscar for "Murder on the Orient Express," but I must clarify that there seems to be a misunderstanding here. No Swedish actress has won an Oscar for Best Supporting Actress for that film. The 1974 "Murder on the Orient Express" was an American production, and the cast was predominantly British and American. If you have any other questions or if there's another
Exact answer: NO ANSWER

Q: {question}
A: {model_answer}
Exact answer:"""

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set (login node; source .openai_key).")
        _client = OpenAI()
    return _client


def extract_model_answer(question, model_answer, model="gpt-5-mini", max_retries=4):
    """Ask the LLM to extract the model's OWN short answer from its generation. Returns the extracted
    string (a substring of model_answer) or "NO ANSWER". Joe's validity rule: non-empty substring of the
    model answer, not "NO ANSWER"."""
    if not isinstance(model_answer, str) or not model_answer.strip():
        return "NO ANSWER"
    prompt = EXTRACT_PROMPT.format(question=str(question), model_answer=model_answer)
    client = _get_client()
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=1, top_p=1,
                messages=[{"role": "user", "content": prompt}])
            out = (resp.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(2)
            continue
        out = out.split("\n")[0].strip()                      # first line (Joe truncates at newline)
        for junk in ("<|end_of_text|>", "<|begin_of_text|>", "Exact answer:"):
            out = out.replace(junk, "")
        out = out.strip().strip(".").strip()
        if out and out != "NO ANSWER" and out.lower() in model_answer.lower():
            return out
        if out == "NO ANSWER":
            return "NO ANSWER"
    return "NO ANSWER"


def locate_extracted_rows(tokenizer, gen_ids, extracted):
    """Locate the EXTRACTED answer's token span in the generated tokens, mapped to per-token-cache ROW
    indices (window [P-1:P+G], so gen token j -> row j+1). Faithful to Joe's get_indices_of_exact_answer
    binary search, run over the generation only. Returns (rows, found)."""
    if not extracted or extracted == "NO ANSWER":
        return [], False
    gen_ids = list(gen_ids)
    full = tokenizer.decode(gen_ids, skip_special_tokens=True)
    idx = full.lower().find(extracted.lower().strip())
    if idx == -1:
        return [], False
    true_ans = full[idx: idx + len(extracted)]
    if true_ans not in full:
        return [], False
    higher = len(gen_ids) - 1
    lo = 0
    while true_ans in tokenizer.decode(gen_ids[lo: higher + 1], skip_special_tokens=True):
        higher -= 1
    higher += 1
    while true_ans in tokenizer.decode(gen_ids[lo: higher + 1], skip_special_tokens=True):
        lo += 1
    lo -= 1
    if lo > higher or lo < 0:
        return [], False
    rows = [t + 1 for t in range(lo, higher + 1)]              # gen token t -> cache row t+1
    return rows, True
