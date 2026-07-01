"""Orgad et al. ("LLMs Know More Than They Show") exact-answer-token location.

Their "important token" is not a saliency or content filter. It is the token span where the
exact answer appears in the generated text: find the answer substring in the decoded output,
then binary-search the token boundaries that span it, then probe a chosen position of that
span (their primary is the last token). This is a faithful port of
`LLMsKnow/src/probing_utils.py:get_indices_of_exact_answer` (and the `exact_answer_last_token`
selection in `get_token_index`), adapted to our cached records and per-token cache.

It is a SHORT-FORM, single-answer method: it assumes one locatable answer span and otherwise
falls back (their code falls back to the last question token and filters the sample). We run it
on sciq / trivia_qa, where a single gold answer exists. Two honest deviations from their code,
both logged by the caller:
  - we locate the GOLD answer (substring match). For wrong answers the gold is usually absent;
    their code then calls a separate LLM to extract the model's own answer. We skip that LLM
    step (API cost) and fall back to the last answer token instead, and report the fallback rate.
  - trivia gold is a list of aliases; we try each and take the first that is found.
"""


def _span_indices(tokenizer, full_ids, exact_answer, lower):
    """Faithful port: token indices [lo..hi] (into full_ids) whose decode spans exact_answer.

    full_ids: prompt_token_ids + gen_token_ids. lower: index of the first generated token (so
    the search is over the generation only). Returns the index list, or None if not found.
    """
    full_qa = tokenizer.decode(full_ids[lower:])
    idx = full_qa.lower().find(exact_answer.lower().strip())
    if idx == -1:
        return None
    true_ans = full_qa[idx: idx + len(exact_answer)]
    if true_ans not in full_qa:
        return None
    higher = len(full_ids) - 1
    while true_ans in tokenizer.decode(full_ids[lower: higher + 1]):
        higher -= 1
    higher += 1
    lo = lower
    while true_ans in tokenizer.decode(full_ids[lo: higher + 1]):
        lo += 1
    lo -= 1
    if lo > higher:
        return None
    return list(range(lo, higher + 1))


def locate_answer_rows(tokenizer, prompt_ids, gen_ids, gold):
    """Locate the gold answer's token span and map it to PER-TOKEN-CACHE ROW indices.

    The per-token cache window is [P-1 : P+G] (last-prompt token + all G answer tokens), so the
    cache row of an absolute token index t is t - (P-1). Returns (rows, found):
      rows  : list of cache-row indices for the span (empty if not found)
      found : True if the gold answer was located, False if it must be a fallback
    `gold` may be a string or a list of aliases (trivia); the first alias found wins.
    """
    full_ids = list(prompt_ids) + list(gen_ids)
    P = len(prompt_ids)
    aliases = gold if isinstance(gold, (list, tuple)) else [gold]
    for ans in aliases:
        ans = str(ans)
        if not ans.strip():
            continue
        span = _span_indices(tokenizer, full_ids, ans, lower=P)
        if span:
            rows = [t - (P - 1) for t in span]                 # absolute -> cache row
            rows = [r for r in rows if 0 <= r <= len(gen_ids)]  # keep inside the window
            if rows:
                return rows, True
    return [], False


def last_token_row(rows, gen_len, found):
    """The exact-answer-last-token cache row (their primary position). Fallback when not found:
    the last answer token (cache row gen_len), and the caller logs the fallback."""
    if found and rows:
        return min(gen_len, rows[-1])
    return gen_len  # last answer token (row 0 = last-prompt token, rows 1..G = answer tokens)
