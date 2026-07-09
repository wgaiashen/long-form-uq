"""SAR / TokenSAR token relevance (Duan et al., "Shifting Attention to Relevance").

A token's relevance is how much the meaning of the whole answer changes when you remove it:

    R(z_i) = 1 - sim( q+s , q+s\\{z_i} )        then   R~(z_i) = R(z_i) / sum_n R(z_n)

where s\\{z_i} is the generation with one unit removed, `sim` is a cross-encoder regression score
(`cross-encoder/stsb-roberta-large`), and the prompt/question q is prepended to BOTH members of the
pair. Load-bearing tokens change the meaning a lot; filler does not. This is an UNSUPERVISED,
task-agnostic "which tokens matter" -- the long-form generalisation of Orgad's short-form answer span.

Two uses (Track A / Idea 2):
  * `tokensar_score`  -- the faithful unsupervised scalar: E(z_i) = -log p(z_i)*R~(z_i),
    TokenSAR = sum_i E(z_i). Higher = more uncertain. (baseline, sits beside MSP.)
  * the normalised relevance R~ as a per-token importance WEIGHT fed into weighted-MSP as a soft mask
    (the contribution -- unsupervised aggregation vs the learned attention pooler).

REMOVAL GRANULARITY (the long-form adaptation the plan calls for):
  * "token"    -- leave-one-token-out on the token ids (faithful to lm-polygraph's TokenSAR). Right for
                  SHORT-form; on long text removing a single sub-word barely changes meaning (near-uniform
                  R, and G cross-encoder passes = slow), so it is NOT used for long-form.
  * "sentence" -- leave-one-SENTENCE-out (our long-form adaptation): far fewer, more meaningful removals;
                  each token inherits its sentence's relevance. ~n_sentences passes per example.

FAITHFULNESS (backed by runnable checks, not prose):
  * The TOKEN-level relevance is numerically verified against lm-polygraph's TokenSAR by
    `scripts/checks/check_sar_vs_lmpolygraph.py` (worst |delta| 2.4e-8). It follows lm-polygraph's
    token-id leave-one-out -- this is a DEVIATION from the SAR authors' LITERAL code
    (`SAR/src/get_tokenwise_importance.py`), which uses substring-replace
    `generated_text.replace(tokenizer.decode(token), '')`. "Faithful to SAR" is therefore only true
    w.r.t. lm-polygraph; the deviation is recorded in the report's method section.
  * The SENTENCE-level variant is OUR OWN invention (no reference exists); its token->sentence mapping
    is validated by `scripts/checks/check_sar_sentence_mapping.py` (every token -> one sentence, ids
    non-decreasing, content jaccard ~0.99 vs a regex split).
We prepend the prompt to both members and force special tokens to relevance 0 (== lm-polygraph).
"""
import itertools
import re

import numpy as np


def load_cross_encoder(name="cross-encoder/stsb-roberta-large", device="cpu"):
    """The semantic-similarity model. Downloaded once (~1.3GB) to HF_HOME; runs on CPU (short-form) or
    GPU (long-form / large sets)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device=device)


def _normalise(R):
    R = np.clip(np.asarray(R, dtype=float), 0.0, None)
    s = R.sum()
    return R / s if s > 0 else np.full(len(R), 1.0 / max(len(R), 1))


# --- sentence segmentation + token->sentence map (char-offset based, no extra deps) -----------------

def _sentence_char_spans(text):
    """List of (start, end) char spans, one per sentence. Splits on . ! ? and newlines, keeping any
    trailing fragment. Robust to the empty string."""
    spans, pos = [], 0
    for m in re.finditer(r"[^.!?\n]*[.!?\n]+|\S[^.!?\n]*$", text):
        if m.group().strip():
            spans.append((m.start(), m.end()))
    if not spans and text.strip():
        spans = [(0, len(text))]
    return spans


def _token_sentence_ids(tokenizer, gen_token_ids, text):
    """Assign each generated token to a sentence index, by where its decoded characters fall in `text`.
    Uses cumulative decode lengths (string ops only -- no model)."""
    spans = _sentence_char_spans(text)
    if len(spans) <= 1:
        return np.zeros(len(gen_token_ids), dtype=int), max(len(spans), 1)
    # char position of each token = midpoint of [len(decode(:i)), len(decode(:i+1)))
    sid = np.zeros(len(gen_token_ids), dtype=int)
    prev_len = 0
    for i in range(len(gen_token_ids)):
        cur = len(tokenizer.decode(gen_token_ids[: i + 1], skip_special_tokens=True))
        mid = (prev_len + cur) / 2.0
        prev_len = cur
        j = 0
        for k, (a, b) in enumerate(spans):
            if mid < b:
                j = k
                break
            j = k
        sid[i] = j
    return sid, len(spans)


# --- relevance ---------------------------------------------------------------------------------------

def token_relevance(ce, tokenizer, question, gen_token_ids, batch_size=16):
    """Per-token relevance by LEAVE-ONE-TOKEN-OUT (faithful TokenSAR; short-form). Returns (R, R_norm)
    over the G generated tokens."""
    toks = list(gen_token_ids)
    G = len(toks)
    if G == 0:
        return np.zeros(0), np.zeros(0)
    if G == 1:
        return np.array([0.5]), np.array([1.0])                 # lm-polygraph single-token convention
    full = question + " " + tokenizer.decode(toks, skip_special_tokens=True)
    cropped = list(itertools.combinations(toks, G - 1))[::-1]   # [::-1]: index i == token i removed
    pairs = [(full, question + " " + tokenizer.decode(list(t), skip_special_tokens=True)) for t in cropped]
    sims = np.asarray(ce.predict(pairs, batch_size=batch_size)).reshape(-1)
    R = 1.0 - sims
    special = set(tokenizer.all_special_ids or [])
    for i, tid in enumerate(toks):
        if tid in special:
            R[i] = 0.0                                          # special tokens -> relevance 0
    return R, _normalise(R)


def sentence_relevance(ce, tokenizer, question, gen_token_ids, gen_text=None, batch_size=16):
    """Per-token relevance by LEAVE-ONE-SENTENCE-OUT (the long-form adaptation). Each token inherits its
    sentence's relevance R_sent = 1 - sim(full, full without that sentence). Returns (R, R_norm) over the
    G generated tokens. Far cheaper + more meaningful than token removal on long text."""
    toks = list(gen_token_ids)
    G = len(toks)
    if G == 0:
        return np.zeros(0), np.zeros(0)
    text = gen_text if gen_text is not None else tokenizer.decode(toks, skip_special_tokens=True)
    sid, n_sent = _token_sentence_ids(tokenizer, toks, text)
    if n_sent <= 1:
        return np.ones(G), _normalise(np.ones(G))               # one sentence -> uniform
    full = question + " " + tokenizer.decode(toks, skip_special_tokens=True)
    pairs = []
    for j in range(n_sent):
        kept = [t for t, s in zip(toks, sid) if s != j]         # drop sentence j's tokens
        pairs.append((full, question + " " + tokenizer.decode(kept, skip_special_tokens=True)))
    sims = np.asarray(ce.predict(pairs, batch_size=batch_size)).reshape(-1)
    R_sent = np.clip(1.0 - sims, 0.0, None)
    R = np.array([R_sent[sid[i]] for i in range(G)], dtype=float)
    return R, _normalise(R)


def relevance(ce, tokenizer, question, gen_token_ids, gen_text=None, granularity="token", batch_size=16):
    """Dispatch on granularity. Returns (R, R_norm) per generated token."""
    if granularity == "sentence":
        return sentence_relevance(ce, tokenizer, question, gen_token_ids, gen_text, batch_size)
    return token_relevance(ce, tokenizer, question, gen_token_ids, batch_size)


def tokensar_score(R_norm, token_logprobs):
    """TokenSAR = sum_i (-log p_i) * R~_i. Higher = more uncertain (the unsupervised scalar baseline)."""
    nll = np.asarray([-lp for lp in token_logprobs], dtype=float)
    n = min(len(nll), len(R_norm))
    return float((nll[:n] * np.asarray(R_norm[:n], dtype=float)).sum())
