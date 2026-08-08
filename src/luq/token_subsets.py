"""Token-subset masks over a generation's tokens -- shared by the MSP ablations and the constrained
weighted-MSP variants (Joe's #4: restrict the learned weighting to fewer, more meaningful tokens).

Lifted out of `scripts/checks/msp_ablations.py` (which is a script, not importable) so both the ablation
battery and `weighted_msp` can use it. Adds a punctuation-only exclusion and a NEGATION carve-back (not/no/
nor flip meaning, so they are kept even though they are stop-words).

`keep_mask(gen_token_ids, pieces, mode)` returns a 0/1 float array over the G generated tokens -- the set
the weighted-MSP softmax is allowed to distribute over (the `keep=` channel in weighted_msp). Modes stack:
  special        exclude only special/reserved tokens (Llama-3 ids >= 128000: <|begin/end_of_text|>)
  special_punct  + exclude punctuation-only / whitespace-only pieces
  content        + exclude stop-words (word-grouped), NEGATION carved back in
"""
import string

import numpy as np

SPACE_MARKERS = ("Ġ", "▁")          # HF byte-level / sentencepiece leading-space glyphs
NEWLINE_GLYPHS = ("Ċ", "ċ", "ĉ")    # byte-level newline / tab glyphs
SENT_END = {".", "!", "?"}
_SPECIAL_ID_MIN = 128000            # Llama-3 reserved/special token range
# ⚠️ LLAMA-3 ONLY, AND A SILENT BUG ON ANY OTHER MODEL (2026-08-08). Qwen2.5's vocabulary is 152,064
# with its specials at 151,643+, so `id >= 128000` would classify a large band of ORDINARY CONTENT
# TOKENS as special and drop them from the keep-mask -- no crash, just a quietly different method.
# Mirrors the fix already made in `weighted_msp.py`; `set_special_ids(tok.all_special_ids)` below
# registers the real ids, and the default preserves the Llama behaviour exactly so no committed
# number moves. Verified on Llama: the only id >= 128000 ever occurring in a cached generation is
# 128001 (<|end_of_text|>), which IS in all_special_ids, so set-membership and the >= test agree.
_SPECIAL_IDS = None                 # None => fall back to the reserved-range test above


def set_special_ids(ids):
    """Register the tokenizer's special ids for the model being run. Pass `tok.all_special_ids`.

    Call once at driver startup for any non-Llama model; None restores the Llama-3 default. Module
    state rather than a new argument, because `keep_mask` is called from several places and changing
    its signature would break them.
    """
    global _SPECIAL_IDS
    _SPECIAL_IDS = None if ids is None else {int(i) for i in ids}


def _is_special(t):
    return int(t) >= _SPECIAL_ID_MIN if _SPECIAL_IDS is None else int(t) in _SPECIAL_IDS

# Negation words: in STOPWORDS but meaning-bearing -- keep them in the "content" set.
NEGATION = {"not", "no", "nor", "never", "none", "cannot", "n't", "without", "neither"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "of",
    "is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "having", "do",
    "does", "did", "doing", "would", "should", "could", "ought", "will", "shall", "can", "may", "might",
    "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "this", "that", "these", "those", "who", "whom", "which", "what",
    "as", "so", "than", "too", "very", "just", "not", "no", "nor", "only", "own", "same", "such", "s",
    "t", "d", "ll", "m", "re", "ve", "there", "here", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "how", "why", "where",
}


def _strip_glyphs(p):
    for mk in SPACE_MARKERS + NEWLINE_GLYPHS:
        p = p.replace(mk, "")
    return p.strip()


def is_punct_piece(piece):
    """A decoded token piece that is punctuation-only or whitespace/newline-only (no content)."""
    s = _strip_glyphs(piece)
    if s == "":
        return True                                        # whitespace / newline only
    return all(c in string.punctuation for c in s)


def word_groups(pieces):
    """Group token positions into words. A word starts at a piece that begins with a space glyph (or the
    first token). Returns [(positions, lowercased_glyph_stripped_word_text)]."""
    groups, cur_pos, cur_txt = [], [], ""
    for i, p in enumerate(pieces):
        if (i == 0 or p.startswith(SPACE_MARKERS)) and cur_pos:
            groups.append((cur_pos, cur_txt)); cur_pos, cur_txt = [], ""
        cur_pos.append(i)
        cur_txt += _strip_glyphs(p) if not p.startswith(SPACE_MARKERS) else p
        for mk in SPACE_MARKERS:
            cur_txt = cur_txt.replace(mk, "")
    if cur_pos:
        groups.append((cur_pos, cur_txt))
    return [(pos, txt.lower()) for pos, txt in groups]


def keep_mask(gen_token_ids, pieces, mode="content"):
    """0/1 float keep-mask over the G tokens for the `keep=` channel of weighted_msp. See module docstring
    for the modes. Never returns an all-zero mask (falls back to all-ones for a degenerate generation)."""
    g = len(gen_token_ids)
    keep = np.ones(g, dtype=np.float32)
    for i, t in enumerate(gen_token_ids):                  # always drop special/reserved tokens
        if _is_special(t):
            keep[i] = 0.0
    if mode == "special":
        return _guard(keep)
    for i, p in enumerate(pieces):                         # drop punctuation / whitespace pieces
        if is_punct_piece(p):
            keep[i] = 0.0
    if mode == "special_punct":
        return _guard(keep)
    if mode == "content":                                  # drop stop-words (negation carved back)
        for pos, txt in word_groups(pieces):
            core = txt.strip("".join(SENT_END) + ",;:\"'()")
            if core in STOPWORDS and core not in NEGATION:
                for j in pos:
                    keep[j] = 0.0
        return _guard(keep)
    raise ValueError(f"mode must be special|special_punct|content, got {mode!r}")


def _guard(keep):
    """A generation that ends up all-excluded (e.g. pure punctuation) falls back to all-ones so the
    softmax does not see an all -inf row."""
    return keep if keep.sum() > 0 else np.ones_like(keep)
