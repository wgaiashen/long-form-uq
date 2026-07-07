"""Step 2 of the answer-span plan: map a character cut to a token cut, and re-pool the per-token
L15 states over the kept span -- WITHOUT re-running the model.

THE WINDOW (from scripts/01h_pertoken.py)
-----------------------------------------
Per example the cached per-token states cover the SAPLMA window `[P-1 : P+G]`:
  row 0        = the LAST PROMPT token (P-1), the anchor SAPLMA also pools over
  rows 1..G    = the G answer tokens, in order (row 1+j == answer/gen token j)
The cached SAPLMA feature is the mean over ALL G+1 rows (verified: 01h's reload gate reproduces
the SAPLMA PRR), so a comparable truncated pool must also keep row 0.

TRUNCATION
----------
`answer_span` returns a CHAR offset `cut_char` into `gen_text` (== decode(gen_token_ids)). We keep
the first `cut_tok` ANSWER tokens, where cut_tok = #answer-tokens whose decoded text ends at/before
cut_char. In states-row space that is rows `[0 : cut_tok+1]` (anchor + kept answer tokens):
  trunc_mean = states[0 : cut_tok+1].mean(0)
  trunc_last = states[cut_tok]         (the last KEPT answer token; == states[0] anchor if cut_tok==0)
No regeneration: everything is a slice of the already-cached states.
"""
import numpy as np


def char_to_tok(tok, gen_ids, cut_char):
    """Number of leading answer tokens whose decoded text ends at/before `cut_char`.

    Binary-searches the cumulative decoded length (monotonic in the token count), decoding PREFIXES
    of the cached `gen_ids` so the mapping is always consistent with the exact cached tokenisation
    (re-tokenising `gen_text` is not guaranteed to round-trip). Returns an int in [0, len(gen_ids)].
    """
    G = len(gen_ids)
    if cut_char <= 0:
        return 0
    # Decode with skip_special_tokens=True to live in the SAME string space as the record's
    # gen_text (generate.py builds it that way). Otherwise a trailing EOS renders as
    # "<|end_of_text|>" and inflates the decoded length, so the no-cut sentinel (cut_char =
    # len(gen_text)) falls short of it and silently drops the final (EOS) token -- which changed
    # ~26% of rows (every naturally-stopped generation), not the ~2% actually cut.
    full = tok.decode(gen_ids, skip_special_tokens=True)
    if cut_char >= len(full):
        return G
    # largest t in [0, G] with len(decode(gen_ids[:t])) <= cut_char
    lo, hi = 0, G
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.decode(gen_ids[:mid], skip_special_tokens=True)) <= cut_char:
            lo = mid
        else:
            hi = mid - 1
    return lo


def truncated_pool(states, cut_tok):
    """Re-pool the per-token `states` (shape (G+1, hidden), row 0 = anchor) over the kept span.

    Returns (mean_vec, last_vec, kept_rows). `cut_tok` is in ANSWER-token units; the kept window is
    rows [0 : cut_tok+1] (anchor + first cut_tok answer tokens). Clamps to the available rows so a
    cut_tok past the end (no cut) keeps everything, matching the raw SAPLMA pool exactly.
    """
    n_rows = states.shape[0]                      # == G + 1
    keep = min(cut_tok + 1, n_rows)               # +1 for the row-0 anchor; clamp to available
    keep = max(keep, 1)                           # always keep at least the anchor
    window = states[:keep]
    return window.mean(axis=0), window[-1], keep


def repool_record(tok, record, states, cut_char):
    """Convenience: map this record's `cut_char` and return the raw and truncated pooled vectors.

    Returns dict with raw_mean/raw_last (full window) and trunc_mean/trunc_last (kept span) plus
    cut_tok and kept_rows, so raw and truncated live side by side (the plan: don't overwrite).
    """
    states = np.asarray(states, dtype=np.float32)
    gen_ids = list(record["gen_token_ids"])
    cut_tok = char_to_tok(tok, gen_ids, cut_char)
    t_mean, t_last, kept = truncated_pool(states, cut_tok)
    return {
        "raw_mean": states.mean(axis=0),
        "raw_last": states[-1],
        "trunc_mean": t_mean,
        "trunc_last": t_last,
        "cut_tok": int(cut_tok),
        "kept_rows": int(kept),
        "n_rows": int(states.shape[0]),
    }
