"""Whitespace-tolerant med_quad answer-span cut, kept SEPARATE from the frozen `answer_span.py`.

WHY THIS EXISTS, AND WHY IT IS NOT A CHANGE TO `answer_span.py`
-----------------------------------------------------------------
`answer_span.med_quad`'s markers are literal substring searches (`text.find("\\nQuestion:")`,
`text.find("\\nAnswer:")`, `text.find("\\nA. ")`): a newline immediately followed by the marker,
no whitespace in between. That misses an invented continuation that opens with extra blank lines
or leading spaces/indentation before the marker (e.g. `"\\n\\n  Question:"`), which is exactly the
gap the D6 eight-dataset sensitivity measured on `google/gemma-2-9b`: 18.9% of its med_quad
generations invent a follow-up `Question:` that the mechanical degeneracy detector cannot see, on
top of whatever the literal-newline rule already catches.

`answer_span.py` is FROZEN: its med_quad rule is the one already used for the two development
populations' canonical clean-span labels (Llama-3.1-8B base, Qwen2.5-14B), and both a
generation-time flag (`01_extract --truncate-answer-span`) and post-hoc analysis paths depend on
it staying exactly what it already is. Editing it in place would risk moving an already-reported,
frozen number. So this is a NEW, separate module, used only for the W-Models eight-dataset D6
extension on `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen2.5-32B` -- it never touches the
frozen module, its cache, or any canonical dev-population artifact.

CONTRACT -- identical shape to `answer_span.answer_span`, med_quad only
-------------------------------------------------------------------------
    answer_span_v2(text, dataset="med_quad", context=None) -> (clean_text, cut_char, reason)
  clean_text : text[:cut_char] stripped of trailing whitespace (== text if no cut)
  cut_char   : char index into the ORIGINAL text where the cut lands (len(text) if no cut)
  reason     : which rule fired, tagged with a `v2` prefix so it can never be confused with a
               frozen-module reason string in a provenance field, e.g. 'med_quad_v2:\\nAnswer:@123'

`dataset` must be `"med_quad"` -- anything else raises `ValueError` (fail loud, matching the
membership-check discipline `answer_span.DATASETS_WITH_RULES` documents: a silent no-cut for an
unsupported dataset is a worse failure than a crash).

WHAT'S DIFFERENT FROM THE FROZEN RULE
--------------------------------------
Only the whitespace tolerance: `\\nQuestion:` / `\\nAnswer:` / `\\nA. ` become `\\n\\s*Question:` /
`\\n\\s*Answer:` / `\\n\\s*A\\.\\s`, so a restart preceded by a blank line or leading indentation is
still caught. `\\s` matches space, tab and further newlines, so a run of blank lines before the
marker is covered too. The 2nd-occurrence-of-"Question:" and "which of the following" markers
already search for the literal substring ANYWHERE in the text (no anchoring newline), so they were
never whitespace-sensitive and are carried over unchanged. The universal soft-loop rule is reused
verbatim from `answer_span.py` (imported, not duplicated).
"""
import re

from .answer_span import _nth_occurrence, soft_loop_cut

MED_QUAD_V2_MARKERS = {
    r"\n\s*Question:": "\\n\\s*Question:",
    r"\n\s*Answer:": "\\n\\s*Answer:",
    r"\n\s*A\.\s": "\\n\\s*A\\.\\s",
}


def _first_regex(text, pattern):
    m = re.compile(pattern).search(text)
    return m.start() if m else None


def answer_span_v2(text, dataset="med_quad", context=None):
    """See module docstring. Returns (clean_text, cut_char, reason)."""
    if dataset != "med_quad":
        raise ValueError(
            f"answer_span_v2 only supports 'med_quad', got {dataset!r} -- this module exists "
            "specifically for med_quad's whitespace/indentation restart gap (see module docstring); "
            "every other dataset should use the frozen answer_span.answer_span instead."
        )
    if not text:
        return text, 0, "empty"

    reasons = {}

    sl = soft_loop_cut(text)
    if sl is not None:
        reasons["all:soft-loop"] = sl

    for pattern, label in MED_QUAD_V2_MARKERS.items():
        pos = _first_regex(text, pattern)
        if pos is not None:
            reasons[f"med_quad_v2:{label}"] = pos

    nth_q = _nth_occurrence(text, "Question:", 2)
    if nth_q is not None:
        reasons["med_quad_v2:2nd-Question:"] = nth_q

    wof = text.lower().find("which of the following")
    if wof != -1:
        reasons["med_quad_v2:which-of-the-following"] = wof

    if reasons:
        label, cut = min(reasons.items(), key=lambda kv: kv[1])
        reason = f"{label}@{cut}"
    else:
        cut, reason = len(text), "med_quad_v2:no-cut"

    clean = text[:cut].rstrip()
    return clean, cut, reason
