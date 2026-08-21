"""Cut a base-Llama generation down to its REAL answer span.

WHY THIS EXISTS
---------------
Base Llama-3.1-8B (no instruct tuning) does not stop when the answer ends. In our few-shot
setup it over-generates past the real answer, and the junk DIFFERS by dataset:
  - med_quad : invents a fresh 'Question:/Answer:' Q&A, or a fake multiple-choice exam ('A. ...').
  - xsum     : rambles into a second line -- forum chatter, 'I'm not sure', a '###' header, a
               bulleted list, a username/date stamp.
  - pubmed_qa: echoes the input abstract, or soft-loops (a 4-gram repeated 3+ times).
This trailing junk pollutes THREE things: the mean-pooled hidden state, the P(True) append-token
position, and the judge label -- and because the junk pattern is dataset-specific, it can FAKE an
OOD distribution shift that is really just a generation artefact.

`answer_span(text, dataset)` returns the earliest defensible cut so we can re-pool the per-token
L15 states over the clean span and measure how much ID/OOD PRR moves. It only reads the decoded
text (+ optionally the input context for the pubmed echo FLAG); no model, no regeneration.

CONTRACT
--------
    answer_span(text, dataset, context=None) -> (clean_text, cut_char, reason)
  clean_text : text[:cut_char] stripped of trailing whitespace (== text if no cut)
  cut_char   : char index into the ORIGINAL text where the cut lands (len(text) if no cut)
  reason     : which rule fired, e.g. 'med_quad:\\nAnswer:@123', 'all:soft-loop@88',
               'xsum:first-nl@45|fallback-sentence@70', 'no-cut'; may carry a trailing
               '|ECHO-FLAG(0.86)' for pubmed (a FLAG, never a cut).

We always take the EARLIEST cut across (a) the universal soft-loop rule and (b) the
dataset-specific markers. sciq/trivia_qa get only first-newline + soft-loop, so they should
barely change -- that invariance is itself a check.
"""
import re
from collections import defaultdict

# datasets where the answer is a single short line: cut at the first newline (few-shot junk
# starts a new 'Question:'), plus the universal soft-loop rule. Should barely move the score.
SHORT_FORM = {"sciq", "trivia_qa"}

# Datasets this module has an actual cut RULE for. Everything else falls through to the universal
# soft-loop check only, and will usually return "no-cut".
#
# WHY THIS SET IS EXPORTED. As a post-hoc analysis tool, "no-cut" for an unknown dataset is a
# perfectly good answer. As a GENERATION-TIME flag (01_extract --truncate-answer-span, added
# 2026-08-02) it is a trap: you ask for the generation to be cut, silently get no cut, and the run
# looks like it worked. That is the "a silent default is worse than a crash" failure. Callers that
# depend on a cut actually existing must check membership here and fail loudly instead.
DATASETS_WITH_RULES = {"med_quad", "xsum", "pubmed_qa"} | SHORT_FORM


# ---- helpers ---------------------------------------------------------------------------

def _tok_starts(text):
    """Char offset where each whitespace-split token starts. Same tokenisation as str.split(),
    so token index i in split() maps to _tok_starts(text)[i]."""
    return [m.start() for m in re.finditer(r"\S+", text)]


def soft_loop_cut(text):
    """Earliest char offset at which a 4-gram that occurs 3+ times FIRST REPEATS (its 2nd
    occurrence). Reuses the soft-loop signal (any 4-gram count >= 3) from the gen-quality
    viewer, but returns WHERE the loop starts so we can cut there. None if no soft loop."""
    toks = text.split()
    if len(toks) < 8:                      # too short to loop meaningfully
        return None
    occ = defaultdict(list)                # 4-gram -> list of token-start indices
    for i in range(len(toks) - 3):
        occ[tuple(toks[i:i + 4])].append(i)
    starts = _tok_starts(text)
    best = None
    for idxs in occ.values():
        if len(idxs) >= 3:                 # repeated 3+ times => soft loop
            second_tok = idxs[1]           # the FIRST repeat = the 2nd occurrence
            if second_tok < len(starts):
                pos = starts[second_tok]
                best = pos if best is None else min(best, pos)
    return best


def _first_of(text, needles, start=0):
    """Earliest char offset among literal `needles` at/after `start` (None if none present)."""
    hits = [text.find(n, start) for n in needles]
    hits = [h for h in hits if h != -1]
    return min(hits) if hits else None


def _nth_occurrence(text, needle, n):
    """Char offset of the n-th (1-based) occurrence of `needle`, else None."""
    pos = -1
    for _ in range(n):
        pos = text.find(needle, pos + 1)
        if pos == -1:
            return None
    return pos


def _first_regex(text, pattern, start=0):
    """Start offset of the first match of `pattern` at/after `start`, else None."""
    m = re.compile(pattern).search(text, start)
    return m.start() if m else None


def _first_sentence_end(text):
    """End offset just after the first sentence terminator ('. ', '! ', '? ' or string end)."""
    m = re.search(r"[.!?](\s|$)", text)
    return m.end() if m else len(text)


def _min_ignore_none(*vals):
    xs = [v for v in vals if v is not None]
    return min(xs) if xs else None


# xsum trailing-junk markers
_XSUM_PHRASES = ("I'm not sure", "What is the main idea", "###")
_XSUM_BULLET = r"(?m)^\s*[•\-\*]\s"                    # a line starting with a bullet
_XSUM_USER = r"[Uu]ser\d+"                             # forum-style 'user123'
_XSUM_DATE = (r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"      # 12/03/2021 or 12-03-21
              r"|\d{4}-\d{2}-\d{2}"                    # 2021-03-12
              r"|\d{1,2}\s+(January|February|March|April|May|June|July|August|"
              r"September|October|November|December))\b")


def _pubmed_abstract(context):
    """Pull the input abstract out of a pubmed prompt (Abstract: ... up to the Question:)."""
    if not context:
        return ""
    i = context.rfind("Abstract:")
    if i == -1:
        return ""
    seg = context[i + len("Abstract:"):]
    j = seg.find("\nQuestion:")
    return (seg[:j] if j != -1 else seg).strip()


# echo = the answer COPIES the abstract. A word-type SET overlap over-fires, because a short
# answer against a long abstract shares most of its few word types with the big abstract
# vocabulary by construction (it flags grounded answers, not copies). Instead measure the
# longest CONTIGUOUS run of answer words that appears verbatim in the abstract, as a fraction
# of the answer length.
# THRESHOLD SET BY THE EXAMPLES, not by the suggested 0.5: eyeballing the matched spans showed
# that on this faithfulness task, quoting the abstract's single FINDING sentence (verbatim up to
# ~0.65 of a short answer) is the CORRECT grounded answer, not pathological echoing -- only when
# the answer is almost entirely a copied span (>=0.8, e.g. two whole abstract sentences) is it a
# real echo. The distribution is bimodal (mass sparse 0.6-0.8, then clusters near 1.0), so 0.8
# cleanly separates wholesale copies from grounded quoting. (0.5 would have flagged grounded
# answers like idx 529/1495/1760.)
ECHO_THRESHOLD = 0.8


def _verbatim_overlap(gen, ref):
    """Longest run of consecutive answer words that appears as a contiguous word sequence in
    `ref`, divided by the answer's word count (0..1). Unlike set overlap this ignores vocabulary
    that merely co-occurs and only rises when the answer literally copies a span of the abstract."""
    g = re.findall(r"\w+", gen.lower())
    r = re.findall(r"\w+", ref.lower())
    if not g or not r:
        return 0.0
    # index abstract word -> positions, so we only try to extend matches from real starts
    pos = defaultdict(list)
    for i, w in enumerate(r):
        pos[w].append(i)
    n = len(g)
    best = 0
    for i in range(n):
        if best >= n - i:            # can't beat current best from here on
            break
        for start in pos.get(g[i], ()):
            L = 0
            while i + L < n and start + L < len(r) and g[i + L] == r[start + L]:
                L += 1
            if L > best:
                best = L
    return best / n


# ---- the main function -----------------------------------------------------------------

def answer_span(text, dataset, context=None):
    """See module docstring. Returns (clean_text, cut_char, reason)."""
    if not text:
        return text, 0, "empty"

    reasons = {}                                   # rule label -> char offset (candidates)

    # (a) universal soft-loop rule (all datasets)
    sl = soft_loop_cut(text)
    if sl is not None:
        reasons["all:soft-loop"] = sl

    # (b) dataset-specific markers
    if dataset == "med_quad":
        cand = {
            "\\nQuestion:": text.find("\nQuestion:"),
            "\\nAnswer:": text.find("\nAnswer:"),
            "2nd-Question:": _nth_occurrence(text, "Question:", 2),
            "\\nA.": text.find("\nA. "),
            "which-of-the-following": text.lower().find("which of the following"),
        }
        for k, v in cand.items():
            if v is not None and v != -1:
                reasons[f"med_quad:{k}"] = v

    elif dataset == "xsum":
        nl = text.find("\n")
        # first newline is the primary cut (few-shot junk starts a new line)
        if nl != -1:
            reasons["xsum:first-nl"] = nl
        # the 'I'm not sure' / 'What is the main idea' / '###' phrases are unambiguous junk
        # wherever they appear, so let them fire anywhere.
        ph = _first_of(text, _XSUM_PHRASES)
        if ph is not None:
            reasons["xsum:phrase"] = ph
        # bullet / username / date are ONLY junk as SECOND-LINE artefacts -- a date or number can
        # sit inside a legitimate one-line summary (e.g. '...Lotto Medal Draw on 27 August 2016'),
        # so restrict these to at/after the first newline. (Fix: they were chopping real content.)
        if nl != -1:
            for k, pat in (("bullet", _XSUM_BULLET), ("user", _XSUM_USER), ("date", _XSUM_DATE)):
                p = _first_regex(text, pat, start=nl)
                if p is not None:
                    reasons[f"xsum:{k}"] = p

    elif dataset == "pubmed_qa":
        cand = {
            "\\nQuestion:": text.find("\nQuestion:"),
            "\\nAbstract:": text.find("\nAbstract:"),
        }
        for k, v in cand.items():
            if v is not None and v != -1:
                reasons[f"pubmed_qa:{k}"] = v

    elif dataset in SHORT_FORM:
        nl = text.find("\n")
        if nl != -1:
            reasons[f"{dataset}:first-nl"] = nl

    # earliest cut wins
    if reasons:
        label, cut = min(reasons.items(), key=lambda kv: kv[1])
        reason = f"{label}@{cut}"
    else:
        cut, reason = len(text), "no-cut"

    # xsum fallback: if cutting at the first newline left a too-short fragment (<4 words),
    # the real summary probably spans into the sentence -- use the first sentence instead.
    if dataset == "xsum" and cut < len(text):
        frag = text[:cut].strip()
        if len(frag.split()) < 4:
            se = _first_sentence_end(text)
            reason += f"|fallback-sentence@{se}"
            cut = se

    clean = text[:cut].rstrip()

    # pubmed echo: a FLAG, never a cut -- record when the generation verbatim-copies the abstract
    if dataset == "pubmed_qa":
        ov = _verbatim_overlap(clean or text, _pubmed_abstract(context))
        if ov > ECHO_THRESHOLD:
            reason += f"|ECHO-FLAG({ov:.2f})"

    return clean, cut, reason
