"""Generation-degeneracy detector for long-form outputs (the quarantine gate).

The base (non-instruct) Llama, pushed by aggressive decoding controls (repetition_penalty +
no_repeat_ngram_size), does not just loop — forbidden from repeating any n-gram, it drops into
NON-repetitive degeneration: word-salad ("ragtime blues spiritual gospel hymns folk songs"),
function-word-dropped rambling ("consisting eight hours starting early morning"), code/markup
emission, or whitespace/enumeration explosions. A repetition-only detector (distinct-n / repeated
sentence) is structurally blind to all of these, so it must NOT be used as the gate.

This detector targets the real signature — LOSS OF GRAMMATICAL STRUCTURE — via three signals,
each tuned and validated on hand-labelled ExpertQA cases (salad / code / enum vs legit numbered
lists / reference sections / prose-cut-at-cap, which it must NOT flag):

  - max_content_run: longest run of consecutive CONTENT words with no function word and no
    punctuation between them. Coherent prose breaks such runs every few words with a function
    word ("of", "the", "is") or punctuation; salad and function-word-dropped rambling do not.
    Legit numbered lists break runs with their markers/numbers, so they pass. (severe >=25; degraded >=15)
  - code density: fraction of "{}<>" chars — markup/code-specific and prose-rare (so legal prose
    with slashes/underscores is NOT flagged, unlike a broad punctuation set).
  - max whitespace gap: widest run of 2+ spaces — catches enumeration/table degeneration.

Two tiers so a config sweep can be graded against a precommitted bar, not the broken baseline:
  - SEVERE  = clear word-salad / code / enum (run>=25 or code or whitespace).
  - DEGRADED = SEVERE plus milder function-word-dropped rambling (run>=15).
"""
import re

# Closed-class function words. Coherent English places one every few tokens; word-salad and
# no-repeat-ngram-induced rambling drop them. Includes stray tokenizer fragments (s, t, re, ve, ll, d).
FUNCTION_WORDS = set("""
a an the this that these those and or but nor so yet for of to in on at by with from into over
under between among through during before after above below up down out off about as is are was
were be been being am do does did have has had can could will would shall should may might must not
no it its they them their he she his her we us our you your i me my which who whom whose what when
where why how if then than because while although though however therefore thus s t re ve ll d
""".split())


def max_content_run(text: str) -> int:
    """Longest run of consecutive content words (not function words, len>1), broken by any
    function word, punctuation, digit, or list marker."""
    toks = re.findall(r"[A-Za-z']+|[^\sA-Za-z']+|\d+", text)
    run = best = 0
    for t in toks:
        w = t.lower().strip("'")
        if re.fullmatch(r"[a-z']+", w) and w not in FUNCTION_WORDS and len(w) > 1:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def code_density(text: str) -> float:
    """Fraction of markup/code-specific chars ({}<>). Prose-rare; CSS/HTML/JS-heavy."""
    return sum(text.count(c) for c in "{}<>") / max(len(text), 1)


def max_whitespace_gap(text: str) -> int:
    """Widest run of 2+ spaces (enumeration / table degeneration)."""
    return max((len(m) for m in re.findall(r" {2,}", text)), default=0)


def classify(text: str) -> dict:
    """Return the three signals plus severe/degraded flags for one generation."""
    run = max_content_run(text)
    code = code_density(text)
    ws = max_whitespace_gap(text)
    severe = (run >= 25) or (code >= 0.005) or (ws >= 8)
    degraded = severe or (run >= 15)
    return {"max_content_run": run, "code_density": round(code, 4),
            "max_whitespace_gap": ws, "severe": severe, "degraded": degraded}


def is_severe(text: str) -> bool:
    return classify(text)["severe"]


def is_degraded(text: str) -> bool:
    return classify(text)["degraded"]
