"""Short-form correctness label: string match (no model).

Used for the first end-to-end run so the label can never be the source of a bug.
Returns 1.0 if the gold answer (or any alias) appears in the model output, else 0.0.
This module is written out in full as a worked example.
"""
import string

_ARTICLES = {"a", "an", "the"}


def normalise(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    Normalising both sides means "The Sun." and "sun" match.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def match(output: str, gold) -> float:
    """gold may be a string or a list of aliases (TriviaQA gives aliases). Match any.

    Returns a float (1.0 / 0.0) so it slots into the same `correctness` column the
    graded LLM-judge label uses.
    """
    golds = gold if isinstance(gold, list) else [gold]
    out_norm = normalise(output)
    for g in golds:
        if normalise(str(g)) in out_norm:
            return 1.0
    return 0.0
