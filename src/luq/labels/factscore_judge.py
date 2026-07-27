"""DRAFT — three-state factuality judge for FActScore-Bio (NOT wired / NOT run until the prompt is cleared).

FActScore-Bio is ExpertQA's SAME-TASK factuality partner, so it uses the SAME three-state judge and the
SAME score definition (factuality = SUPPORTED/(SUPPORTED+CONTRADICTED) = factual PRECISION, which is
exactly what FActScore measures) — only the framing changes from "expert question + expert reference answer"
to "biography + the person's Wikipedia article as the reference". Keeping the judge identical is deliberate:
it makes ExpertQA and FActScore-Bio directly comparable (one yardstick, per the no-mixed-judges rule).

Key difference from ExpertQA's reference: Wikipedia is a fairly COMPREHENSIVE single-page reference (vs
ExpertQA's one non-exhaustive expert answer), so expect a LOWER `uncovered` fraction here. The judge still
uses ONLY the reference (never its own outside knowledge) — the FActScore design (verify each claim against
the person's Wikipedia page). Reference = `documents.text` for the entity title (schema verified:
`documents(title PRIMARY KEY, text)`, plain prose with a leading `<s>` sentinel + the title).

parse() is reused unchanged from the ExpertQA judge (same JSON contract). Re-validate gpt-5-mini vs gpt-5 on
this prompt before adopting, as for ExpertQA.
"""

PROMPT = """You are judging the FACTUAL CORRECTNESS of a model-generated BIOGRAPHY of a person.

You are given the person's name, the model's biography, and that person's WIKIPEDIA ARTICLE as the
REFERENCE. The reference is a factual, fairly comprehensive account, but it may still not mention every
detail. Do NOT reward similarity to the reference, and do NOT penalise the biography for being organised
differently, shorter, or less complete than the article. Judge only whether the biography's factual claims
hold up against the reference.

Sort the biography's substantive factual claims into three states, using ONLY the Wikipedia reference —
never your own outside knowledge — to decide:
- SUPPORTED: the reference states or clearly implies the claim.
- CONTRADICTED: the reference conflicts with the claim.
- UNCOVERED: the reference neither states, implies, nor contradicts the claim. A claim that is plausible,
  fluent, and on-topic but that the reference simply does not address is UNCOVERED, NOT supported. Do not
  count a claim as supported merely because it sounds correct.

If the model biography collapses into incoherent or non-language text, word-salad, list-spam, or code — even
after a coherent opening — mark it not coherent: such an answer is untrustworthy as a whole. If coherent is
false, factuality must be 0.0 and uncovered must be 0.0, regardless of the answer's opening.

Compute factuality as a PURE PROPORTION: SUPPORTED / (SUPPORTED + CONTRADICTED), weighting every claim
equally with no severity weighting. For example, a biography with one supported and one contradicted claim
scores 1 / (1 + 1) = 0.5. Factuality is computed ONLY over SUPPORTED + CONTRADICTED claims; uncovered is
computed over ALL substantive claims. These are different denominators — do not normalise them together. If
there are NO supported and NO contradicted claims (every substantive claim is uncovered), factuality is
undefined — set it to null.

Respond with ONLY a JSON object, no explanation:
{"factuality": <0.0-1.0 = SUPPORTED / (SUPPORTED + CONTRADICTED); null if no covered claims>,
 "uncovered": <0.0-1.0 = UNCOVERED / (all substantive claims)>,
 "coherent": <true|false>}

Examples:

Person: Ada Lovelace
Reference: Augusta Ada King, Countess of Lovelace (1815-1852), was an English mathematician known for her
work on Charles Babbage's Analytical Engine. She is regarded as one of the first to recognise that the
machine had applications beyond pure calculation. [...]
Model Biography: Ada Lovelace was a 19th-century English mathematician who worked with Charles Babbage on
the Analytical Engine and is often described as the first computer programmer.
{"factuality": 1.0, "uncovered": 0.0, "coherent": true}

Person: Marie Curie
Reference: Marie Curie (1867-1934) was a physicist and chemist who conducted pioneering research on
radioactivity, won Nobel Prizes in Physics (1903) and Chemistry (1911), and was the first woman to win a
Nobel Prize. [...]
Model Biography: Marie Curie was a physicist who studied radioactivity and won two Nobel Prizes. She was
born in France in 1855.
{"factuality": 0.6666666666666666, "uncovered": 0.0, "coherent": true}

Person: A minor historical figure
Reference: [short article that covers the person's birthplace and main occupation only]
Model Biography: They were born in the town named in the article and worked in the stated occupation. They
also secretly funded three expeditions to the South Pole and invented an early type of bicycle.
{"factuality": 1.0, "uncovered": 0.5, "coherent": true}

Person: Some person
Reference: [a normal biographical article]
Model Biography: This person was a notable figure born in bicycle bicycle washing machines dishwashers
ovens toasters kettles blenders microwaves freezers refrigerators.
{"factuality": 0.0, "uncovered": 0.0, "coherent": false}

Person: %%QUESTION%%
Reference: %%REFERENCE%%
Model Biography: %%ANSWER%%
"""


def fill(person: str, reference: str, answer: str) -> str:
    """str.replace (not str.format) — the prompt has literal JSON braces in its examples."""
    return (PROMPT.replace("%%QUESTION%%", person)
                  .replace("%%REFERENCE%%", reference)
                  .replace("%%ANSWER%%", answer))


def build_reference(wiki_text: str, max_chars: int = 6000) -> str:
    """The Wikipedia article for the entity as the reference. Strips the leading `<s>` sentinel and trims to
    a judge-sized window (the lead + early sections carry the biographical facts; a full article can be huge).
    Verified schema: documents.text is plain prose beginning with `<s>` then the title."""
    t = wiki_text.lstrip()
    if t.startswith("<s>"):
        t = t[3:].lstrip()
    return t[:max_chars].strip()


# parse() is byte-identical to the ExpertQA judge's — reuse it to keep the JSON contract common.
from luq.labels.expertqa_judge_draft import parse  # noqa: E402,F401
