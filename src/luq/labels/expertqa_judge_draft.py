"""Three-state factuality judge for ExpertQA: each substantive claim is marked supported,
contradicted or uncovered.

This is the labelling path ExpertQA actually uses, through `scripts/02_label_expertqa.py`, and the
prompt below is the one that produced the labels behind every ExpertQA result. The filename retains
the word draft for historical reasons only.

Projection this label captures (stated explicitly, per the label-definition decision): FACTUAL
FAITHFULNESS of the response's claims, judged against the expert gold answer + its citations as a
NON-EXHAUSTIVE reference. It deliberately does NOT capture similarity-to-gold, completeness, scope,
or style — those are different projections of "good" and folding them in gives garbage labels for a
long-form answer that legitimately differs from one expert reference.

Three states per substantive claim (the key design, so uncovered fabrications are not silently
scored correct):
  - SUPPORTED   : consistent with / entailed by the reference + its evidence.
  - CONTRADICTED: conflicts with the reference + its evidence.
  - UNCOVERED   : the reference neither states nor implies it (plausible-but-unaddressed).

Outputs, as JSON:
  - factuality : SUPPORTED / (SUPPORTED + CONTRADICTED) — a PURE PROPORTION, every claim weighted
                   equally, NO severity weighting (severity is a second axis the judge applies
                   inconsistently across expert domains; add it later with data if proportion misses
                   catastrophic-error-buried-in-trivial-truths). Different denominator from uncovered
                   (covered claims only). Judge-knowledge-INDEPENDENT — it never asks the judge to
                   know the truth of an uncovered claim, so it stays robust on the expert long tail.
  - uncovered    : UNCOVERED / (all substantive claims). RECORDED, not folded into factuality — it
                   quantifies the benchmark's blind spot (how much of the model's output that
                   factuality-to-evidence cannot see). Different denominator, do not co-normalise.
  - coherent     : false if the answer collapses into incoherent / non-language text, word-salad,
                   list-spam, or code (the degenerate-tail rule). coherent=false FORCES
                   factuality=0.0 and uncovered=0.0, regardless of a good opening.

STAMPED downstream label decision (uniform, so no discontinuity in the trust signal the probe learns):
degenerate generations are a DISTRUST LABEL, NOT dropped. Both severities get the same treatment:
  - SEVERE (detector-quarantined pre-judge, luq.degeneracy.is_severe): factuality=0.0, not judged.
  - coherent=false (marginal survivor that opened clean then derailed on a hard question): factuality=0.0.
Quarantined-severe and coherent-false-marginal are the SAME failure mode at different severities;
dropping one and distrust-labelling the other would put a discontinuity into exactly the region of
the trust signal the probe is learning. So both -> distrust label (factuality=0.0), uniformly.

Judge model: re-validate gpt-5-mini vs gpt-5 on THIS prompt before adopting (the old r=0.83 was on
the old gold-matching prompt); use one judge for both train and eval.
"""

# The uncovered branch is the one the judge naturally gets wrong (it resolves plausible-but-
# unaddressed claims into "supported" because they read fine), so the instruction hammers it and
# few-shot #3 is built entirely around it.
PROMPT = """You are judging the FACTUAL CORRECTNESS of a model's answer to an expert question.

You are given the question, the model answer, and an expert-written REFERENCE answer with its cited
evidence. The reference is correct but NOT exhaustive: a good model answer may be longer, organised
differently, and cover additional or different valid points. Do NOT reward similarity to the
reference, and do NOT penalise differences in scope, length, wording, structure, or an answer being
less complete than the reference. Judge only whether the model's factual claims hold up.

Sort the model answer's substantive factual claims into three states, using ONLY the reference and
its evidence — never your own outside knowledge — to decide:
- SUPPORTED: the reference or its evidence states or clearly implies the claim.
- CONTRADICTED: the reference or its evidence conflicts with the claim.
- UNCOVERED: the reference neither states, implies, nor contradicts the claim. A claim that is
  plausible, fluent, and on-topic but that the reference simply does not address is UNCOVERED, NOT
  supported. Do not count a claim as supported merely because it sounds correct.

If the reference answer contains non-substantive artifacts (leaked file paths, "the source cannot be
found", editorial notes like "claim not needed"), disregard them and judge against the substantive
reference claims and citations only.

If the model answer collapses into incoherent or non-language text, word-salad, list-spam, or code —
even after a coherent opening — mark it not coherent: such an answer is untrustworthy as a whole. If
coherent is false, factuality must be 0.0 and uncovered must be 0.0, regardless of the answer's opening.

Compute factuality as a PURE PROPORTION: SUPPORTED / (SUPPORTED + CONTRADICTED), weighting every
claim equally with no severity weighting. For example, an answer with one supported and one
contradicted claim scores 1 / (1 + 1) = 0.5. Factuality is computed ONLY over SUPPORTED +
CONTRADICTED claims; uncovered is computed over ALL substantive claims. These are different
denominators — do not normalise them together. If there are NO supported and NO contradicted claims
(every substantive claim is uncovered), factuality is undefined — set it to null.

Respond with ONLY a JSON object, no explanation:
{"factuality": <0.0-1.0 = SUPPORTED / (SUPPORTED + CONTRADICTED); null if no covered claims>,
 "uncovered": <0.0-1.0 = UNCOVERED / (all substantive claims)>,
 "coherent": <true|false>}

Examples:

Question: What are first-line treatment options for early-stage Hodgkin lymphoma?
Reference: Early-stage Hodgkin lymphoma is usually treated with combination chemotherapy (commonly
ABVD) together with involved-site radiotherapy; the number of cycles depends on risk group. [Evidence: treatment guideline excerpt]
Model Answer: Early-stage Hodgkin lymphoma is generally managed with ABVD chemotherapy, often
followed by radiation to the involved region. The exact number of cycles is chosen by risk group.
{"factuality": 1.0, "uncovered": 0.0, "coherent": true}

Question: What are the main functions of the liver?
Reference: The liver detoxifies metabolites, synthesises proteins such as albumin and clotting
factors, and produces bile that aids digestion. [Evidence: physiology textbook excerpt]
Model Answer: The liver produces bile that helps digestion. It plays no role in protein synthesis.
{"factuality": 0.5, "uncovered": 0.0, "coherent": true}

Question: What factors drive antibiotic resistance in hospital settings?
Reference: Antibiotic resistance in hospitals is driven mainly by over-prescription of broad-spectrum
antibiotics and by poor hand-hygiene compliance enabling transmission. [Evidence: infection-control review]
Model Answer: Hospital antibiotic resistance is driven by over-prescription of broad-spectrum agents.
It is also caused primarily by contaminated hospital water systems, which are the single largest
reservoir of resistant organisms in most hospitals.
{"factuality": 1.0, "uncovered": 0.5, "coherent": true}

Question: How does the Coriolis effect influence large-scale weather systems?
Reference: The Coriolis effect deflects moving air to the right in the Northern Hemisphere and to the
left in the Southern Hemisphere, giving cyclones their rotation. [Evidence: atmospheric-science text]
Model Answer: The Coriolis effect deflects moving air and shapes cyclone rotation refrigerators
washing machines dishwashers ovens toasters kettles blenders microwaves freezers.
{"factuality": 0.0, "uncovered": 0.0, "coherent": false}

Question: %%QUESTION%%
Reference: %%REFERENCE%%
Model Answer: %%ANSWER%%
"""


def fill(question: str, reference: str, answer: str) -> str:
    """Fill the template. Uses str.replace (NOT str.format) because the prompt contains literal
    JSON braces {"factuality": ...} in its examples, which str.format would misparse as fields."""
    return (PROMPT.replace("%%QUESTION%%", question)
                  .replace("%%REFERENCE%%", reference)
                  .replace("%%ANSWER%%", answer))


def build_reference(gold: str, evidence: list[str] | None = None, max_ev_chars: int = 400) -> str:
    """Assemble the reference block: gold answer + a trimmed slice of its cited evidence."""
    ref = gold.strip()
    if evidence:
        joined = " | ".join(e.strip() for e in evidence if e and e.strip())
        if joined:
            ref += f"\n[Evidence: {joined[:max_ev_chars]}]"
    return ref


def parse(text: str):
    """Parse the judge's JSON. Returns dict or None if malformed."""
    import json
    import re
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        fraw = d["factuality"]
        f = None if fraw is None else float(fraw)   # null = undefined (all claims uncovered)
        u = float(d["uncovered"]); c = bool(d["coherent"])
    except (KeyError, ValueError, TypeError):
        return None
    if f is not None and not (0.0 <= f <= 1.0):
        return None
    if not (0.0 <= u <= 1.0):
        return None
    return {"factuality": f, "uncovered": u, "coherent": c}
