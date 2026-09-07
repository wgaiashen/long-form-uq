"""Orgad exact-answer location the RIGHT way (the way): extract the MODEL'S OWN short answer with an
LLM, then locate that span -- instead of string-matching the GOLD answer.

WHY (the audit fix): locating the GOLD answer makes "located" ~ the correctness label (located|correct
99%, located|incorrect 0.2%), so masked (located) rows are ~all correct and unmasked (unlocated) rows are
~all wrong -> the +/-mask comparison is confounded along the label axis (see the STEP-2 leak). the code
instead asks the model to extract ITS OWN short answer from ITS OWN generation (right or wrong), so a span
is found for correct AND incorrect rows and "located" no longer tracks correctness.

This is a faithful port of `temp_idea_1_msp_probe/extract_exact_answer/get_exact_answer.py`
(`extract_exact_answer` + `get_indices_of_exact_answer`), with two deliberate changes, both documented in
the report's method section:
  * the extraction uses an API model (gpt-5-mini) rather than a local generate() call, which is
    cheaper; the prompt is the published one verbatim (2 few-shot examples, "NO ANSWER" escape).
  * SHORT-FORM only, exactly like the reference implementation (which asserts dataset in {truthfulqa, sciq, medquad}); a long
    multi-sentence answer has no single short answer.
Validity rule is the: the extracted string must be a non-empty substring of the model answer and not
"NO ANSWER"; else the row is unlocated.
"""
import os
import time

# the extraction prompt, verbatim (extract_exact_answer, the f-string body).
EXTRACT_PROMPT = """
Extract from the following long answer the short answer, only the relevant tokens. If the long answer does not answer the question, output NO ANSWER.

The short answer ('Exact answer') must be a subset of the tokens from the short answer. If required, they can be exactly the same.

Q: Which musical featured the song The Street Where You Live?
A: The song "The Street Where You Live" is from the Lerner and Loewe musical "My Fair Lady." It is one of the most famous songs from the show, and it is sung by Professor Henry Higgins as he reflects on the transformation of Eliza Doolittle and the memories they have shared together.
Exact answer: My Fair Lady

Q: Which Swedish actress won the Best Supporting Actress Oscar for Murder on the Orient Express?
A: I'm glad you asked about a Swedish actress who won an Oscar for "Murder on the Orient Express," but I must clarify that there seems to be a misunderstanding here. No Swedish actress has won an Oscar for Best Supporting Actress for that film. The 1974 "Murder on the Orient Express" was an American production, and the cast was predominantly British and American. If you have any other questions or if there's another
Exact answer: NO ANSWER

Q: {question}
A: {model_answer}
Exact answer:"""

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set (login node; source .openai_key).")
        _client = OpenAI()
    return _client


def extract_model_answer(question, model_answer, model="gpt-5-mini", max_retries=4):
    """Ask the LLM to extract the model's OWN short answer from its generation. Returns the extracted
    string (a substring of model_answer) or "NO ANSWER". the validity rule: non-empty substring of the
    model answer, not "NO ANSWER"."""
    if not isinstance(model_answer, str) or not model_answer.strip():
        return "NO ANSWER"
    prompt = EXTRACT_PROMPT.format(question=str(question), model_answer=model_answer)
    client = _get_client()
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=1, top_p=1,
                messages=[{"role": "user", "content": prompt}])
            out = (resp.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(2)
            continue
        out = out.split("\n")[0].strip()                      # first line (the reference implementation truncates at newline)
        for junk in ("<|end_of_text|>", "<|begin_of_text|>", "Exact answer:"):
            out = out.replace(junk, "")
        out = out.strip().strip(".").strip()
        if out and out != "NO ANSWER" and out.lower() in model_answer.lower():
            return out
        if out == "NO ANSWER":
            return "NO ANSWER"
    return "NO ANSWER"


# --------------------------------------------------------------------------------------------------
# Task-adaptive summarisation variant (the ONE method extended to summaries).
# QA has a single "exact answer"; a summary has no answer, so the important-token CONCEPT becomes
# "the key information-bearing spans" (entities/numbers/claims). Same idea, task-appropriate prompt --
# exactly the QA-vs-summarisation split the response-quality judge already uses.
# --------------------------------------------------------------------------------------------------
SUMMARY_DATASETS = {"xsum", "cnn_dailymail", "samsum"}

SUMMARY_PROMPT = """Extract from the following summary the key fact-bearing terms: the specific names, places, organisations, numbers, and dates a reader would fact-check. Copy each term verbatim from the summary, one per line, and keep each term SHORT -- a name, place, organisation, or number, NOT a whole clause or sentence. If the summary states no specific facts, output NO ANSWER.

Summary: The winning Lotto ticket was bought in Merthyr Tydfil for the Team GB-inspired Medal Draw in August 2016, the National Lottery said.
Spans:
Merthyr Tydfil
Team GB
August 2016
National Lottery

Summary: Officials have announced that new measures will be introduced in due course.
Spans:
NO ANSWER

Summary: {summary}
Spans:"""


def is_summarisation(dataset):
    return dataset in SUMMARY_DATASETS


def extract_summary_spans(summary, model="gpt-5-mini", max_retries=4):
    """Important-token extraction for SUMMARISATION: ask the LLM for the key information-bearing spans
    (entities/numbers/claims) in the model's summary. Returns a list of validated spans (each a
    non-empty substring of the summary; possibly empty). Same validity rule as extract_model_answer."""
    if not isinstance(summary, str) or not summary.strip():
        return []
    prompt = SUMMARY_PROMPT.format(summary=summary)
    client = _get_client()
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=1, top_p=1,
                messages=[{"role": "user", "content": prompt}])
            out = (resp.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(2)
            continue
        if out.upper().startswith("NO ANSWER"):
            return []
        spans = []
        for line in out.splitlines():
            s = line.strip().lstrip("-*0123456789. ").strip()
            for junk in ("Spans:", "Exact answer:"):
                s = s.replace(junk, "")
            s = s.strip().strip(".").strip()
            if s and s != "NO ANSWER" and s.lower() in summary.lower():
                spans.append(s)
        return spans
    return []


# --------------------------------------------------------------------------------------------------
# Refined LONG-FORM QA variant (the "broad prompt"). The short-form exact-answer prompt is
# degenerate on multi-sentence QA answers -- on pubmed it collapses to the yes/no verdict for 73% of
# examples, on med_quad it fails (NO ANSWER) 42% of the time. A long QA answer has no single exact answer:
# the important-token CONCEPT becomes the SET of claim-bearing spans (the verdict PLUS the findings/
# entities/numbers), exactly like the summary variant but keeping the yes/no verdict. Returns a list.
# --------------------------------------------------------------------------------------------------
# A DATASET MISSING FROM THIS SET DOES NOT FAIL -- IT SILENTLY GETS THE WRONG PROMPT. `extract_important`
# falls through to `extract_model_answer`, the SHORT-ANSWER prompt, which asks for "the short answer" to a
# question. On a biography or a long-form QA answer that mostly returns "NO ANSWER", so the run completes,
# costs real money, and writes a cache that looks fine and is nearly empty. This was caught before
# spending on factscore/asqa. Anything long-form and claim-bearing belongs here.
#   asqa      -- long-form QA, answers carry multiple verifiable claims
#   factscore -- biographies; not literally "QA", but the claim-span prompt (entities, dates, numbers) is
#                exactly right for them, and the short-answer prompt is exactly wrong
LONGFORM_QA_DATASETS = {"pubmed_qa", "med_quad", "expertqa", "asqa", "factscore"}

# DOMAIN-NEUTRAL BY DESIGN. Naming medical entity types in the prompt
# explicitly -- "drugs, genes, conditions, procedures" -- and BOTH few-shot examples were biomedical
# (DMSO/telomerase, ETHE1). That was written when this path served pubmed_qa and med_quad only. Applied to
# a BIOGRAPHY (factscore) it asks for drugs and genes when the claim-bearing terms are names, dates, places
# and roles; applied to open-domain QA (asqa) it steers the same way. In-context examples shape the output
# strongly, so this was not cosmetic -- it would have produced systematically wrong spans on every
# non-medical dataset, and expertqa (CROSS-DOMAIN expert QA) had already been extracted under it.
# The entity list is now generic and the three examples span biomedical / biographical / general knowledge,
# so no single domain dominates the demonstration.
LONGFORM_QA_PROMPT = """Extract from the following answer the claim-bearing terms a reader would need to verify to judge whether the answer is correct: any explicit yes/no verdict, the key findings or conclusions, and the specific entities (people, places, organisations, works, substances, or other named things), numbers, and dates. Copy each term verbatim from the answer, one per line, and keep each term SHORT -- a word or short phrase, NOT a whole sentence. If the answer states nothing verifiable, output NO ANSWER.

Question: Does dimethyl sulfoxide (DMSO) cause a reversible inhibition of telomerase activity?
Answer: Yes, DMSO causes a reversible inhibition of telomerase activity in a Burkitt lymphoma cell line.
Spans:
Yes
reversible inhibition
telomerase activity
Burkitt lymphoma

Question: Tell me a bio of Ada Lovelace.
Answer: Ada Lovelace was an English mathematician born in 1815, the daughter of the poet Lord Byron. She worked with Charles Babbage on the Analytical Engine and is often described as the first computer programmer.
Spans:
English
mathematician
1815
Lord Byron
Charles Babbage
Analytical Engine
first computer programmer

Question: Where was the 1936 Summer Olympics held?
Answer: The 1936 Summer Olympics were held in Berlin, Germany, from 1 to 16 August 1936. They were the first Olympics to be televised.
Spans:
1936 Summer Olympics
Berlin
Germany
1 to 16 August 1936
first Olympics to be televised

Question: {question}
Answer: {model_answer}
Spans:"""


# PER-DOMAIN PROMPT SPLIT. The claim-span prompt comes in two forms and
# the dataset chooses. This is a DELIBERATE, RECORDED inconsistency, not an accident, and it must be stated
# in any table caption that uses Orgad masks:
#   MEDICAL_QA_DATASETS  -> the biomedical prompt (drugs/genes/conditions/procedures, biomedical examples).
#                           Correct for pubmed_qa and med_quad, which ARE biomedical, and their existing
#                           masks were built with it, so they need no re-extraction.
#   everything else      -> the domain-neutral prompt (people/places/organisations/works/substances, with
#                           biomedical + biographical + general-knowledge examples).
# expertqa is NOT medical -- it is multi-domain expert QA (law, engineering, healthcare, and more) -- so
# it moves to the neutral prompt and its old masks are DISCARDED and re-extracted. Extracting cross-domain
# answers under a prompt that names only biomedical entity types would bias every span it produced.
MEDICAL_QA_DATASETS = {"pubmed_qa", "med_quad"}


def is_medical_qa(dataset):
    return dataset in MEDICAL_QA_DATASETS


def is_longform_qa(dataset):
    return dataset in LONGFORM_QA_DATASETS


# The ORIGINAL biomedical form, kept verbatim so pubmed_qa/med_quad masks stay reproducible from the code
# that made them. Do NOT "improve" it -- their cached masks were extracted under exactly this text.
MEDICAL_QA_PROMPT = """Extract from the following answer the claim-bearing terms a reader would need to verify to judge whether the answer is correct: any explicit yes/no verdict, the key findings or conclusions, and the specific entities (drugs, genes, conditions, procedures), numbers, and dates. Copy each term verbatim from the answer, one per line, and keep each term SHORT -- a word or short phrase, NOT a whole sentence. If the answer states nothing verifiable, output NO ANSWER.

Question: Does dimethyl sulfoxide (DMSO) cause a reversible inhibition of telomerase activity?
Answer: Yes, DMSO causes a reversible inhibition of telomerase activity in a Burkitt lymphoma cell line.
Spans:
Yes
reversible inhibition
telomerase activity
Burkitt lymphoma

Question: What causes ethylmalonic encephalopathy?
Answer: Mutations in the ETHE1 gene cause ethylmalonic encephalopathy. The ETHE1 enzyme is a mitochondrial protein involved in the metabolism of sulfur-containing amino acids.
Spans:
ETHE1 gene
ethylmalonic encephalopathy
mitochondrial protein
sulfur-containing amino acids

Question: {question}
Answer: {model_answer}
Spans:"""


def extract_longform_qa_spans(question, model_answer, model="gpt-5-mini", max_retries=4, medical=False):
    """Refined important-token extraction for LONG-FORM QA: the SET of claim-bearing spans (verdict +
    findings + entities + numbers). Returns a list of validated spans (each a non-empty substring of the
    model answer; possibly empty). Same validity rule as the summary/exact-answer variants."""
    if not isinstance(model_answer, str) or not model_answer.strip():
        return []
    tmpl = MEDICAL_QA_PROMPT if medical else LONGFORM_QA_PROMPT
    prompt = tmpl.format(question=str(question), model_answer=model_answer)
    client = _get_client()
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=1, top_p=1,
                messages=[{"role": "user", "content": prompt}])
            out = (resp.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(2)
            continue
        if out.upper().startswith("NO ANSWER"):
            return []
        spans = []
        for line in out.splitlines():
            s = line.strip().lstrip("-*0123456789. ").strip()
            for junk in ("Spans:", "Exact answer:", "Answer:"):
                s = s.replace(junk, "")
            s = s.strip().strip(".").strip()
            if s and s != "NO ANSWER" and s.lower() in model_answer.lower():
                spans.append(s)
        return spans
    return []


def extract_important(dataset, question, model_answer, model="gpt-5-mini", variant="exact"):
    """Task-adaptive dispatch. Returns the value to cache (str for short QA, list for the span variants).
      * summarisation                     -> key fact-bearing spans (list)
      * long-form QA + variant="broad"    -> the refined claim-bearing span SET (list)
      * short QA (or long QA, variant="exact") -> the single exact answer (str)
    variant="exact" preserves the original behaviour; only long-form QA under variant="broad" changes."""
    if is_summarisation(dataset):
        return extract_summary_spans(model_answer, model=model)
    if variant == "broad" and is_longform_qa(dataset):
        return extract_longform_qa_spans(question, model_answer, model=model, medical=is_medical_qa(dataset))
    return extract_model_answer(question, model_answer, model=model)


def locate_important_rows(tokenizer, gen_ids, cached):
    """Locate cached important tokens -- a str (QA exact answer) OR a list (summary spans) -- as
    per-token-cache ROW indices, unioned across spans. Returns (rows, found)."""
    if isinstance(cached, list):
        allrows, found = [], False
        for span in cached:
            rows, f = locate_extracted_rows(tokenizer, gen_ids, span)
            if f:
                allrows.extend(rows)
                found = True
        return sorted(set(allrows)), found
    return locate_extracted_rows(tokenizer, gen_ids, cached)


def locate_extracted_rows(tokenizer, gen_ids, extracted):
    """Locate the EXTRACTED answer's token span in the generated tokens, mapped to per-token-cache ROW
    indices (window [P-1:P+G], so gen token j -> row j+1). Faithful to the get_indices_of_exact_answer
    binary search, run over the generation only. Returns (rows, found)."""
    if not extracted or extracted == "NO ANSWER":
        return [], False
    gen_ids = list(gen_ids)
    full = tokenizer.decode(gen_ids, skip_special_tokens=True)
    idx = full.lower().find(extracted.lower().strip())
    if idx == -1:
        return [], False
    true_ans = full[idx: idx + len(extracted)]
    if true_ans not in full:
        return [], False
    higher = len(gen_ids) - 1
    lo = 0
    while true_ans in tokenizer.decode(gen_ids[lo: higher + 1], skip_special_tokens=True):
        higher -= 1
    higher += 1
    while true_ans in tokenizer.decode(gen_ids[lo: higher + 1], skip_special_tokens=True):
        lo += 1
    lo -= 1
    if lo > higher or lo < 0:
        return [], False
    rows = [t + 1 for t in range(lo, higher + 1)]              # gen token t -> cache row t+1
    return rows, True
