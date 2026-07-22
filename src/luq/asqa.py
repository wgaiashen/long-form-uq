"""ASQA loader for the long-form UQ pipeline (a second FACTUALITY task, CLOSED-BOOK).

ASQA (Stelmakh et al., din0s/asqa on HF): ambiguous factoid questions, each paired with human-written
disambiguating long answers plus a set of disambiguated (question, short_answers) qa_pairs. We run the
model CLOSED-BOOK (question only, no retrieved passages) so ASQA stays a FACTUALITY test of world
knowledge, not a RAG/faithfulness test. This patterns with biographies / ExpertQA, not with Xsum/PubMedQA.

Gold (instance-level, for the LLM judge): the TWO annotator long_answers JOINED. ASQA gives exactly two
long_answers per question, same facts in different wording (like TriviaQA aliases), so concatenating both
avoids penalising a model answer that matches the second annotator's phrasing (decision confirmed with
the author 2026-07-21). The qa_pairs' short_answers are kept per record for the FREE Str-EM coverage
cross-check (labels/asqa_strem), not the primary label.

Eval-only (like ExpertQA): ASQA is a NEW factuality EVAL target for the cross-task OOD ladder — probes
are trained on the existing pool and tested on ASQA — so there is no ASQA train split. We use the full
dev split (948), in its natural (deterministic) order, so record idx aligns to load_records() for the
Str-EM re-join with no manifest needed.

Prompt + budget are frozen here (design-once; the frozen prompt stabilises 01_extract's prompt_hash
guard). See the worklog 2026-07-21 entry for the pilot GATE that justified building this set.
"""
HF_NAME = "din0s/asqa"
SPLIT = "dev"

# FROZEN prompt wording (same as the W8 pilot that passed the gate). Closed-book: the ambiguous
# question only, asking for a paragraph that resolves the ambiguity — the ambiguity instruction suits
# ASQA (questions are ambiguous by construction) and is what the pilot's healthy Str-EM spread was
# measured on. Carries the Question:/Answer: markers the judge parser reads.
PROMPT = ("Answer the following question with a detailed, self-contained paragraph. If the question is "
          "ambiguous, cover each distinct correct interpretation.\n\nQuestion: {question}\nAnswer:")

# Generation budget: gold(joined) Llama-tokenizer p50=168, p90=252, p95=283. 256 covers ~p90 of natural
# answer length, giving room to cover each interpretation while staying a cap, not a target. Paired with
# --repetition-penalty 1.2 (the ExpertQA-proven fix) to hold base-Llama looping down at this budget.
MAX_NEW_TOKENS = 256


def _short_answers(qa):
    """Robust to field-name variants across ASQA HF ports (din0s/asqa uses 'short_answers')."""
    for k in ("short_answers", "short_answer", "answers", "answer"):
        v = qa.get(k)
        if v:
            return v if isinstance(v, list) else [v]
    return []


def load_records(split=SPLIT):
    """One dict per ASQA question, in the split's natural order so idx is stable:
    {question, gold (both long_answers joined), qa_pairs (short-answer lists for Str-EM), sample_id}."""
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=split)
    out = []
    for ex in ds:
        question = ex["ambiguous_question"]
        gold = " ".join(a["long_answer"].strip() for a in ex["annotations"] if a.get("long_answer"))
        qa_pairs = [{"short_answers": _short_answers(qa)} for qa in ex["qa_pairs"]]
        out.append({"question": question, "gold": gold, "qa_pairs": qa_pairs,
                    "sample_id": ex.get("sample_id", "")})
    return out
