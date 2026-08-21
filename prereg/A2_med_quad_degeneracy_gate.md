# PRE-REGISTRATION — med_quad's degeneracy gate is different from the other three

**Written 2026-07-31, BEFORE med_quad is regenerated.** Registered in advance precisely because it
predicts a gate *failure* that should not be read as a failure, and that is the kind of reasoning which
is worthless if produced after seeing the result.

---

## The observation

From `scripts/checks/generation_quality.py` over `cache/records/*` (v1):

| dataset | % capped | % severe degeneracy | % degraded |
|---|---|---|---|
| **med_quad** | **97.7** | **12.33** | **12.33** |
| samsum | 65.9 | 0.72 | 0.72 |
| xsum | 30.0 | 0.00 | 0.00 |
| cnn_dailymail | 28.4 | 0.03 | 0.03 |

med_quad is an order of magnitude worse on degeneracy than the other three, **and** it is 97.7% capped.

## The consequence: the cap may be hiding loops

Put those two numbers together. **A generation cut off at 128 tokens cannot be detected as a loop that
would have run to 400.** `luq.degeneracy` detects looping via `max_content_run` (≥25 tokens severe, ≥15
degraded), so a repetition that only becomes visible after the cap is structurally undetectable in v1.

med_quad's 12.3% is therefore a **lower bound** on its true v1 degeneracy rate, and it is the only one of
the four where the cap binds hard enough for this to matter.

## REGISTERED PREDICTION AND GATE CORRECTION

**For med_quad, "degeneracy must not rise" is the WRONG gate.** Raising the cap from 128 to ~768 removes
a loop-truncator, so **med_quad's measured degeneracy rate may rise, and a rise would be the expected
consequence of removing the truncation rather than evidence the fix failed.**

**If med_quad's degeneracy rises, the conclusion is that med_quad needs BOTH a larger budget AND the
anti-loop measure** (`no_repeat_ngram_size=3`), not one or the other. That is consistent with med_quad
being the only dataset that needed the anti-loop measure in v1 in the first place
(`pbs/extract_neighbour.pbs:39-43`: the anti-loop default was 3, opt-*out* for short sets like samsum).

**This exemption applies to med_quad ONLY.** For samsum, xsum and cnn the severe rate is ≤0.72%, so
there is essentially nothing being masked by their caps, and **for them a degeneracy rise IS a genuine
failure signal** and must be treated as one.

## What would falsify the "hidden loops" explanation

If med_quad's degeneracy **stays flat or falls** at the larger budget, then the cap was not masking
anything and the 12.3% is the real rate — in which case the anti-loop measure is doing its job already
and no extra intervention is needed. Either outcome is informative; both are reportable.

## Interaction with the repetition penalty

med_quad is also the dataset where `--repetition-penalty` was previously **catastrophic**: applied over
its long few-shot context (mean prompt ≈1184 tokens, by far the longest of the four) it forced immediate
EOS and produced ~35% EMPTY generations, which is why v1 med_quad uses `no_repeat_ngram_size=3` instead.

So med_quad's regeneration must **not** simply inherit the samsum pilot's winning arm. The empty-generation
gate (≤2%) is load-bearing here, and the expected configuration is **larger budget +
`no_repeat_ngram_size=3`, without a repetition penalty** — to be confirmed by its own smoke run, not
assumed.

---

# AMENDMENT — 2026-08-02

**Logged as a deviation, not a clarification.** Everything below changes a registered comparison
*after* seeing data. It is recorded so the choice is auditable rather than merely correct.

## A1. The severe-degeneracy prediction was CONFIRMED IN NUMBER BUT NOT IN MECHANISM

**Registered:** med_quad's 12.3% severe was a lower bound because the 128 cap truncated loops before
the detector's run-length threshold could fire; a rise after uncapping would be the expected
consequence of removing a loop-truncator.

**Observed:** the rate did rise, 12.33% → 30.56%. **But the mechanism is wrong.** Breaking the flag
down by which condition fired (`severe = run≥25 or code≥0.005 or ws≥8`):

| | severe | fired by **whitespace** only | fired by **looping** (run≥25) |
|---|---|---|---|
| old (cap 128) | 222 (12.33%) | 218 — **98.2%** | **0** |
| new (cap 768) | 550 (30.56%) | 527 — **95.8%** | **3** |

**There is essentially no looping in either.** The flag is firing on the `max_whitespace_gap ≥ 8`
rule, and **med_quad's own gold reference answers trip that same rule on 24.8% of rows** — it is
detecting the indented-list formatting the references themselves use. The registered prediction was
confirmed on the headline number by a mechanism it did not name.

**Do not quote med_quad's severe rate without this decomposition.** Report `run≥25` separately.

## A2. The real failure is FEW-SHOT CONTINUATION, which the detector cannot see

Base Llama is not instruction-tuned; under a few-shot prompt it answers and then writes the *next*
`Question:/Answer:` pair itself, inventing both — often on an unrelated topic.

| | contains a fabricated `Question:` | mean real-answer length | mean fraction that is the real answer |
|---|---|---|---|
| old (cap 128) | **47.8%** | 416 chars | 0.744 |
| new (cap 768) | **92.6%** | **916 chars** | ~0.34 |

**The budget fix genuinely worked** — the real answer more than doubled. But the invented part grew
tenfold, so ~66% of the average new generation is text the model was never asked to produce, and the
judge scores the whole saved output. This is now reported by
`scripts/checks/generation_quality.py` as `%fabr` / `ansfrac`.

## A3. DEVIATION: the v1 side of the comparison changes from `correctness` to `correctness_raw`

**Originally registered:** "med_quad should improve" — comparing the regenerated dataset's judge
labels against v1's.

**The problem, found after the fact:** v1's live `correctness` is byte-identical to
`correctness_clean`, the judge on **answer-span-CUT** text (mean 0.4180). `correctness_raw`, the judge
on the **full** text, is 0.3935. They differ on 31.8% of rows (13.1% thresholded). If v2 is judged on
full text, the registered comparison mixes the generation change with a change in *what the judge was
shown*.

**Amendment:** use **`correctness_raw` (0.3935) as the v1 side**. Same judge model, same full-text
basis, already cached, zero cost. The clean/raw pair remains the archived v1 record; both stay
labelled and are never merged.

**Why this is a deviation and not a fix:** the substitution was chosen after observing the clean/raw
gap. It is defensible on its own terms — you cannot compare two judgements of different strings and
attribute the difference to the generations — but it was not the registered comparison.

## A4. OPEN, decided by one number still to land

med_quad's labels and features currently describe **different strings**: labels from the cut span,
features (`saplma`, `pertok`, `token_logprobs`) from the full text. It is the only dataset like this.
**The fabricated-`Question:` rate in the rep-pen arm selects the resolution**, and the choice must be
recorded here when it lands:

- **A** — rate falls sharply → judge the full text, no cut. Labels == features, consistent with the
  other nine, no extra work.
- **B** — rate low but nonzero → use `correctness_raw`. Labels == features, zero cost, label includes
  some fabrication.
- **C** — rate stays high → **cut the FEATURES too**, extracting hidden states over the span the judge
  sees. **A and B are not escapes at a high rate**: both then produce a label computed over text
  containing a fabricated follow-up question, and `correctness_raw` inherits that problem precisely
  because raw *is* the full-text judge. Only C actually aligns the two.

---

# AMENDMENT 2 — 2026-08-02 (later the same day)

## A5. The rep-pen arm FAILED its gate. It is not shipped.

Smoke, n=50: **%empty 30.0%** against a ≤2% bar. Mechanism confirmed rather than inferred — all 15
empty rows emitted exactly one token, `128001` = `<|end_of_text|>`. The historical ~35% failure
reproduced almost exactly, and it is why samsum's 0.0% did not transfer: the mechanism scales with
few-shot context length, and med_quad's prompt is 1184 tokens against samsum's 147.

It also moved the wrong way on both other measures: **true looping (run≥25) 5 of 50**, versus 3 of 1800
on the n-gram arm; **%fabr 56.0%** versus 47.8% at v1. The arm added to suppress repetition had the most
of it.

**Kept as evidence, not as data.** With the n-gram arm it forms the contrast that shows a 128-token cap
concealed the repetition rather than preventing it.

## A6. THE DIAGNOSIS WAS WRONG, AND THAT IS WHY EVERY DECODING FIX FAILED

Three arms, three failures: v1 truncates real answers; the n-gram arm sits exactly at the cap with 92.6%
fabrication; rep-pen empties 30% of rows.

**The problem was never repetition.** Base Llama is not instruction-tuned; under a few-shot prompt it
finishes the answer and then **continues the format**, writing a fresh `Question:` and inventing both
question and answer, often on an unrelated topic. **A repetition penalty cannot fix "invents a new
question"** — which is exactly why it perturbed the decoding without touching the mechanism.

**The fix is to stop at the boundary, which is this project's own documented convention** (short-form
cuts at the first newline *at extraction time*, so record, logprobs, features and label describe the
same text). Implemented as `--truncate-answer-span`, applied at the same site as `truncate_at_newline`,
i.e. **before** `out.scores[:n_gen]` and **before** the hidden-state pooling.

Simulated on the existing 768-token generations, then verified against the real tokenizer:

| | v1 (128) | n-gram (768) | rep-pen (768) | **768 + cut** |
|---|---|---|---|---|
| median real answer | 498 ch | — | — | **591 ch** |
| mean real answer | 415 ch | 948 of 2673 | — | **948 ch** |
| % empty | 0 | 0 | **30.0** | **0** |
| % fabricated | 47.8 | 92.6 | 56.0 | **2.0–2.3** |
| median at the cap? | yes | yes | no | **no** |
| true looping | 0 | 0.17% | 5/50 | **0.06%** |

Char→token mapping verified on 300 real generations: **overshoot ≤1 char (one token boundary), zero
undershoot**, so no real answer text is dropped.

## A7. §A4 RESOLVES TO OPTION A — and A3's deviation is RETRACTED

Cutting at generation time means labels, logprobs and features all describe the **same** text, so
**option A applies** and med_quad's uniqueness disappears. **No feature re-extraction (option C) is
needed.**

**The `correctness_raw` substitution registered in A3 is hereby RETRACTED.** It was needed only
because v2 was going to be judged on *full* text while v1's live `correctness` is the *cut*-text judge
label. With v2 generated-and-cut, **v1(cut) vs v2(cut) is like-for-like and the ORIGINALLY REGISTERED
comparison stands.** Recording the retraction rather than deleting A3: the deviation was genuinely
contemplated, and the reason it became unnecessary is itself part of the record.

## A8. Residual risks, carried not hidden

- **`answer_span` is now load-bearing at generation time**, having only ever been an analysis tool.
  The stratified 20-row spot-check must now also judge **cut quality**, not just answer quality.
- **~2% of cut generations still contain a fabricated `Question:`.** Report the data as
  low-fabrication, never as fabrication-free.
- 71 rows fall under 40 characters after cutting.
  **CORRECTED 2026-08-02 (DoC).** An earlier draft said "the same 71 as in v1", which is wrong as
  worded and would have overstated the case. Verified: **v1 RAW has ZERO rows under 40 characters**;
  it is **v1 after the SAME cut** that has 71, and those are the identical 71 (overlap 71, neither-only
  0). So the correct claim is that the short rows are a property of the *cut*, applied to either arm,
  not of the new generations — a pre-existing set of genuinely short answers surfaced by cutting, not a
  new failure mode. Say "v1 after the same cut", never "v1".

## A9. Fabrication is med_quad-ONLY — cutting it is a fix, not a new uniformity break

Measured across the grid (DoC, 2026-08-02), share of generations containing a fabricated `Question:`:

| med_quad v1 | med_quad n-gram 768 | pubmed | sciq | trivia | xsum | cnn | samsum |
|---|---|---|---|---|---|---|---|
| **47.8%** | **92.6%** | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 |

This **reverses the concern raised in §A4**. The worry was that cutting med_quad would make it the odd
one out — the only dataset whose labels and features came from a trimmed string. In fact **med_quad is
already the odd one out**: it is the only dataset with this pathology at all, and every other set is at
0.0–0.1% with nothing to cut. Applying the cut therefore **removes** a dataset-specific defect rather
than introducing a study-wide inconsistency, and `--truncate-answer-span` is correctly scoped to the one
dataset that needs it.

State it this way in the write-up. "We cut med_quad and not the others" reads like an inconsistency
until the 0.0–0.1% column is shown; with it, it reads as the only sensible choice.

---

## AMENDMENT 3 (2026-08-03) — CLOSED. The gate PASSED and v2 is nonetheless ARCHIVED.

A pre-registration abandoned without a recorded reason is worse than none, so this records both halves.

### The generation-quality gate PASSED, emphatically

| | v1 | v2 (n-gram) | **v2_span (adopted arm)** |
|---|---|---|---|
| % capped | **97.7** | 54.7 | **6.1** |
| % fabricated `Question:` | 47.8 | 92.6 | **2.3** |
| % severe degeneracy | — | 30.6 | **10.9** |
| mean answer fraction | — | — | **0.987** |

The span cut did what §A5–A7 predicted. As a piece of generation engineering it is a clear success.

### And the correctness prediction is where it stops

The §A2 prediction was *"med_quad should improve"* on the judge label. Paired over all 1800 examples,
same judge (`gpt-5-mini`, verified identical on both sides — no judge mixing):

**mean correctness 0.4180 → 0.4412, paired delta +0.0232 (sd 0.186, t +5.31).**
**33.6% improved, 25.8% got WORSE, 40.6% unchanged.**

Statistically clear, practically small, with substantial churn in both directions.

### The apparent regime flip is a LENGTH ARTIFACT, not a result

On the med_quad ID cell the raw floors appear to flip the taxonomy label CONCENTRATED → SPREAD:

| floor | v1 | v2_span | delta |
|---|---|---|---|
| msp_min | +0.1492 | −0.0424 | **−0.1916** |
| msp_sum | +0.0834 | −0.1910 | **−0.2744** |
| perplexity | +0.0771 | +0.1105 | +0.0334 |
| **msp_min, length-normalised** | +0.1448 | +0.1784 | **+0.0336** |

The two floors that collapse are exactly the two that scale with sequence length (min-over-T, sum-over-T)
while generations went from a hard 128-token cap to p90 = 560. Length-normalise `msp_min` and it *rises*
by +0.034, matching perplexity. **There is no regime change — there is a longer sequence.**

### DECISION: v1 remains canonical. v2 is archived and is not reported.

Three reasons, in order of weight:

1. **Sibling-budget consistency, which is a pre-existing documented design rule, not a post-hoc excuse.**
   `src/luq/data.py:66` sets med_quad's 128 *"matching its pubmed sibling"*, and `samsum: 56` matches xsum
   *"so the two are budget-consistent in the QA DiffTask pool"*. `_load_med_quad` states med_quad *"is a
   training source for the pubmed SameTask rung, never an eval target"*. **v2's 768 budget is 6× pubmed's
   128** and would put med_quad-as-training-source in a different length regime from the eval it feeds.
2. **A partial adoption is a mixed population.** med_quad is a training source for other evals' OOD rungs,
   so swapping only its ID cell would contaminate cells not even labelled med_quad. That is the P0 bug
   class this project has a standing rule against.
3. **The gain does not justify the cost.** +0.023 correctness, of which the striking floor movement is
   artifact, against regenerating a 3.3GB pertok cache and re-running two full ladders.

**HOW TO STATE THE LIMITATION HONESTLY IN THE REPORT.** Say: token budgets are fixed per dataset and
chosen for consistency with the sibling each set trains against; med_quad's is 128; the consequence is
that med_quad generations are usually truncated (97.7% reach the cap). **Do NOT claim the budget was
chosen to prevent degeneracy** — that would be a rationalisation invented after the fact, and v1's own
47.8% fabricated-continuation rate contradicts it.

**Archived to** `../ARCHIVED_v2_med_quad/` (mirroring the `../ARCHIVED_nonllama_caches/` precedent, which
exists so a stray glob cannot reach a non-canonical cache): all four v2 cache dirs, the five v2 genqual
CSVs, and `cache/_v3bak_med_quad/` — a 6-July pre-labelling backup (1800 rows, `correctness` all null, no
judge stamp) found during the sweep and archived for the same reason.

**Verified after archiving:** exactly ONE med_quad record file and ONE feature file remain under `cache/`,
and the floors recompute to the v1 values **exactly** (msp_min +0.1492, perplexity +0.0771, msp_sum
+0.0834) — proving no driver silently reads v2. Canonical hashes now recorded in
`prereg/CANONICAL_v1_MANIFEST.md`, closing the gap that made "v1 is unchanged" un-checkable tonight.
