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

## ⚠️ REGISTERED PREDICTION AND GATE CORRECTION

**For med_quad, "degeneracy must not rise" is the WRONG gate.** Raising the cap from 128 to ~768 removes
a loop-truncator, so **med_quad's measured degeneracy rate may rise, and a rise would be the expected
consequence of removing the truncation rather than evidence the fix failed.**

**If med_quad's degeneracy rises, the conclusion is that med_quad needs BOTH a larger budget AND the
anti-loop measure** (`no_repeat_ngram_size=3`), not one or the other. That is consistent with med_quad
being the only dataset that needed the anti-loop measure in v1 in the first place
(`pbs/extract_neighbour.pbs:39-43`: the anti-loop default was 3, opt-*out* for short sets like samsum).

⚠️ **This exemption applies to med_quad ONLY.** For samsum, xsum and cnn the severe rate is ≤0.72%, so
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

# ⚠️ AMENDMENT — 2026-08-02

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

⚠️ **Do not quote med_quad's severe rate without this decomposition.** Report `run≥25` separately.

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
  sees. ⚠️ **A and B are not escapes at a high rate**: both then produce a label computed over text
  containing a fabricated follow-up question, and `correctness_raw` inherits that problem precisely
  because raw *is* the full-text judge. Only C actually aligns the two.
