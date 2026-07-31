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
