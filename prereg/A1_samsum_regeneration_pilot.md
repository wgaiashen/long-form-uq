# PRE-REGISTRATION — A.1: the samsum regeneration pilot

**Written 2026-07-31, BEFORE generating or judging anything.** Committed ahead of the run.

---

## The finding that reframed this pilot

Measured at the time of writing from `cache/records/*` with `o200k_base` tokenisation, comparing each dataset's
token budget against the p90 length of its **gold reference answer**:

| dataset | budget | gold p90 | budget ÷ gold p90 | % capped | what truncation IS |
|---|---|---|---|---|---|
| **med_quad** | 128 | **591** | **0.22×** | 97.7% | **the DEFECT** — the budget cannot fit the answer |
| samsum | 56 | 45 | 1.25× | 65.9% | a **SYMPTOM** — the model over-generates |
| xsum | 56 | 33 | 1.70× | 30.0% | a **SYMPTOM** |
| cnn_dailymail | 128 | 83 | 1.54× | 28.4% | a **SYMPTOM** |

**These are two different problems and must not share one justification.**

- For **med_quad**, "we cannot claim to study long-form generation when the output is truncated" is
  straightforwardly true. Raising the cap fixes something real.
- For **samsum / xsum / cnn**, the budget is *already* 1.25–1.7× the reference summary, and 28–66% of
  generations still run into it. The model is producing summaries substantially longer than the target.
  Raising the cap does not fix over-generation, it **accommodates** it. The honest statement is *"the
  model over-generates relative to the reference, and the cap is where that becomes visible"* — which is
  a property of the task setup that is arguably worth **reporting rather than removing**.

## The two hypotheses, with opposite predictions

| hypothesis | mechanism | prediction when the cap is raised |
|---|---|---|
| **H-trunc: truncation is hurting** | generations are cut mid-sentence, and the judge penalises an incomplete summary | letting sentences finish **IMPROVES** judge scores |
| **H-over: over-generation is hurting** | the model drifts past the reference; more room means more drift | scores **FALL or hold FLAT** |

samsum is the sharpest test available: it is the most-capped of the three symptom datasets (65.9%) and
its budget most exceeds its reference in relative terms after xsum.

## REGISTERED PREDICTIONS

1. **samsum (this pilot): FLAT or SLIGHTLY WORSE.** H-over. Predicted before running.
2. **med_quad (later): IMPROVES.** H-trunc — it is the one dataset with a genuine budget defect.
3. **xsum, cnn (later): FLAT or SLIGHTLY WORSE.** H-over.
4. ⚠️ **If ALL FOUR improve, be suspicious.** That pattern is better explained by the judge than by the
   decoding change. Check the noise-floor result (below) before believing any of it.

## Order of operations — this matters, and step 1 is not what it first appeared

**Step 1 — the noise floor, on UNCHANGED v1 text, BEFORE any regeneration.**

⚠️ **Correction to the plan as briefed:** samsum's v1 labels were **already produced by gpt-5-mini**
(verified: all 1800 rows carry `correctness_model = gpt-5-mini`). So there is **no judge change to
isolate** for samsum. What re-judging unchanged v1 text actually measures is **judge run-to-run
variance**, because `llm_judge._gpt_response` calls at `temperature=1, top_p=1` — the judge is
stochastic, and repeat calls on identical input disagree.

That makes step 1 **more** necessary, not less: without it, a pilot score movement of a few hundredths
is uninterpretable. The quantity needed is the **standard error of a dataset-level mean** under judge
noise, since the pilot compares means over ~1800 rows.

**Step 2 — the pilot generation**, judged with gpt-5-mini, compared against the **re-judged** v1 labels
rather than the original stored ones, so the comparison is like-for-like.

## Gates (all must pass)

1. realised truncation rate < ~10%
2. degeneracy rate does not rise (`luq.degeneracy.classify`, severe/degraded)
3. **empty-generation rate ≤ 2%** — the med_quad failure mode, where a repetition penalty applied over a
   long few-shot context forced immediate EOS and produced ~35% empty generations
4. teacher-forced pooling-consistency guard passes (`feature_pertok_consistency.py`, `TOL = 1e-4`)
5. PRR moves in a direction that can be explained
6. **judge score must not fall** (read against the step-1 noise floor, not against zero)

## Report the LENGTH DISTRIBUTION, not only the cap rate

Median and p90 generation length before and after, beside the reference p50/p90. **If samsum's median
moves 56 → ~90 while the reference stays at 45, that confirms over-generation regardless of what the
judge says**, and it is the cleanest single piece of evidence either way.

## Decision rule, fixed in advance

- **Scores improve** → H-trunc; proceed with the regeneration of all four as planned.
- **Scores fall or are flat** → H-over. **Only med_quad is regenerated for budget reasons.** The other
  three are regenerated *only* to unify the decoding regime — a weaker rationale that must be stated as
  such in the write-up. Do not regenerate three datasets on a justification the data does not support.

## Pilot configuration

- samsum, budget **56 → 96** (≈2× gold p90, deliberately generous so that if the model runs to 96 the
  over-generation is unambiguous), `--repetition-penalty 1.2`, everything else held fixed
- namespaced to its own `--prompt-regime`, so no v1 path can be overwritten
