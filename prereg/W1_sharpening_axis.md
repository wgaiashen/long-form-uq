# PRE-REGISTRATION — W1: the training-free sharpening family

> **Status (2026-08-09):** run. The primary claim FAILED its registered bar (margin +0.0209 PASS, signs 6/8 PASS, Wilcoxon p = 0.148 FAIL — NOT ESTABLISHED); both honest selection arms collapsed to msp_min on every fold. What survives is the regime map (descriptive). Record: the project results log

**Written 2026-08-08, BEFORE the driver was implemented and before any cell was run.**
Population: `meta-llama/Llama-3.1-8B`, the 8 long ProbeDriftLong evals, legacy carve.
Plan: `../the project plan. Results doc: `../the project results log.

---

## 1. What is being registered

`perplexity` and `msp_min` are treated in the literature as two separate baselines. They are the two
endpoints of one expression:

```
q = Σ_t  w_t · nll_t
```

with `w` uniform for `perplexity` and all-on-one-token for `msp_min`. This registers a
one-parameter family that connects them, and asks whether a **single value of that parameter, fixed
in advance and used on every dataset**, beats both.

```
z_t   = (nll_t − mean(nll)) / std(nll)        standardised WITHIN each answer
w     = softmax(τ · z)
score = Σ_t w_t · nll_t                        scored with the RAW nll, not the standardised one
```

- `τ = 0` gives uniform weights, so `score = mean(nll)` = **`perplexity`** exactly.
- `τ → ∞` puts all mass on the largest `nll`, so `score = max(nll)`, which has the same **ranking**
  as **`msp_min`**.
- The asymmetry is deliberate: standardise inside the weighting, score with the raw NLL. That keeps
  both endpoints exact while making `τ` dimensionless, so the same `τ` means the same thing on a
  different model later.

**The mechanism being claimed:** a fixed `τ` self-adapts per answer, because the softmax reads that
answer's own NLL distribution. An answer with one dominant bad token concentrates; an answer with a
flat NLL profile stays near uniform. No gate, no length rule, no extra parameter.

## 2. PRIOR EVIDENCE THAT ALREADY BEARS ON THIS, RECORDED BEFORE THE RUN

This section exists so that a negative cannot be presented afterwards as new information, and a
positive cannot be presented as unsurprising. Three closely related things have already been tested
in this project and have come back negative.

| # | test | source | outcome |
|---|---|---|---|
| P1 | best FIXED k in the hard version of this family (mean of the k lowest logprobs), cross-dataset mean | `results/topk_floor_sweep__meta-llama_Meta-Llama-3.1-8B.csv` | **k = 1 (= `msp_min`) is the maximum, at +0.151 over the 7 datasets covered.** Every interior k is lower; k = all gives +0.087. The per-example fractional version also peaks at the endpoint. |
| P2 | label-free leave-one-dataset-out selection of that k | the project results log | **NULL.** Every one-feature predictor lands +0.254 to +0.258, **below always-`msp_min` +0.284.** |
| P3 | `ZGAP = (mean − min)/std` predicts which regime a dataset is in | `prereg/R1_taxonomy_label_free.md`, `results/regime_R1_label_free__meta-llama_Meta-Llama-3.1-8B.csv` | **FALSIFIED.** 8/12 pairs, AUC 0.667. `cnn_dailymail`, the most spread dataset in the grid, has the **highest** ZGAP of all eight (3.93). Plain generation length separated the groups better than ZGAP did. |

P3 is the most directly damaging, and it must be stated plainly: **`ZGAP` is exactly the quantity
that governs how much `softmax(τ · z)` concentrates.** A high `ZGAP` means the worst token is a
large outlier in its own answer, which is precisely when the softmax puts its mass there. So R1
already measured the family's steering signal at the dataset level and found it points the wrong
way on `cnn_dailymail`, the dataset where `perplexity` beats `msp_min` by the largest margin in the
grid (0.290).

**Why the question is nevertheless still open, stated narrowly.** P1, P2 and P3 are all
*dataset-level* results. P1 and P2 select one parameter *per dataset*; P3 predicts a *dataset's*
label. The claim registered here is *per-example*: that a single global `τ`, read against each
answer's own distribution, helps on average even though per-dataset selection of the same parameter
does not. That is the only thing this pre-registration tests. **The prior is not good, and that is
recorded here rather than discovered afterwards.**

## 3. A STRUCTURAL FACT THE DESIGN DEPENDS ON

**Training-free methods are rung-invariant.** Verified on the master ladder: `msp_min` on
`pubmed_qa` is +0.3710 at all five rungs, and the same holds for `perplexity` and `msp_sum` on all
eight datasets (120 rows, 24 distinct values). A free method never sees a training pool, so the rung
cannot touch it.

Three consequences, all registered:

1. **`τ` is a per-dataset question, not a per-cell one. The unit of analysis is the DATASET,
   n = 8, NOT 40.**
2. The probe's training pool never enters `τ` selection, so "held-out source" does not apply and the
   1ds-DiffTask single-source problem does not arise.
3. **Any free-vs-free comparison reported over "32 OOD cells" is 8 values counted four times.** Every
   interval in this workstream is computed on n = 8. This is already recorded at
   the project results log; it is restated here because it sets the sample size for every
   test below.

## 4. THE REGISTERED QUESTIONS, WITH THRESHOLDS FIXED NOW

### Q1 (PRIMARY, go / no-go)

> Does a single, PRE-COMMITTED, a-priori `τ` — fixed across all datasets — beat **both** endpoints
> on the per-dataset mean?

**The registered value is `τ = 1` on standardised NLL.** Justification, fixed in advance and not
derived from any result: on standardised units, `τ = 1` gives a token one standard deviation above
its answer's mean `e` times the weight of an average token. That is a natural scale chosen from
first principles.

**Bar, all three parts required:**

| part | threshold |
|---|---|
| margin | mean PRR at `τ = 1` exceeds `msp_min`'s **+0.1855** by **more than +0.010**, paired per dataset |
| sign count | beats `msp_min` on at least **6 of 8** datasets |
| test | two-sided Wilcoxon signed-rank on the 8 paired differences, **p < 0.05** |

The margin alone is deliberately **not** sufficient. At n = 8 with per-dataset endpoint margins
ranging from 0.066 to 0.545, a point estimate can move +0.010 on one dataset's noise. The minimum
attainable two-sided Wilcoxon p at n = 8 is 0.0078, so the test is achievable but demanding, which
is the intent.

**Registered failure reading:** if the margin passes but the sign count or the test does not, the
result is reported as **"not established"**, not as a win. It is not re-described as "directional
support" and no secondary threshold is invented afterwards.

**A clean negative is a publishable outcome and is reported as such.** It would say the field's two
baselines really are distinct tools rather than one family, which is on-topic for a thesis about
regimes, and it is far better than a quietly tuned positive.

### Q2 (SECONDARY, ordering)

> Does the per-dataset best `τ` track the concentration regime?

**Prediction, stated before looking:** `pubmed_qa` and `factscore` toward **HIGH** `τ`
(concentrated); `cnn_dailymail`, `asqa` and `samsum` toward **LOW** `τ` (spread).

Scored as the rank separation between the two named groups, the same Mann-Whitney form R1 used. This
is an ordering test with no threshold and nothing fitted.

**Registered failure:** if the two groups do not separate, the concentration story does not explain
where the optimum sits, and that is reported plainly.

### Q3 (SECONDARY, transfer)

> Does a `τ` chosen by a **procedure** rather than an oracle transfer to data it was not chosen on?

Scored by the arms in §5, with the transfer rate defined there.

### Q4 (REGISTERED DIRECTIONAL PREDICTION, the control that catches a false positive)

> P3 implies that a fixed `τ` should **hurt `cnn_dailymail`**, because cnn's answers concentrate the
> most while cnn is the dataset that most wants the mean.

**This is registered as a prediction, not as a caveat.** If Q1 passes **and** `cnn_dailymail`
improves, that combination contradicts R1 and is treated as a **suspected bug** until it is
explained, not as a stronger result.

## 5. ARMS — four, reported in separate columns, never pooled

| arm | how `τ` is chosen | status |
|---|---|---|
| **A0** | **a priori, `τ = 1`, committed above** | **PRIMARY** |
| A1 | leave-one-dataset-out: selected on the other 7, applied to the held-out one | secondary |
| A2 | external: selected on `sciq` + `trivia_qa` only | secondary |
| A3 | oracle: best `τ` on test | **CEILING ONLY, never a result** |

- **A1** reports the score **the procedure achieves**, never the best `τ`'s score. It uses the
  **one-standard-error rule**: the `τ` nearest an endpoint that is within 1 SE of that fold's best,
  not the raw argmax, because with 8 noisy units the argmax is optimistic. **All 8 selected `τ` and
  their spread are reported.** Agreement across folds is the evidence that a single global constant
  exists; disagreement is a finding, not a nuisance, and is reported as one.
  **The tie-break is fixed here, before the run, because it is a free choice.** "Nearest an
  endpoint" is measured in grid-index steps as `min(i, last − i)`. If two candidates are equidistant
  from opposite ends, the one with the **higher training mean** wins. Stating this afterwards would
  be a parameter chosen on results.
- **A2** touches no evaluation data at all: `sciq` and `trivia_qa` are labelled, cached, and not
  among the 8 long evals. A failure here is informative rather than fatal — it would mean `τ` is
  regime-specific, which is itself on-topic.
- **A3** is always labelled an oracle, never appears in a column with A0 to A2, and is never
  described as a result.

**Transfer rate, registered for A1 and A2:** on how many of the 8 datasets did the selected `τ` beat
**both** endpoints. Precedent to keep in view: the auxiliary-loss λ, chosen by an analogous
procedure, beat λ = 0 on only 19 of 60 rows. **If `τ` transfers no better than that, it is said
plainly.**

**Sweep:** `τ ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, ∞}`. **The FULL curve per dataset is reported,
not the argmax.** The shape is the evidence for Q2 and it makes the sensitivity visible instead of
hidden.

## 6. THE FAMILY CONTROL (registered, not optional)

Two further one-parameter families with the **same two endpoints** are run alongside:

| family | score | at 0 | at ∞ |
|---|---|---|---|
| power mean | `M_p = (mean(nll^p))^(1/p)` | `p = 1`, mean | max |
| Lehmer, via the existing `weighting.beta_sharpen` | `w ∝ nll^β`, then `Σ w·nll / Σ w` | `β = 0`, mean | max |

**Why this is registered rather than added afterwards:** if softmax-`τ` works and the other two do
not, the result is about the **specific weighting function**; if all three behave alike, it is about
the **family**. Running only one and reporting it would leave that unresolved, and it is the obvious
reviewer question.

Recorded asymmetry: softmax-`τ` standardises and is therefore dimensionless, while the power mean
and Lehmer act on raw NLL magnitudes and are not. They are not like-for-like on scale invariance,
only on endpoints. That is a property of the comparison, stated now so it is not presented later as
a discovery.

## 7. VERIFICATION — run FIRST, and the run stops if V1 fails

**V1. ENDPOINT IDENTITY.** At `τ = 0` the score's PRR must equal `perplexity`'s, and at `τ → ∞` it
must equal `msp_min`'s, on every dataset, on the same test rows.

**This is a RANKING identity, not a value identity.** `src/luq/msp.py:39` computes `msp_min` as
`1 − min(exp(lp))`, which is a monotone transform of `max(nll)` and **not equal to it**. PRR is
rank-based, so the PRRs are identical while the score vectors are not. A check written as
"values agree to 1e-6" would fail for the wrong reason. The registered check is
`|PRR(τ→∞) − PRR(msp_min)| < 1e-9` and `|PRR(τ=0) − PRR(perplexity)| < 1e-9`, plus an external gate
that both reproduce the published master values within 0.01:

| dataset | `msp_min` | `perplexity` |
|---|---|---|
| pubmed_qa | +0.3710 | −0.1736 |
| factscore | +0.4283 | +0.3260 |
| expertqa | +0.2054 | +0.0393 |
| xsum | −0.0149 | −0.1719 |
| med_quad | +0.1492 | +0.0771 |
| asqa | +0.2498 | +0.3161 |
| cnn_dailymail | +0.1198 | +0.4096 |
| samsum | −0.0243 | +0.1128 |

**IF V1 FAILS THE RUN STOPS AND THE FAILURE IS REPORTED.** It would mean the NLL convention or
the row population differs from the master table, and nothing downstream would be interpretable.

**V2. DEGENERATE CASES.** `std = 0` (all NLLs in an answer identical) falls back to uniform weights
**explicitly**, never a division by near-zero. The count of answers with fewer than 5 tokens is
reported per dataset, because the standardisation is unstable on very short answers.

**V3. NLL CONVENTION.** Confirmed against the code that wrote the cache, not from memory:
`generate.py:159-162` stores `log p(chosen token)` from `torch.log_softmax`, so the cached values
are **natural-log LOGPROBS (negative), not NLLs**, one per generated token (length G, no prompt
anchor). The driver negates them itself and prints the convention it used.

**V4. AVAILABILITY.** Cached records must exist for all 8 long datasets **and** for `sciq`/`trivia_qa`
(needed for A2). `expertqa`, `asqa` and `factscore` live under `cache/*_rp12/`, so a hard-coded
`cache/` glob sees only 5 of 8. Resolution goes through
`Config(prompt_regime=PROMPT_REGIME.get(dataset, ""))`. **The realised dataset count is asserted and
the run fails loud on a short load.** The model slug is pinned; no model-agnostic glob is used.

**V5. POPULATION.** The same test rows as the master table: `xl_rungs.eval_split`, seed 0, judge
label, finite-label filter. The carve-then-filter change **has landed** (`LUQ_CARVE`, added
2026-08-08) and **defaults to `legacy`**, which reproduces the pre-change behaviour exactly. This
run uses the default legacy carve so it is comparable with the existing numbers, and **stamps the
realised `LUQ_CARVE` value into every output row**, because the new carve moves `expertqa` and
`factscore`.

**V6. GRID COMPLETENESS.** `factscore` is absent from `topk_floor_sweep__*.csv`, so the existing
family evidence is 7 of 8. It is included here, and any table that cannot cover all 8 says so and
names the missing cells rather than reporting a partial grid as a full one.

## 8. DIAGNOSTICS REPORTED WHATEVER THE VERDICT

- **Weight concentration.** Per example, `ESS = 1 / Σ w²`, the effective number of tokens the weight
  lands on. Reported per dataset at `τ = 1`, and correlated within dataset against answer length and
  against `ZGAP`.
  **Registered confound.** The largest attainable `z` in an answer of `n` tokens is bounded by
  roughly `√(n − 1)`. So at a fixed `τ`, a long answer **can** concentrate far more than a short one,
  which means the family is **implicitly length-dependent**. Any claim that "length is not the axis"
  must be checked against this, not asserted.
- **The `cnn_dailymail` directional check** (Q4), reported either way.
- **Which endpoint each dataset's curve moves toward**, as the evidence for Q2.

## 9. WHAT WOULD MAKE THIS RESULT UNTRUSTWORTHY

Recorded now, so the checks are not chosen after seeing the numbers:

- V1 fails on any dataset → the population is wrong, stop.
- Q1 passes **and** `cnn_dailymail` improves → contradicts R1, treat as a suspected bug and find the
  cause before reporting (Q4).
- `ESS` correlates strongly with length within datasets → the family is acting as a length rule, and
  the per-answer-shape story is not established even if Q1 passes.
- A1's 8 selected `τ` disagree widely → no single global constant exists, and A0's result, positive
  or negative, is a property of one arbitrary point rather than of the family.

## 10. PROVENANCE

Driver: `scripts/checks/sharpening_family.py`. Inputs: `record["token_logprobs"]` from
`cache/records/*.jsonl` and the three `cache/*_rp12/records/*.jsonl`, nothing else — no per-token
hidden states, no GPU. Output: `results/sharpening_family__meta-llama_Meta-Llama-3.1-8B.csv`, with
the `sharpening_` prefix keeping it outside every master-assembler glob. Machine: RCS, CPU only.
Independent of the Qwen workstream and of the ProbeDriftLong library move.
