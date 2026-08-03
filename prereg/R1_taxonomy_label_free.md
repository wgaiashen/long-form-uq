# PRE-REGISTRATION — R1: is the regime taxonomy decidable from UNLABELLED data?

**Written 2026-08-03, BEFORE anything is fitted.** This is the gate on the whole regime-conditional
programme: if the regime a dataset belongs to cannot be decided without looking at correctness labels,
then the routing story is fitted to its own evaluation and every result downstream of it is circular.

---

## 1. The problem being fixed

The taxonomy as it stands is **label-derived, not label-adjacent**. `results/floor_taxonomy__meta-llama_Meta-Llama-3.1-8B.csv`
assigns each dataset its regime from `best_floor` — *whichever floor achieves the highest PRR* — and PRR
is computed against the judge labels. Current content (long evals, ID rung):

| eval | msp_min | perplexity | msp_sum | best_floor | best_k | label |
|---|---|---|---|---|---|---|
| factscore | **0.4283** | 0.326 | 0.2821 | msp_min | *blank* | CONCENTRATED |
| pubmed_qa | **0.371** | −0.1736 | 0.2018 | msp_min | 1 | CONCENTRATED |
| expertqa | **0.2054** | 0.0393 | 0.1438 | msp_min | 100 | CONCENTRATED |
| med_quad | **0.1492** | 0.0771 | 0.0834 | msp_min | 10 | CONCENTRATED |
| cnn_dailymail | 0.1198 | **0.4096** | −0.0848 | perplexity | all | SPREAD |
| asqa | 0.2498 | **0.3161** | 0.1483 | perplexity | 0.25 | SPREAD |
| samsum | −0.0243 | **0.1128** | −0.1334 | perplexity | 0.5 | SPREAD |
| xsum | **−0.0149** | −0.1719 | −0.2049 | msp_min | 1 | NOT-IN-PROBABILITIES |

Two defects in the label itself, fixed here **before** any fitting:

- **`best_k` does not track `best_floor`** (expertqa is CONCENTRATED with `best_k=100`; factscore's is
  blank). Two different quantities are being reported as one regime label. **Decision, recorded before
  fitting: the regime label is `best_floor` ONLY.** `best_k` is dropped from the label definition
  entirely and reported separately as a diagnostic.
- **factscore's blank `best_k` is a MISSING VALUE and must never become a category.** Since `best_k` is
  no longer part of the label, this cannot leak in; the rule is recorded so it does not return.

## 2. ⚠️ THE DESIGN CORRECTION — why this is NOT a fitted classifier

The plan said "fit on nine datasets, predict the tenth, rotate." Two things are wrong with that and both
are fixed here rather than discovered afterwards.

**(a) There are 8 evals, not 10.** sciq and trivia_qa are absent from the taxonomy file — it is the long
grid. So the pool is 8.

**(b) n = 8 cannot support a fitted multi-feature classifier, and the rungs do not help.** A floor's PRR
depends only on the **eval set**, which is identical across rungs, so the five rungs contribute no
independent points. With 8 points and a handful of candidate features, leave-one-out would separate the
classes almost regardless of whether the underlying claim is true — a passing result would carry no
information. **A LODO-validated fitted classifier here would be a false positive generator.**

⭐ **So R1 fits NOTHING. It registers a single parameter-free rule with a direction fixed in advance, and
reports how many datasets it gets right.** A rule with no free parameters cannot be tuned to the answer,
which is a stronger guarantee than cross-validation gives at this sample size.

## 3. The rule (fixed in advance, no free parameters)

### 3.0 ⚠️ A rule considered and REJECTED before running — recorded so it is not re-proposed

The first draft of this pre-registration used **`SPREAD_min > SPREAD_mean`**: the across-example standard
deviation of each example's min logprob against that of its mean logprob. **It was withdrawn on
inspection of `src/luq/msp.py`, before any run.**

`msp_min` ranks on `1 − exp(min lp)`; `perplexity` ranks on `−mean(lp)`. **PRR is rank-based, so both are
invariant to any monotone transform — but standard deviation is not.** Measuring the spread in
log-space and in probability-space gives different answers, and the two are not even in comparable units
(`[0,1]` against `[0,∞)`). The rule therefore contained a **free choice of measurement space disguised as
a parameter-free comparison**, which is precisely the failure this pre-registration exists to prevent.
Recorded rather than deleted, because the mistake is instructive: *"no free parameters" has to be checked
against the code that computes the quantity, not asserted from the form of the formula.*

### 3.1 The registered statistic

**Mechanism.** The two regimes should differ in **where within a generation the uncertainty sits**.

- **CONCENTRATED** — the failure localises to one or a few tokens, so an example's worst token is a
  dramatic **outlier** against the rest of its own generation. The min then carries information the mean
  does not, and `msp_min` wins.
- **SPREAD** — uncertainty is distributed across many tokens, so the whole curve is depressed together
  and the worst token is unremarkable. The mean carries it, and `perplexity` wins.

The statistic that expresses exactly this, **within each example** (so it needs no cross-example scale
choice, which is what sank the rejected rule), is how many of its **own** standard deviations the worst
token sits below its own mean:

```
zgap[i] = ( mean(lp_i) - min(lp_i) ) / std(lp_i)          # dimensionless, per example
ZGAP(dataset) = mean over examples of zgap[i]
```

It is **dimensionless and invariant to any affine rescaling of the logprobs**, so there is no
log-vs-probability choice left to make.

### 3.2 ⚠️ REGISTERED PRIMARY RULE — a RANKING test, so there is no threshold either

A cutoff on `ZGAP` would be a fitted free parameter. Instead the rule is stated as an ordering:

> **Rank the datasets by `ZGAP`, descending. The CONCENTRATED datasets should occupy the top ranks and
> the SPREAD datasets the bottom ranks.**

Scored as the rank-separation (Mann-Whitney) statistic between the two label groups: with 4
CONCENTRATED and 3 SPREAD datasets there are 12 cross-group pairs and C(7,4) = 35 possible orderings.
No threshold, no weights, nothing fitted.

## 4. ⚠️ REGISTERED PREDICTIONS AND SUCCESS THRESHOLD

**Population for the primary test: the 7 datasets whose label is CONCENTRATED or SPREAD** — factscore,
pubmed_qa, expertqa, med_quad (concentrated) and cnn_dailymail, asqa, samsum (spread). xsum is the sole
NOT-IN-PROBABILITIES case and is handled by §5; it is **not** dropped quietly, it is scored there.

**P1 (primary).** `ZGAP` separates the two groups **perfectly** — all 4 CONCENTRATED datasets rank above
all 3 SPREAD datasets (12/12 cross-group pairs correct, AUC = 1.0). Exact one-sided p under the null that
the labels are unrelated to the ordering: **1/35 = 0.029.**

The full null distribution, fixed here so the result cannot be read against a moving bar:

| cross-group pairs correct | AUC | one-sided p | registered reading |
|---|---|---|---|
| 12/12 | 1.000 | 0.029 | **P1 supported** |
| 11/12 | 0.917 | 0.057 | suggestive only, NOT support |
| 10/12 | 0.833 | 0.114 | inconclusive |
| ≤9/12 | ≤0.75 | ≥0.20 | **P1 falsified** |

⚠️ **With n=7 nothing weaker than perfect separation is worth acting on**, which is why the bar is set
there rather than at a majority. A near miss is reported as a near miss.

**P2 (the competing simple explanation).** `ZGAP` is not merely a proxy for length. ⚠️ The max-of-L
statistic grows roughly with √(2·ln L), so a longer generation has a more extreme worst token **for
purely statistical reasons**, and our datasets differ in length by an order of magnitude. Reported
alongside, always: Spearman correlation of `ZGAP` with mean generation length across the 8 datasets, and
**whether mean length alone separates the two groups just as well.** If length separates them equally,
**the mechanism claim is NOT established even though P1 passed** — the honest report is then "regime is
predictable from length", which is a weaker and much less interesting claim, and it goes in the write-up
as such rather than being omitted.

**What would falsify P1:** ≤9 of 12 cross-group pairs. Then the regime is not decidable from where the
uncertainty sits within a generation, the parameter-free rule fails, and — per the plan — **we say so
plainly and stop**, reporting the negative.

⚠️ **A pass does not license a fitted controller later.** R4 fits routing rules; those must carry their
own shuffled controls. R1 establishes only that the regime label is *recoverable without labels*.

## 5. The third class, scored not excluded

xsum's label is NOT-IN-PROBABILITIES, meaning **no floor works** (best PRR −0.0149). The binary rule in
§3 must assign it something, and whatever it assigns is wrong. Two registered readings:

- The ranking test is scored on the 7 two-class datasets (§4). **xsum's `ZGAP` and its rank among all 8
  are reported regardless**, so the reader can see where it falls. It is never silently dropped.
- **No rule is registered for detecting the third class.** With exactly one NOT-IN-PROBABILITIES dataset,
  any rule that identifies it is fitted to a single point and cannot be validated — inventing one would
  manufacture a third category out of n=1. ⚠️ **The honest statement is that the taxonomy's third class
  is not testable on the data we have**, and that is what the write-up will say. If xsum turns out to sit
  at an extreme of the `ZGAP` ordering that is recorded as an observation for future work, explicitly
  **not** as a validated rule.

## 6. Secondary: does adding the short-form sets change the picture?

sciq and trivia_qa have no regime label because the taxonomy file is the long grid. Computing their
floor PRRs would give n=10. **Registered as SECONDARY, with the confound named in advance:** short-form
generations are ~20 tokens against 128–768, so length differs systematically and a rule that "works" once
they are added may be working for a trivial reason. Primary conclusions come from the 8 long evals.

## 7. Population and provenance

v1 generations. `cache/records/meta-llama_Meta-Llama-3.1-8B__<ds>__ID.jsonl` for the 5 canonical long
sets, plus `cache/asqa_rp12/`, `cache/expertqa_rp12/`, `cache/factscore_rp12/` for the namespaced three.
⚠️ **A glob over `cache/records/*` silently returns 7 datasets while claiming 10** — the script must carry
the regime map and **assert the realised dataset count, failing loud on a short load.**

Label field `correctness` (judge) is used ONLY to read the pre-existing regime label from
`floor_taxonomy__*.csv`. It is never touched when computing `ZGAP` — which is the whole point of the
test, since `ZGAP` is built from the cached logprobs alone. (Corrected before commit: this line
previously named `SPREAD_min` / `SPREAD_mean`, the rule **withdrawn in §3.0**. A pre-registration that
still names a rejected statistic in its provenance section invites exactly the ambiguity about what was
registered that the document exists to remove.)

Output: `results/regime_R1_label_free__<slug>.csv` — the `regime_*` prefix keeps it outside every
master-assembler glob. **Independent of the med_quad regeneration running on DoC.**
