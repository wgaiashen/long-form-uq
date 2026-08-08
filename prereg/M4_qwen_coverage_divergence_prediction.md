# PRE-REGISTRATION — M4: predicted label-coverage divergence for Qwen2.5-14B

**Written 2026-08-08, on DoC, BEFORE the full Qwen generation run and BEFORE any Qwen judge call.**
Nothing here has seen a Qwen coverage number, because none exists yet: no Qwen row has been labelled.
The entire point of writing it now is that the flag in `M3_split_rule_and_coverage.md` should fire
**mechanically**, so that a six-dataset primary analysis reads as a rule that was already agreed rather
than an excuse constructed after seeing an inconvenient Qwen result.

This registers a **prediction**. It does not change any rule. The decision rule is M3's, unchanged.

---

## 1. The rule being predicted against (restated from M3, NOT modified)

Llama-3.1-8B label coverage, the baseline:

| dataset | labelled / rows | coverage |
|---|---|---|
| pubmed_qa, xsum, cnn_dailymail, med_quad, samsum, asqa | all | **100%** |
| expertqa | 1724 / 2016 | **85.5%** |
| factscore | 455 / 500 | **91.0%** |

Registered thresholds: **>5 pp divergence → record. >10 pp → flag as unreliable**, and per M3 §4,
*"if coverage divergence on expertqa or factscore exceeds the 10 pp flag, the six-dataset version
becomes primary for that claim."*

Those thresholds were measured, not chosen: at 10 pp the PRR noise from coverage loss alone reaches
~0.5× the R3 effect size (+0.054), and M3 notes they are **lower bounds**, because the real drop is
not random — dropped rows are score-correlated.

---

## 2. Why a divergence is expected at all: two different routes to `uncovered == 1.0`

M3 §5 established the Llama mechanism: `uncovered == 1.0` requires **zero adjudicable claims**, which
is easiest when an answer makes **few** claims. Hence Llama's dropped rows are significantly *shorter*
than its kept rows (expertqa median 177.0 vs 200.5; factscore 47.0 vs 90.0; Mann-Whitney p = 1.2e-4 /
3.4e-4).

**Qwen introduces a second, independent route to the same state.** Degenerate text — the repetition
the `luq.degeneracy` detector fires on — is *long* but carries no adjudicable claim. So a degenerate
Qwen row lands in `uncovered == 1.0` for the opposite length reason.

This matters beyond the headline number, because M3 §5 also registered a directional expectation:
*"dropped rows will again be shorter than kept rows on both datasets. A reversal would mean the
absence mechanism differs by model and would invalidate treating coverage as a simple scalar."*

**Registered here: on Qwen that directional expectation may legitimately fail, and failing it is a
FINDING, not a bug.** If Qwen's dropped rows are bimodal in length (a short/few-claims mode plus a
long/degenerate mode), or median-longer than kept rows, that is evidence the absence mechanism is
model-dependent — exactly the thing M3 said would invalidate a scalar treatment of coverage. It must
be reported as such and not quietly smoothed into the Llama story.

---

## 3. The input measurement, and its known bias

Severe-degeneracy rate, `luq.degeneracy.is_severe`, ID split:

| dataset | Llama (full population) | Qwen (n=200) | change |
|---|---|---|---|
| expertqa | 11.76% (n=2016) | 23.00% | **+11.2 pp** |
| factscore | 0.20% (n=500) | 17.00% | **+16.8 pp** |
| asqa | 1.37% (n=948) | 10.00% | +8.6 pp |
| med_quad | 12.33% (n=1800) | 11.50% | −0.8 pp |
| samsum | 0.72% (n=1800) | 3.50% | +2.8 pp |
| pubmed_qa | 1.11% | 1.50% | +0.4 pp |
| xsum / cnn_dailymail | 0.00 / 0.03% | 0.00 / 0.00% | ~0 |

⚠️ **The Qwen column is from a HEAD SLICE and is therefore biased.** `01_extract --limit` breaks after
the first n rows; it is not a sample. Demonstrated on expertqa: the first 200 rows have gold p90 = 450
against 349 for the full 2016, and 450 lies outside the [316, 376] range of 2,000 random 200-row draws.
A random-sample re-measurement (`--sample-n`, added 2026-08-08) is running and **will replace this
column**.

**The formula in §4 and the thresholds in §1 are fixed now and do not move when that column is
replaced.** Only the inputs update. The re-measured table will be appended below with the arithmetic
re-run, so the difference between "predicted" and "predicted after unbiased input" is auditable.

---

## 4. The registered prediction

**Predictor (fixed now):** a severe-degenerate generation carries no adjudicable claim, so it lands in
`uncovered == 1.0`. Predicted coverage drop ≈ the increase in severe-degeneracy rate:

```
predicted_coverage_Qwen  ≈  coverage_Llama  −  (severe_Qwen − severe_Llama)
```

| dataset | Llama coverage | predicted Qwen coverage | predicted drop | >5 pp record? | >10 pp flag? |
|---|---|---|---|---|---|
| **expertqa** | 85.5% | **~74.3%** | **~11.2 pp** | yes | **yes — predicted to TRIP** |
| **factscore** | 91.0% | **~74.2%** | **~16.8 pp** | yes | **yes — predicted to TRIP** |
| asqa | 100% | ~91.4% | ~8.6 pp | yes | no (borderline) |
| med_quad | 100% | ~100% | ~0 pp | no | no |
| samsum | 100% | ~97.2% | ~2.8 pp | no | no |
| pubmed_qa, xsum, cnn_dailymail | 100% | ~100% | ~0 pp | no | no |

**This is a point prediction on a one-directional assumption, so bound it.** Two errors run opposite
ways: not every severe-degenerate row is unadjudicable (a row may state a claim and *then* degenerate,
which the judge can still score), and not every uncovered row is degenerate (Llama's few-claims route
persists). Registered interval: **expertqa 74–81% (drop 4.5–11.5 pp), factscore 74–88% (drop 3–17 pp).**

**Headline registered expectation: expertqa and factscore both exceed the 5 pp record threshold, and
both are predicted to exceed the 10 pp flag — factscore more severely than expertqa**, because its
degeneracy rises from an essentially clean baseline (0.20%, i.e. an ~85× increase) whereas expertqa's
roughly doubles from an already-high 11.76%.

⚠️ Note this refines the informal expectation that expertqa is the one at risk. On the current
evidence **factscore is predicted to trip harder.** Registered now so that ordering cannot be claimed
after the fact.

### Consequence if it trips, applied mechanically

Per M3 §4, no further judgement required: **the six-dataset version (excluding expertqa and factscore)
becomes primary for the affected claim**, with all eight still reported alongside at equal status.
Coverage is reported per model per dataset in every table, never a footnote.

---

## 5. What would falsify this prediction

Any of the following is a clean negative and is reported as such:

1. **expertqa and factscore coverage within 5 pp of Llama's** despite the degeneracy increase. That
   would mean severe degeneracy does *not* prevent adjudication — the judge still finds a claim in
   repetitive text — and the §4 predictor is simply wrong. It would also mean coverage is a poorer
   proxy for generation quality than assumed.
2. **Coverage drops far beyond the interval** (e.g. expertqa below 70%). That would indicate a route
   to `uncovered` that is neither the few-claims mechanism nor degeneracy, and needs finding before any
   cross-model claim is made.
3. **Qwen's dropped rows are not shorter than kept rows** — the M3 §5 reversal. Registered in §2 above
   as a finding about mechanism, not a failure of this prediction.
4. **A dataset outside {expertqa, factscore, asqa} losing coverage.** The other five are at 100% on
   Llama and their degeneracy barely moves, so any loss there is unexplained and blocks the comparison.

## 6. Standing constraints this does not override

- Nothing is re-tuned on Qwen results. This is a replication.
- Coverage is reported for **both** models in every table; the two populations are never pooled.
- A not-measured cell is **blank, never zero**.

---

## Appendix A — re-measured degeneracy on a RANDOM sample

*To be appended when the `--sample-n` run lands, replacing the biased head-slice column in §3. The
formula and thresholds above are fixed and are not revised at that point.*
