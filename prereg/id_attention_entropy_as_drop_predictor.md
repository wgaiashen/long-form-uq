# Pre-registration — does in-distribution attention entropy alone predict the OOD drop?

> **Recorded outcome (added 2026-09-07):** run. Fails all three registered bars at the honest unit of analysis, and the registered stopping rule closed this line.

**Written 2026-08-05, before running this analysis.** Committed ahead of the run so the procedure and the
decision rule are timestamped.

---

## READ THIS FIRST — THIS ONE IS NOT BLIND, AND SAYING SO IS THE POINT

Prereg 0.1 (Δentropy) was written before its numbers existed. **This one is not.** The A1 run on
2026-08-05 printed a `ne_ID` column for all eight datasets and a `d_PRR` column for all 32 cells, and I
have read both. So the inputs to this analysis were already on screen before this file was written.

That means **this document pre-registers the PROCEDURE, the CONTROLS and the DECISION RULE — it does not
pre-register a blind prediction**, and no result from it may be presented as if it did. It is a post-hoc
analysis conducted under a fixed rule, which is weaker than 0.1 and must be labelled that way in the
write-up. The reason to write it at all is that fixing the rule in advance still prevents the actual
failure mode: running several variants and reporting whichever looks best.

---

## Why this analysis exists

Δentropy (prereg 0.1) is dead as a switching signal — the breakdown by rung and by dataset found no slice
where it works. But Δentropy has a practical drawback the ID entropy does not:

> **Δentropy requires an OOD pass to compute.** You must already have run the probe on the target data to
> know how much its attention moved. **ID entropy is a property of the trained probe alone**, measurable
> once, before deployment, with no target data at all.

So if ID entropy carried the signal, it would be a *cheaper* gate than the one we just lost, not merely a
replacement. That is the whole motivation, and it is worth one bounded test before entropy is closed.

## The quantities

- **Predictor: `ne_ID`** — mean per-example normalised attention entropy `H / log T` of the trained
  attention pooler on its own ID cell. **One number per dataset.** Normalised for the same reason as in
  0.1: raw `H` scales with length and our datasets differ by an order of magnitude in output length.
- **Target A (primary): the PRR drop** — `armA` OOD PRR minus ID PRR, averaged over the four OOD rungs.
  One number per dataset. Negative = worse.
- **Target B (secondary): the OOD PRR level** — `armA` PRR at the OOD rungs, averaged. A router picks the
  best method *at the target*, so the level is arguably the more decision-relevant quantity than the
  change. Reported alongside, never instead.

## PSEUDO-REPLICATION — the honest n is 8, not 32

`ne_ID` is **constant within a dataset**. A per-cell correlation over 32 cells therefore repeats each
predictor value four times and adds no independent information; it would report n=32 while the effective
n is 8, narrowing the CI by roughly a factor of two for free. That is exactly the kind of well-formed,
entirely wrong number this project keeps catching.

**Registered: the dataset-level test at n=8 IS the test.** The per-cell version may be reported for
completeness but is explicitly labelled pseudo-replicated and never carries the headline or the CI.

## Direction — registered as TWO-SIDED, because two mechanisms argue opposite ways

1. **Sharp ID attention is brittle.** A low-entropy pooler has latched onto a few specific token
   positions; those positions do not exist in the same form on another dataset, so it should break
   harder. → low `ne_ID` goes with a bigger (more negative) drop → **positive** correlation between
   `ne_ID` and the drop.
2. **Flat ID attention means nothing was learned.** A high-entropy pooler is nearly mean-pool, so its ID
   score is carried by the classifier rather than the aggregation, and it has no pooling advantage to
   lose. → **negative** correlation.

Both are plausible; neither is the project's stated prior. **Registered as two-sided.** A result in
either direction counts, and the sign must be reported with the mechanism it supports rather than being
narrated after the fact.

**Expectation, stated openly as informed by having seen the numbers:** I expect this to be **weak and
non-monotone**. From the visible `ne_ID` values, pubmed is by far the sharpest (0.607) and has the
largest drop, while cnn (0.941) and expertqa (0.973) are the flattest and also drop hard, with the
middle-entropy datasets dropping least. That is a U-shape, and a rank correlation will not see it.

## Controls, fixed in advance

- **Permutation null** on the n=8 Spearman (all 8! orderings are enumerable; use exact or 20,000 draws).
- **Length confound**, as in 0.1: report `corr(mean length, ne_ID)` and `corr(mean length, drop)`
  alongside. If `ne_ID` is mostly a length proxy, that must be visible in the same table.
- **The secondary U-shape test is declared HERE, not invented after:** correlate `|ne_ID − median(ne_ID)|`
  against the drop. It is a second test on the same eight points, so it carries a multiple-comparisons
  cost and is reported with a Bonferroni-adjusted threshold (α = 0.025). It is **exploratory** and cannot
  by itself reopen entropy.

## Decision rule, fixed in advance

- **ADOPT AS CANDIDATE** only if the primary n=8 Spearman has **|ρ| ≥ 0.70**, a bootstrap CI excluding 0,
  **and** it beats its permutation null at p < 0.05. It must then be compared head-to-head against the
  **length** gate leave-one-dataset-out before anything is built on it.
- **ANYTHING ELSE → ENTROPY IS CLOSED**, in all its forms (Δentropy from 0.1, ID entropy here). The
  project moves to the length gate and to the top-k surprisal prior, and no further entropy variant is
  run. This is the stopping rule and it is binding: "try one more entropy statistic" is how a dead line
  consumes a fortnight.
- **A favourable result here is a SUSPECT**, because the data were already visible. If it clears the
  bar, the required follow-up before it is believed is a **leave-one-dataset-out** check: refit the rule
  on 7 datasets, predict the 8th, and report whether the ordering holds out of sample. n=8 with a
  post-hoc rule is otherwise far too easy to fit.

## Outputs

The n=8 table (`ne_ID`, drop, OOD level, mean length per dataset), Spearman + Pearson with bootstrap CI
and permutation p for both targets, the length-confound rows, the declared U-shape secondary, and a
scatter with n stated on the figure. Written to
`results/id_entropy_vs_drop__meta-llama_Meta-Llama-3.1-8B.{csv,png}`.
