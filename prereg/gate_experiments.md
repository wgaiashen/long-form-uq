# PRE-REGISTRATION — two label-free gate attempts (written 2026-07-30 BEFORE any gate was fit)

Population: the ProbeDriftLong OOD cells (`router_pdl`'s 32 cells = 8 long evals × {SameTask, DiffTask, LOO,
1ds}-long). Seed regime: **seed-1** post-hoc reads off the saved attention poolers (NOT the 3-seed ladder — do
not cross-quote with the 3-seed aggregate). Gate fit **leave-one-DATASET-out**; the held-out dataset is never
seen during fitting. Every feature is asserted label-free (computable at test from logprobs alone, no train set).

## ITEM 1 — per-example signal-shape gating (3-way {msp_min, perplexity, probe})
Features (per example, from logprobs): min−mean logprob gap, effective-k (participation ratio of surprise),
fraction of surprise in the top-decile tokens, skew, kurtosis, + length (so the incumbent length gate is nested).
Gate = small multinomial logistic (report param count).

**PREDICTION (registered):** the signal-shape gate beats the length gate **MOST on datasets whose §D.6 taxonomy
assignment is unambiguous** — pubmed (concentrated → msp_min), cnn (spread → perplexity), xsum (absent → probe) —
and **LEAST on the plateau/ambiguous ones** (expertqa, med_quad). **A uniform gain everywhere would be SUSPICIOUS**
(would indicate capacity, not signal-location information).

**The §D.6 predictability NULL DOES NOT APPLY here.** That test tried to predict ONE k per dataset from 8
datapoints (near-guaranteed to fail). THIS is per-example, thousands of rows, gate fit LODO — a different
experiment. The decisive quantity is **(real-feature gate) − (shuffled-feature gate)**, reported BEFORE the
headline score; if shuffling the features within each cell does not drop the score, the gain is bin/parameter
capacity, not signal-location information, and the gate is a null.

## ITEM 2 — length-gated entropy control (C1-revisited, conditional)
Post-hoc on the saved poolers; a single length threshold (or smooth sigmoid) fit LODO, label-free, routing short
output → V2 (re-sharpened attention) and long output → plain armA. Motivated by: V2 helped pubmed (+0.168→+0.262),
destroyed expertqa (+0.209→−0.094), corr(median length, V2−armA) = −0.418 (V2 helps short, hurts long).

**PREDICTION (registered):** length-gated V2 **improves the QA/factuality family** {pubmed, med_quad, asqa,
expertqa, factscore} and **leaves summarisation** {xsum, cnn, samsum} **unchanged** (the gate should route summ to
plain armA). **If it improves summarisation too, the gate is NOT doing what it claims** and the gain is elsewhere.
Report BY FAMILY, never a pooled mean (a pooled mean would hide the split, which is the whole point). Same
shuffled-length control.

## Reporting discipline (both items)
Control margins reported BEFORE the headline. Paired CIs = per-cell differences bootstrapped over cells (NOT two
independent intervals). Oracles (2-way, 3-way, per-example, per-dataset) are DIAGNOSTICS, never method rows. If
both fail: report plainly that the per-example headroom is large and signal-structured but no label-free gate
captures it — realising it is the open problem. A favourable result is a suspect until its control clears it.

## ITEM 3 (added 2026-07-30, BEFORE running) — FAMILY RULE (task identity is a legitimately label-free INPUT)
Every gate above was per-example label-free. But the TASK FAMILY (summarisation vs QA/factuality) is known at
deployment — an input property, not a correctness label — so a family rule is legitimately label-free. Learn per
family which of {msp_min floor, attention probe} wins on the TRAIN datasets, apply to the held-out dataset by its
family, LODO. Ceiling = dataset-level oracle +0.252 vs always-probe +0.213 → +0.039 available.
**PREDICTION (registered):** the family rule HELPS on QA/factuality (where the floor wins) and is a NO-OP on
summarisation (where it routes to the probe anyway). If it helps summarisation too, something else is happening.
**Mandatory control:** SHUFFLED-FAMILY-LABEL (assign families at random) — report (real − shuffled) as decisive,
paired CIs over cells. If this also nulls, gating is CLOSED and we say so.
