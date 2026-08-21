# Pre-registration — the Qwen2.5-14B replication of the ProbeDriftLong findings

## 0. Provenance — read this first

The thresholds below were fixed in the project's planning notes (a planning document kept outside this repo),
committed in `9731fea` on 2026-08-08 02:42 +0100, **before a single Qwen record existed** (Qwen
generation began 2026-08-09). This file **transcribes them verbatim** on 2026-08-10; nothing is
re-derived, no threshold is changed.

**State of the world at transcription time, stated exactly:**

- Qwen generation (18,464 rows, 8 long evals), gpt-5-mini labelling, the all-49-layer feature
  cache, and the per-token layer-23 extraction (in progress) are complete or underway on DoC.
- **No supervised ladder number exists.** R2, R2-desc, R3 and R4′ are fully unobserved.
- **The training-free floors have been partially observed**: the W6 Lehmer analysis
  (`prereg/lehmer_aggregation_qwen.md`, the project's working notes) computed and reported per-dataset
  `msp_min` and the sharpening-family endpoints (including `perplexity`) on all 8 Qwen datasets
  before this file was written. R1a/R1b's inputs are therefore partly seen at transcription time.
  This is recorded here rather than hidden; the R1 thresholds themselves predate every Qwen
  artifact (`9731fea`), and nothing in them was adjusted after W6.

**Population caption for every table** (corrected from §3's original, which said "layer selected
on dev" — superseded on 2026-08-09 (decision recorded in the project's working notes); **no dev split exists**):

> *ProbeDriftLong, 8 long evals × `cells_long`, 3 seeds, `Qwen/Qwen2.5-14B` (base), fp32 + eager,
> judge label gpt-5-mini, layer 23 by the fixed rule `ceil(N/2) − 1` (the convention; Llama ran
> layer 15 by the same rule).*

**Unit of analysis is the DATASET (n = 8), not the cell**, for every test below.

## 1. The claims (verbatim from the project's planning notes, commit `9731fea`)

| id | claim | Llama value | replicates iff | fails iff |
|---|---|---|---|---|
| **R1a** | msp_min > msp_sum on long-form OOD | +0.1310, 8/8, p=0.0078 | one-sided Wilcoxon p < 0.05, same sign | p ≥ 0.05 or sign flips |
| **R1b** | msp_min > perplexity | +0.0686, 5/8, p=0.46 | — | **registered as UNESTABLISHED on Llama.** Reported descriptively on both models; not a replication target |
| **R2** | Regress SAPLMA on msp_min across the 8 datasets; H0: slope = 1 | b=0.242, se 0.195, one-sided p=0.0041 | b significantly < 1, one-sided, α=0.05 | fail to reject |
| **R2-desc** | sd of PRR across datasets: probe < free baseline | msp_min 0.164, SAPLMA 0.088, pooler 0.064 | sd(probe) < sd(msp_min) | otherwise |
| **R3** | ID→OOD reversal as DiD, clustered by eval | +0.0540, 7/8, p=0.0156 | DiD > 0, Wilcoxon p < 0.05 | p ≥ 0.05 or sign flips |
| **R4′** | wMSP@2 ≥ SAPLMA on the two hardest rungs | +0.0022 / +0.0228, p=1.00 / 0.64 | **directional only** — report mean, win-count, per-eval spread | n/a — no inferential claim is made |

## 2. Caveats (carried verbatim, per §3's instruction)

- **R2:** b is *also* not distinguishable from 0 on Llama (95% CI **[−0.236, +0.720]**,
  p(b>0) = 0.131). The claim is **"the probe does not track the baseline"**, NOT "advantage
  compresses at rate 0.24". Do not report a point estimate as if it were pinned down at n = 8.
- **R2:** the shared-term artefact also invalidated the attention-pooler version (observed −0.924
  vs an independence null of −0.932). The slope form is used for **every** method, not just SAPLMA.
- **R1:** state explicitly that the earlier "32 OOD cells" framing was 4× pseudo-replication.
- **R4′:** state that R4 had no power on Llama, so a Qwen "non-replication" of R4 would carry no
  information. Registered *before* seeing Qwen so it cannot be claimed afterwards.
- **Anti-favourable-result guard:** if **all four** replicate cleanly, treat it as a suspected
  configuration leak and re-verify that no Qwen test PRR was inspected before the layer was fixed.

## 3. Configuration policy (from the project's planning notes, unchanged)

- **PRIMARY (A):** transfer Llama's configuration **unchanged** — same shrink λ (2.0 / 10.0), same
  β = 1.0, same temperature-selection procedure, same pooler recipe, same SAPLMA recipe.
- **SENSITIVITY (B):** any re-selection happens on held-out data only and is reported separately,
  never pooled. (The dev-split machinery was never built; the layer is fixed by rule, so no
  selection of any kind has occurred.)
- **FORBIDDEN (C):** looking at Qwen test PRR, adjusting, looking again. Banned outright; any
  deviation is recorded, not improvised.
- The layer does not transfer numerically and is fixed by the rule above, not by selection.
- The capped-draw divergence (Qwen probes drawing a different random subset of the same OOD pools)
  is a closed, documented noise source (an extra seed's worth of noise on 28 capped OOD cells; eval
  rows, floors and ID pools are identical) and is not to be "fixed" mid-replication.

## 4. What is scored

The Tier-1 paper method set on the complete grid — 8 evals × 5 `cells_long` rungs × 3 seeds
(seeds 1, 2, 3): `msp_min`, `perplexity`, `msp_sum`, `fair_floor` (footnote only), SAPLMA,
mean-pool control, attention pooler, `wMSP-norm`, `wMSP-shrink@2`, plus P(True) and Lookback as
baselines (the project's planning notes). Coverage is stated per table; a missing cell is
named, never silently absent. expertqa/factscore are scored on their judge-covered subsets
(expertqa 1603/2016, factscore 466/500 on this population) — a null label is the judge declining
(`uncovered = 1.0`), not incomplete labelling, and captions must say so.
