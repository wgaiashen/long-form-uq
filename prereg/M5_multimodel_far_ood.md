# PRE-REGISTRATION — M5: multi-model replication of the wMSP shrinkage effect under cross-task shift

> **Filename note.** The planning document called this `M3_multimodel_far_ood.md`. `M3` and `M4` were
> already taken in this directory (`M3_split_rule_and_coverage.md`,
> `M4_qwen_coverage_divergence_prediction.md`), so it is registered as **M5**. Nothing else changed.

## 0. Provenance — read this first

Written and committed on **2026-08-15**, **before any record, feature, per-token cache, label or
ladder number exists for any of the three replication populations**. No weights had been downloaded
for them at the time of writing.

**State of the world at registration time, stated exactly:**

- The three replication populations (§1) have **never been run through this pipeline** in any form.
  Nothing about them has been observed — not generations, not degeneracy rates, not floors, not
  supervised numbers.
- The two development populations are fully observed. Their published numbers are the *motivation*
  for this registration and are **not** evidence for it.
- The six-dataset reduced panel (§2) and the rung restriction (§3) were fixed by the author on
  2026-08-15 on **measurement-validity and cost grounds only** (§2.1), before any new-model artifact
  existed. No alternative subset was scored and compared.
- The shrinkage coefficient is **transferred, not selected**: λ = 2, fixed. See §4.

**Population caption for every table produced under this registration:**

> *Reduced six-dataset ProbeDriftLong multi-model replication panel — `pubmed_qa`, `xsum`,
> `cnn_dailymail`, `samsum`, `asqa`, `factscore`; rungs ID / DiffTask-long / 1ds-Diff-long; 3 seeds;
> layer by the fixed rule `ceil(N/2) − 1`; judge label gpt-5-mini.*
> **This is not the eight-dataset ProbeDriftLong benchmark and must never be captioned as such.**

**Unit of analysis is the DATASET (n = 6) within a model**, and the **MODEL** across populations.
Raw model × dataset cells are never pooled into one significance test.

---

## 1. Populations

**Development populations** — the populations the hypothesis was formed on. They are development
evidence and are **not** counted as confirmatory replications.

| model | layers → layer | hidden | dtype |
|---|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B` | 32 → 15 | 4096 | fp32 |
| `Qwen/Qwen2.5-14B` | 48 → 23 | 5120 | fp32 |

**Replication populations** — pre-specified, never used to select anything.

| model | layers → layer | hidden | dtype | cluster |
|---|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 32 → **15** | 4096 | fp32 | RCS |
| `google/gemma-2-9b-it` ⛔ **superseded — see D1** | 42 → **20** | **3584** | bf16 + eager | RCS |
| `Qwen/Qwen2.5-32B` | 64 → **31** | 5120 | bf16 | DoC |

⛔ **This table records what was REGISTERED, and is deliberately not rewritten.** `gemma-2-9b-it` was
withdrawn on 2026-08-15 and replaced by `google/gemma-2-9b` (base) — see **deviation D1 in §8** for
the evidence and reasoning. A pre-registration is an audit trail: changes are recorded as deviations,
never edited into the original text.

Layer indices follow the fixed rule `ceil(N/2) − 1` and are **never re-selected from results**.
⚠️ Gemma is **20**; the discarded 2026-06 Gemma-2-9B-It run used 21 under the repo's other
convention (`03_probe.py`'s `n_layers//2`). That is not a precedent and does not license a change.

Layer counts and hidden sizes were read from each checkpoint's `config.json` on 2026-08-15.

**Interpretation limit, registered in advance.** The panel supports Llama base vs Instruct, Qwen 14B
vs 32B, and Gemma as an additional family as **matched robustness checks**. It does **not** isolate
the causal effect of instruction tuning or of scale: checkpoints differ in more than the attribute
being varied, and Qwen 14B vs 32B additionally differs in dtype. No causal language will be used.

---

## 2. The reduced panel

Six evaluation datasets: **`pubmed_qa`, `xsum`, `cnn_dailymail`, `samsum`, `asqa`, `factscore`.**
Dropped: **`med_quad`**, **`expertqa`**.

### 2.1 Why these two were dropped — fixed before any new-model artifact existed

- **`med_quad`: measurement validity.** On the accepted Llama base population it shows **47.8% of
  generations carrying an invented follow-up question** that the judge nonetheless scores,
  **12.33% severe degeneracy**, and **97.7% capped**. It additionally carries the answer-span
  sensitivity recorded in `STOCKTAKE_cleanspan.md`, including a standing DO-NOT-CLAIM on its own
  `msp_min` ranking.
- **`expertqa`: cost and a known confound.** ~29% of the entire eight-dataset generation token load,
  plus the length/degeneracy confound in `QWEN_GENERATION_VALIDITY_AUDIT.md`.
- **`cnn_dailymail` is deliberately RETAINED.** It is behaviourally unusual but
  measurement-valid (0.1% fabricated, 0.03% severe). An unusual but valid aggregation regime is
  scientifically informative and is not a reason for exclusion.

### 2.2 Generation regimes — part of the population definition

- `pubmed_qa, xsum, cnn_dailymail, samsum` → default regime, **no repetition penalty**
- `asqa` → `--prompt-regime asqa_rp12 --repetition-penalty 1.2`
- `factscore` → `--prompt-regime factscore_rp12 --repetition-penalty 1.2`

Labelling: `02_label.py --judge gpt-5-mini` (passed **explicitly**) for the first four;
`02_label_factscore.py --prompt-regime factscore_rp12` for factscore. Judges are never mixed within
a comparison.

---

## 3. Rungs and cell counts

**Stage A (primary, must finish): `ID`, `DiffTask-long`, `1ds-Diff-long` → 18/18 cells per model.**

The progression is *matched distribution → cross-task transfer with broad support → cross-task
transfer with narrow single-source support*. Existing results indicate estimator differences become
most visible under cross-task shift, which is why ID plus the two cross-task settings are the
pre-specified reduced panel. These rungs are **not** chosen as the ones where any method is expected
to win.

**Stage B (opportunistic, may be omitted): `SameTask-long`, `LOO-long` → 11/12 cells per model.**
`factscore/SameTask-long` is **OMITTED and left genuinely unmeasured**: dropping `expertqa` makes the
`factuality` family a singleton. It will be reported blank, never filled with a substitute setting.

**Total 29/30 cells per model.** Computed from `cells_long`, not assumed.

Effect of the six-dataset restriction on rung composition, relative to the eight-dataset grid:

| rung | composition |
|---|---|
| `ID` | **identical** for all six evals |
| `1ds-Diff-long` | **identical** for all six evals |
| `SameTask-long` | identical for xsum / cnn / samsum; changed for pubmed_qa / asqa; omitted for factscore |
| `DiffTask-long` | changed for all six (1800 shared across fewer sources) |
| `LOO-long` | changed for all six (LOO over 5, not 7) |

**Consequence, registered:** development-model `ID` and `1ds-Diff-long` numbers transfer directly and
serve as the correctness control (§7). Only `DiffTask-long` requires re-scoring on the development
models, and that re-score is **not** a new result — it is a restriction of an existing one.

---

## 4. Methods — fixed, and λ is transferred not selected

Six methods: `perplexity` (mean token NLL), `msp_min`, `SAPLMA`, the attention pooler,
`wmsp_norm` (λ = 0), `wmsp_shrink2` (λ = 2).

**No λ grid is run on any replication population, and no model-specific λ selection is performed.**
λ = 0 is the unregularised control and λ = 2 the pre-specified regularised configuration, both
carried over unchanged from the development populations. This is the entire reason the new
populations are informative: they have never been used to choose the shrinkage strength.

λ = 1.5 may be reported later as a **supplementary sensitivity only**, after the primary fixed-λ
result is complete, and may not alter the primary conclusion.

---

## 5. Hypotheses

### 5.1 Primary — does the shrinkage intervention itself transfer

For each replication population, and separately for each of `DiffTask-long` and `1ds-Diff-long`:

> **Δ_shrink = macro mean over the six datasets of [ PRR(wMSP-shrink@2) − PRR(wMSP-norm) ]**

> **Directional model-level replication occurs when Δ_shrink > 0.**

No additional per-dataset bar (such as 4/6 or 5/6 positive) is imposed. Strength is conveyed by the
reported statistics in §5.3, so a small macro carried by two datasets will transparently read as weak
while a broad gain will read as strong.

**Motivation, stated accurately.** Existing development evidence shows a broad macro-OOD shrinkage
benefit of approximately **+0.090 PRR on Llama** and **+0.117 on Qwen**. These are **broad macro-OOD
figures over the eight-dataset grid — not** the exact DiffTask-long / 1ds-Diff-long statistic being
replicated here. Matched values will be filled in once the reduced six-dataset development re-scoring
lands; that must not block launch and does not change any threshold in this file.

### 5.2 Secondary — competitive comparisons, reported separately

> **Δ_SAPLMA = macro over six datasets of [ PRR(wMSP-shrink@2) − PRR(SAPLMA) ]**
> **Δ_attention = macro over six datasets of [ PRR(wMSP-shrink@2) − PRR(attention pooler) ]**

on the same two rungs, per population.

⛔ **These are never collapsed into a synthetic `max(SAPLMA, attention)` comparator.** Which learned
baseline is stronger is itself model-dependent and must remain visible. If shrunk wMSP exceeds both,
the report may say it is the strongest among the compared learned methods — no more.

⚠️ Standing context: `wMSP-shrink@2 > SAPLMA` is a **DO-NOT-CLAIM** on the development populations
(`CLAUDE.md`). Nothing in this registration promotes it. It is registered as **secondary and
descriptive**.

### 5.3 Required reporting, per population and per rung

All of: macro mean Δ · median Δ · number of positive datasets out of six · the six per-dataset
deltas · a dataset-bootstrap CI for the macro where the tooling already supports it cleanly ·
Wilcoxon **descriptively only** at n = 6, never as the verdict.

Final table, one row per population:

| model | Δ_shrink DiffTask | pos/6 | Δ_shrink 1ds-Diff | pos/6 | Δ_SAPLMA DiffTask | Δ_SAPLMA 1ds-Diff | Δ_attention DiffTask | Δ_attention 1ds-Diff |

### 5.4 Cross-model inference — the ceiling on what may be claimed

- **3/3 replication populations positive** on a rung is reported as *"the direction reproduced on all
  three pre-specified new model populations"*. Under a one-sided sign test this is **p = 0.125**:
  **directional replication, not conventional significance.**
- **2/3** is reported as partial replication; **1/3 or 0/3** as failure to replicate.
- ⛔ **No attempt will be made to reach p < 0.05 by pooling the five populations.** The development
  models are not fresh confirmatory evidence, and five deliberately chosen, partly family-related
  checkpoints are not five exchangeable samples. The all-five view is **descriptive consistency
  evidence only**: *"the same direction was observed across all five tested model populations."*
- A non-replication is a **reportable finding**, not a failure of the experiment, consistent with §7
  of the 14 August meeting notes.

---

## 6. Prompt-regime validity gate — decided per instruct model, before any PRR

**Pre-specified and UQ-performance-independent. PRR is never inspected to make this choice.**

**Smoke:** ~**40 deterministic rows per dataset × 6 datasets = 240 rows per model**, selected by fixed
index (not sampled), under an isolated `--prompt-regime <slug>_probe` so smoke caches never touch
real ones. Manual inspection set fixed in advance: **the first 20 records by index**.

Measured with `scripts/checks/generation_quality.py --model <M> --datasets <D> --out <scratch>`.
⚠️ `--out` is mandatory — the default path writes a tracked results file.

| signal | column | raw few-shot passes if |
|---|---|---|
| degeneracy, severe | `pct_severe` | ≤ 5.0 |
| degeneracy, degraded | `pct_degraded` | ≤ 10.0 |
| empty / malformed | `pct_empty` | ≤ 1.0 |
| invented continuation / template restart | `pct_fabricated` | ≤ 10.0 **and** `mean_answer_frac` ≥ 0.90 |
| extreme cap behaviour | `pct_capped` | ≤ base-model value + 20 points on the same dataset |
| manual inspection | — | fixed 20-record set read before any scoring |

Thresholds are anchored to the accepted Llama base population, measured 2026-08-15 on the four
default-regime panel datasets: `pct_severe` ≤ 1.11, `pct_fabricated` ≤ 0.1, `mean_answer_frac`
1.000, `pct_empty` 0.00. The detector is `luq.degeneracy`, which is deliberately **not**
repetition-based.

**Decision rule, per model, independently.** If a model passes on all six panel datasets, it keeps
**raw few-shot**. If it fails, that model uses **its own native chat template**. One instruct model
failing does **not** move another onto a chat template. If the instruct populations end up on
different valid regimes, that is recorded explicitly and no causal claim about instruction tuning is
made.

---

## 7. Verification gates that must pass before any number is recorded

1. **The free control.** `ID` and `1ds-Diff-long` pool specs are identical between the reduced and
   full grids, so the development-model reduced re-score **must reproduce the published `ID` and
   `1ds-Diff-long` numbers exactly** (`results/pdl_master__meta-llama_Meta-Llama-3.1-8B.csv`, md5
   `9aa5643d2e8864badbb6165a141a4be0`). Compared on **per-example score vectors**, not PRR. Any drift
   means the restriction is implemented wrongly, and stops the experiment.
2. **Cell-count assertion.** `probedriftlong.py` prints `"<d>: no pertok cache -> skip"` and
   continues, so a missing dataset silently shrinks the grid. Assert **18** Stage-A cells and **29**
   overall per model before recording. A blank means not measured; a number means measured; the two
   must never be confusable.
3. **`set_special_ids()` fired** for every new tokenizer, with a non-empty id count in the log — not
   the Llama `id >= 128000` fallback, which would silently zero ordinary content tokens.
4. **Hidden-dim registry** matches the checkpoint (fail-loud; gemma-2-9b-it is 3584).
5. **Feature / per-token consistency** (`feature_pertok_consistency.py`) passes for every
   (model, dataset); `01e_repool` is mandatory after `01_extract`.
6. **Positional alignment**: pertok window `[last_prompt_token] + gen_tokens` = G+1 against G NLLs;
   assert, never pad or trim silently.
7. **Explicit dtype and attention backend** on every run, stamped in provenance with model, layer and
   prompt regime.
8. **Judge coverage** recorded per dataset as judged/total. Declines are data, not gaps; never write
   0 for not-measured.
9. **Generation-validity monitoring on the real generations as they land**, not only on the smoke.

---

## 8. Deviations

Any deviation from this file is recorded here with its date and reason, before the affected number is
reported. Adding a method, changing λ, changing the dataset panel, changing a rung, or changing a
threshold after any replication-population PRR has been observed invalidates the primary claim, which
must then be reported as exploratory.

---

### D1 — `google/gemma-2-9b-it` → `google/gemma-2-9b` (base). 2026-08-15.

**Recorded before any PRR was computed on any replication population.** No supervised number, no
floor, and no ladder cell existed for either Gemma checkpoint at the time of this decision. The
change was made on **generation-validity evidence only**, which is the same standard §6 sets for the
prompt-regime choice.

**What was measured.** `gemma-2-9b-it` under raw few-shot, on real generations (not the smoke):

| dataset | n | finding |
|---|---|---|
| `pubmed_qa` | 200 | **100% EMPTY** — every record is a bare `"\n"`, 1 token |
| `samsum` | 400 | **82.2% assistant chatter**; real answer only 69% of saved text |
| `cnn_dailymail` | 200 | clean (0.5% chatter) |
| `asqa` | 200 | clean (0.0% chatter) |

**Mechanisms, both specific to instruction tuning:**
1. *Whitespace-fronting.* An instruct model under raw prompting emits a leading newline. `pubmed_qa`
   generates with `--truncate-long` (Joe's `generate_until=['\n']`), so the newline truncates the
   entire answer away. The failure is total and silent — the job exits 0.
2. *Assistant persona.* The model answers correctly, then continues ("Let me know if you'd like me to
   analyze any other text!"). The judge scores the whole saved output, so the filler is graded as if
   it were the summary — the same measurement-validity defect that removed `med_quad` from this panel
   (§2.1).

**Why base rather than a chat template.** Post-hoc truncation is unavailable — `luq.template_restart`
covers template restarts and base-model pretraining artefacts, not assistant persona, and it is by
design "strictly MODEL-AGNOSTIC", so adding a rule that fires on one checkpoint would be retuning on
a finding. The native chat template was **not built** at the time of the decision.

> **⚠️ AMENDMENT, same day, before any PRR.** This deviation as first written claimed §6's
> chat-template fallback "has no working code path", reasoning from `probe_drift`'s `instruct=True`
> covering only 3 of the 6 panel datasets. **That inference was wrong.** §6 specifies the model's
> *native* chat template, and `tok.apply_chat_template(...)` acts on the **final prompt string**, so
> it is dataset-agnostic across all six, custom loaders included. Both instruct checkpoints ship a
> `chat_template` (verified). It is ~5 lines in `01_extract`.
> **What this does and does not change.** The chat template was genuinely *not implemented*, so it
> was not an available option on the day — but "not built" is not "cannot be built", and the original
> wording overstated the constraint. The decision itself was the author's, taken on the
> generation-validity evidence in the table above, which is unaffected. The swap's remaining
> justifications also stand independently: `gemma-2-9b` is regime-matched to both development
> populations, and Gemma's role in the panel is the third FAMILY, not the instruct axis.
> Recorded here rather than silently edited, because this file is the audit trail.

**What the swap costs and does not cost.** `gemma-2-9b` is the same family, size, hidden width (3584)
and probe layer (20). Gemma's role in the panel is the **third model family**; the instruct axis is
carried by `meta-llama/Llama-3.1-8B-Instruct`, which is unaffected. The base checkpoint is also
**regime-matched to both development populations**, which are base models — so this deviation makes
the panel more internally consistent, not less.

⛔ **`gemma-2-9b-it` is WITHDRAWN.** Its partial caches are not to be scored, promoted, or reported.
It stays in the hidden-dim and layer registries only so that a stale cache fails loudly rather than
being silently mistaken for the base population.

---

### D2 — `gemma-2-9b` exceeds the `pct_severe` gate on two datasets; ACCEPTED by the author. 2026-08-15.

**Recorded before any PRR was computed on any replication population.**

`gemma-2-9b` (base) clears the two failures that withdrew the instruct checkpoint — `pubmed_qa` 0.0%
empty (was 100%), `samsum` 9.5% chatter (was 82.2%) — but exceeds the §6 `pct_severe` ceiling of
**5.0** on two datasets, measured on the first 200-400 records of the real run:

| dataset | `pct_severe` | what the detector is actually flagging |
|---|---|---|
| `pubmed_qa` | **6.0%** | HTML markup: `'<strong>Yes</strong>.'`, `'<b>Yes</b>.'` — the **answer is correct**, the formatting is web-scraped. Fires the detector's code-density branch (`{}<>`), by design |
| `asqa` | **7.0%** | genuine web / pretraining artefacts, e.g. *"The answer to this puzzle was submitted by `<b>David</b>` and it can be found on page #2354"* |

For calibration: Llama base is ~1.1% on `pubmed_qa`, and the 5.0 ceiling was anchored to it with
headroom, fixed before any Gemma number existed.

**§6 HAS NO REMEDY FOR A BASE MODEL, and that is a gap in the registration.** Its decision rule is
"if raw few-shot fails, use that model's native chat template" — which presupposes an instruct
checkpoint. A base model has no alternative regime, so "fail" is undefined for it. Registering the
gap rather than quietly reinterpreting the rule.

**Author's decision: ACCEPT `gemma-2-9b` and report the numbers prominently.** Reasons: the flagged
content is largely correct with anomalous formatting rather than degenerate; 6-7% is far from the
catastrophic failures that withdrew the instruct checkpoint (100% / 82%); and the ceiling was
calibrated on Llama's unusually clean output, so it is a strict bar for a different family. Gemma's
role in the panel is the third FAMILY, and this markup behaviour is a real property of it.

**Binding conditions on this acceptance:**
1. ⛔ **The threshold is NOT moved.** `wmodels_gate.py` continues to report `FAIL` for these
   datasets. This is an explicit, recorded override, not a re-tuned gate — so the gate never
   misreports what it measured.
2. `pct_severe` for `gemma-2-9b` **must be reported in the results table**, not relegated to a
   footnote, wherever this population appears.
3. The distinction between `pubmed_qa` (cosmetic markup) and `asqa` (genuine artefacts) is stated,
   not averaged away.
4. If a Gemma-specific result later depends on `asqa`, this acceptance is revisited before that
   result is claimed.
