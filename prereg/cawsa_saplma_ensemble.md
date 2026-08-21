# Pre-registration — does CAWSA supply a complementary signal to SAPLMA?

## 0. Provenance and honest chronology — read this first

Written and committed on **2026-08-17**, before any `CAWSA λ=2 + SAPLMA` ensemble PRR exists anywhere
in the project.

**This is a late, supervisor-motivated experiment.** It follows the 14 August 2026 supervision meeting
(the supervision meeting notes and §15 P0 #2), where the
best-of-both-worlds idea was raised. The registration is therefore explicit about what is and is not
being fixed prospectively.

**What IS fixed prospectively (§2-§6):** the component pair, the combiner, the population, the primary
estimands, the unit of analysis, the interpretation rule, and the commitment not to sweep λ or combiners.

**What is NOT claimed:**

- **The ensemble hypothesis did not exist before 14 August 2026.** It came from the supervisors.
- **CAWSA was not invented for this experiment.** It is the project's existing method contribution,
  developed and published internally well before this registration
  (the project's working notes, `results/analysis/WMSP_SHRINKAGE_MECHANISM.md`).
- **The component results were NOT unseen.** SAPLMA, unconstrained activation weighting, CAWSA λ=2, the attention pooler and the
  three floors are all fully observed on this exact population and are quoted in §1 below. They are
  the *motivation* for this registration, not evidence for it.
- **Two ensembles have already been run on this grid** and are also quoted in §1, one of which is
  stale (§1.2). What has **never** been formed is `CAWSA λ=2 + SAPLMA`, which is the subject here.

**Population caption for every table produced under this registration:**

> *Complete ProbeDriftLong long-form grid — `pubmed_qa`, `med_quad`, `asqa`, `xsum`, `cnn_dailymail`,
> `samsum`, `expertqa`, `factscore` × rungs ID / LOO-long / SameTask-long / DiffTask-long /
> 1ds-Diff-long; `meta-llama/Meta-Llama-3.1-8B`, layer 15, seeds 1,2,3; 40/40 cells.*

**Unit of analysis is the DATASET, n = 8.** The 32 OOD cells are never treated as 32 independent
observations: the unsupervised floors do not depend on the training pool, so their PRR is identical
across all four OOD rungs within an eval, and cell-level pooling is 4× pseudo-replication.

---

## 1. State of the world at registration time, stated exactly

### 1.1 Observed component performance (canonical `pdl_master__meta-llama_Meta-Llama-3.1-8B.csv`)

Macro over the 8 datasets. Rung order is the report / Hidden Failures convention.

| method | ID | LOO | SameTask | DiffTask | 1ds-Diff | OOD macro |
|---|---|---|---|---|---|---|
| `msp_min` | +0.186 | +0.186 | +0.186 | +0.186 | +0.186 | +0.186 |
| **SAPLMA** | **+0.587** | +0.237 | +0.316 | +0.203 | +0.209 | +0.241 |
| attention-pool (armA) | +0.623 | +0.256 | +0.265 | +0.182 | +0.186 | +0.222 |
| unconstrained activation weighting (`wmsp_norm`) | +0.450 | +0.158 | +0.146 | +0.128 | +0.125 | +0.139 |
| **CAWSA λ=2** (`wmsp_shrink2`) | +0.521 | +0.240 | +0.238 | +0.205 | +0.232 | +0.229 |

### 1.2 Ensembles already run, and their audited status

`results/ensemble_wmsp_saplma_full__meta-llama_Meta-Llama-3.1-8B.csv`, written **2026-07-29**, carries no
provenance columns (stamping was introduced 2026-08-05). Audited cell-by-cell on 2026-08-17 against the
canonical master, whose rows come from `pdl_fam_*` files, so the comparison is not circular:

| component in the old artefact | agreement with canonical master, 40 cells |
|---|---|
| `saplma`, `floor_min`, `floor_ppl`, `floor_sum` | **max &#124;Δ&#124; = 0.0000** on every cell |
| `wmsp` (unconstrained activation weighting) | **13/40 cells differ**; max &#124;Δ&#124; = **0.5206**; mean &#124;Δ&#124; = 0.0342 |

The unconstrained activation weighting disagreement spans 8 evals and 4 rungs, and 12 of the 13 differences have the old value
*lower*. Cause: `src/luq/weighted_msp.py` received weighting corrections **after** the artefact was
written — `056f50b` (2026-08-03, "Fix NaN wMSP weights on asqa and make PRR refuse to score NaN") and
`a792e6e` (2026-08-05, segment-softmax all-excluded NaN). Worst cell ID/asqa: **−0.044 old vs +0.477
canonical**.

Therefore:

| old ensemble number | status |
|---|---|
| `rankavg{unconstrained activation weighting, SAPLMA}` OOD macro **+0.201**, ID +0.503 | **STALE.** Its unconstrained activation weighting leg predates the weighting fixes. Not a gate, not a target, not quotable. |
| `rankavg{msp_min, SAPLMA}` OOD macro **+0.273**, ID +0.493 | **CONFIRMED CURRENT** — recomputed on 2026-08-17 from the canonical `results/pdl_perex/` sidecars (which reproduce the master's SAPLMA on **40/40 cells at max &#124;Δ&#124; = 0.0000**) and it returns **OOD +0.273, ID +0.492**. Not stale. |

### 1.2b That reference ensemble does NOT survive this registration's own statistics

Recomputed under the §4/§5 analysis (dataset as the unit, n = 8) rather than as a macro difference:

| leg | macro | median | signs | bootstrap 95% CI | exact Wilcoxon p |
|---|---|---|---|---|---|
| OOD, `msp_min+SAPLMA` − SAPLMA | **+0.0322** | +0.0458 | **5/8** | **[−0.0294, +0.0899]** | **0.461** |
| ID, same | **−0.0947** | −0.0809 | **0/8** | [−0.1382, −0.0565] | **0.0078** |

So the +0.273-vs-+0.241 comparison that makes this ensemble look like it beats SAPLMA far OOD is **not
statistically supported once the dataset is the unit of analysis** — it rests on 5 of 8 datasets with a
CI spanning zero — while its ID loss is large, consistent (0/8) and significant. Under §6 this reference
would be classified as **no established OOD improvement, with a material ID loss**.

This is recorded *before* the primary is computable, and it is the reason §4 fixes the dataset as the
unit of analysis: a macro-difference reading of these ensembles is materially more favourable than the
paired test at n = 8. It also sets the honest prior for the primary — see §1.3.

**No old ensemble number is used as a continuity gate or as a pre-registered target.** The validity
gate is component-level (§7).

### 1.3 Prior evidence that bears on how a result here must be read

- the project's working notes **§9**: SAPLMA and the floor are genuinely distinct and stable — seed
  self-agreement 0.772 against cross-correlation 0.194, disattenuated 0.222. There is a real second
  signal.
- **§8**: every oracle ceiling previously quoted is a max-over-K artefact. The per-cell
  over-4-methods oracle has net **−0.0158, p = 0.994**. So **per-cell or per-dataset selection among
  methods has no signal to recover**, and none is attempted here.
- **C2** (fusing NLL into the probe as an extra input feature) measured incremental value at **+0.017**
  (27/32 cells, per-cell p = 0.0009; n = 8 datasets p = 0.109). **A small effect is the honest
  expectation.** A null is a valid and reportable outcome.

---

## 2. Method names

| report-facing name | implementation key |
|---|---|
| **unconstrained activation weighting** | `wmsp_norm` |
| **CAWSA (λ = 2)** | `wmsp_shrink2` |

On first mention in prose: *CAWSA (λ = 2, implementation key `wmsp_shrink2`)*. Implementation keys and
committed result files are **not** renamed. This registration was written while the method was called
CAWSA; the name was settled as CAWSA for the write-up, and the prose here uses the settled name.

---

## 3. The primary ensemble, fixed

**`rankavg{CAWSA λ=2, SAPLMA}`** — equal weighting, one fixed combiner, on the population in §0.

**λ = 2 is fixed and transferred, not selected.** λ ∈ {1, 1.5, 10, …} will **not** be tried and the
best ensemble picked. This is not a λ-selection exercise.

### 3.1 `rankavg` semantics, audited before adoption (2026-08-17)

`ensemble_ladder.rankavg` is `np.mean([rankdata(u) for u in us], axis=0)`. `rankdata` ranks **within the
vector passed in**, which is that cell's test cohort, so the score assigned to response *i* depends on
every other test response present. Measured on `xsum__DiffTask-long`, n = 2000:

- a fixed example scores 460.0 in the full cohort and 232.0 in a half cohort;
- **2 / 400** random pairs flip relative order between the full cohort and a 200-example cohort;
- PRR on the same 1000 rows is **+0.0866** with ranks recomputed in-subset vs **+0.0860** with ranks
  inherited from the full cohort, Δ **+0.0006**;
- `zavg` is cohort dependent too, through the cohort mean and sd, but more weakly.

**Registered interpretation.** This is a **rank-ensemble diagnostic**, which is what the meeting asked
for, and it is **not** presented as a deployable response-level estimator. The measured sensitivity is
small enough that PRR conclusions are very unlikely to be an artefact of cohort dependence, and that is
reported with the numbers above rather than assumed. **No fixed-normalisation follow-up is designed or
run under this registration.**

---

## 4. Primary estimand A — OOD benefit

For each of the 8 datasets:

1. average PRR across its four OOD rungs for `rankavg{CAWSA λ=2, SAPLMA}`;
2. average PRR across the same four rungs for SAPLMA;
3. delta = ensemble − SAPLMA.

**Unit of analysis = dataset, n = 8.** Reported:

- macro mean delta;
- median delta;
- sign count (how many of 8 datasets improve);
- **exact two-sided Wilcoxon signed-rank** over the 8 datasets;
- dataset-level bootstrap 95% CI on the macro mean delta;
- leave-one-dataset-out macro range (how far the macro moves when any single dataset is dropped).

---

## 5. Primary estimand B — ID cost

On the 8 ID cells, delta = `rankavg{CAWSA λ=2, SAPLMA}` − SAPLMA. Reported: macro mean, median, sign
count, bootstrap 95% CI.

### 5.1 The ID reference scale, derived from measured variability

No desired effect size is invented. From the existing per-seed sidecars in `results/pdl_perex/`,
SAPLMA's ID macro over the 8 datasets across the three seed sets is **+0.5889 / +0.5822 / +0.5905** →
**sd 0.0044, SE 0.0025**; mean within-cell seed sd at ID is **0.0114**.

So **0.0044 is the run-to-run reproducibility of an ID macro on this grid.** It is registered as an
*interpretive scale only*: an ID delta whose bootstrap CI lies entirely below **−0.0044** is larger than
run-to-run noise and counts as a **material** ID loss. This is not a pass/fail threshold and not a
target.

---

## 6. Interpretation rule, fixed before the result is seen

**"Best of both worlds" may be claimed only if BOTH legs hold:**

1. a convincing and reasonably consistent OOD improvement over SAPLMA (§4), and
2. no **material** ID loss (§5.1).

- OOD improvement **with** material ID loss → reported as a **trade-off**, not as success on both
  objectives. Still a useful result.
- No OOD improvement → **null**, which is a valid result and is reported as one.
- The ensemble is **not** required to win every rung for the result to be interesting (§9).

---

## 7. Validity gates, checked before the primary result is read or interpreted

1. **Component continuity.** unconstrained activation weighting, CAWSA λ=2, SAPLMA, attention-pool, mean-pool and the three floors
   from the new pass must reproduce their canonical `pdl_master` values within seed tolerance. This
   replaces any gate based on an old ensemble number.
2. **Sidecar integrity.** Same evaluation rows; equal vector lengths; 3 seeds where applicable;
   recomputed PRR agreeing with the in-memory value (the writer's own 1e-6 self-gate); the three floors
   showing seed self-agreement of exactly 1.0, since they are deterministic functions of cached
   logprobs.
3. **Coverage.** 40/40 cells, or the missing cells named up front. A partial grid is never reported as
   if it were the whole one.

---

## 8. Control ensemble — secondary mechanism evidence

**`rankavg{SAPLMA, attention-pool}`.** SAPLMA and attention pooling are both predominantly *learned
hidden-state* signals, whereas CAWSA keeps **token NLL as the scored quantity** and uses activations
only to reweight the token evidence. Two descriptive tests:

1. is {CAWSA λ=2, SAPLMA} **less correlated** than {SAPLMA, attention-pool}, using seed self-agreement
   as the reliability ceiling and reporting the disattenuated value `r_xy / sqrt(r_xx · r_yy)`;
2. does the {CAWSA λ=2, SAPLMA} pair **gain more** from combination than {SAPLMA, attention-pool}.

Registered caveat: **lower correlation by itself is not evidence of a better method.** §1.3 already
records that distinctness is not incremental value. This is **secondary mechanism evidence and forms no
part of the success criterion in §6.**

### 8.1 References / ablations

Recomputed from the **same new per-example vectors**, so that everything is internally comparable:
`rankavg{msp_min, SAPLMA}` and `rankavg{unconstrained activation weighting, SAPLMA}`. These are references, **never** entered into a
winner-selection sweep. If they happen to reproduce the old numbers in §1.2, that is noted as a
reassuring finding after the fact, not as a gate.

---

## 9. Rung profile

A compact rung-level table, plus a report-quality figure, for SAPLMA · CAWSA λ=2 ·
CAWSA λ=2 + SAPLMA · SAPLMA + attention, in the report / Hidden Failures rung order:

> **ID → LOO → SameTask → DiffTask → 1ds-Diff**

LOO and SameTask are **not** silently reordered. The scientific question this addresses is whether the
combination retains more of SAPLMA's performance close to ID while inheriting some of CAWSA's
robustness as shift strengthens; that profile may be more informative than the single OOD macro.

---

## 10. Anti-artefact commitments

- **One primary combiner** (`rankavg`, equal weight). `zavg` may appear as a robustness footnote only,
  never as a headline, and never as an alternative from which the better is chosen.
- **No per-cell or per-dataset selection among methods** — closed by §8 of
  the project's working notes (net −0.0158, p = 0.994).
- **No λ sweep, no combiner sweep, no pair sweep, no interpolator.**
- **No new generation, extraction or GPU job.** CPU-only, from existing caches.

---

## 11. HBO — secondary and conditional

Not part of this registration's primary or secondary analysis. Only after the simple ensemble is
finished and interpreted, and only as the existing **read-only saturation diagnostic**: report the
actual Mahalanobis **R distributions** and the fraction above the HBO threshold, per rung. Prior
short-form context: `frac_R>0.5 = 1.000` on all four OOD rungs
(`results/regime_R4b_hbo_validation__meta-llama_Meta-Llama-3.1-8B.csv`), where HBO collapsed to pure
MSP.

**No arbitrary cutoff (such as "90% on 3 of 4 rungs") is treated as a scientific result.**
**No interpolator is trained without returning to the author first.**

---

## 12. Report implications, fixed in advance

The dissertation narrative is **not** altered by this registration. **CAWSA remains the principal
method contribution.**

- If positive: *CAWSA provides a complementary probability-grounded signal that can improve a
  conventional hidden-state estimator under distribution shift.* The ensemble **strengthens** the case
  for CAWSA rather than replacing it as the thesis contribution.
- If null: *different rung-level robustness profiles do not automatically imply useful score-level
  complementarity under simple ensembling.*

---

## 13. Deviations and disclosure

Recorded here as they occur, numbered, with the date and the reason.

**D1 — what was observed before this file was committed (2026-08-17).** The analysis driver
`scripts/checks/complementary_ensemble.py` was smoke-tested against the pre-existing
`results/pdl_perex/` sidecars, which contain only the floors and SAPLMA. That test necessarily produced
the **reference** ensemble `rankavg{msp_min, SAPLMA}` numbers now recorded in §1.2 and §1.2b. Disclosed
because the chronology matters:

- the **primary** ensemble `CAWSA λ=2 + SAPLMA` was **not computable** on that population — unconstrained activation weighting, CAWSA λ=2 and the attention pooler are absent from those sidecars — and no primary or control number has
  been observed by anyone at the time of writing;
- the reference's headline value (+0.273) was already public in the project before this registration,
  from the 2026-07-29 artefact; what §1.2b adds is its **paired n = 8 statistics**, which had never been
  computed and which weaken rather than strengthen the ensemble premise;
- §4-§6 were written before §1.2b was computed, so the dataset-level analysis was not chosen to produce
  that outcome.

**D2 — a seed-handling bug in the driver, found and fixed before any primary result (2026-08-17).** The
first version of `complementary_ensemble.py` scored the **seed-averaged uncertainty vector** rather than
averaging the **per-seed PRRs**. Those differ: SAPLMA at LOO-long is +0.2369 the correct way and +0.287
the wrong way, a **+0.050 inflation** — larger than any effect this registration is looking for, and it
would have flattered the ensemble, since averaging rewards whichever score gets averaged. Caught by the
§7 continuity gate against `pdl_master`, which failed loudly. All reported quantities now use *mean over
seeds of the per-seed PRR*, and ensembles are combined **within** each seed before scoring, matching
`ensemble_ladder.py`'s long-standing convention. The module carries the measurement in a comment so the
trap is not re-entered.

**D3 — a crash in the analysis driver's final step, after all substantive output (2026-08-18).** The
first full-grid run of `complementary_ensemble.py` completed every gate, estimand, control, reference and
complementarity table, then raised `TypeError` on the closing `zavg` robustness footnote: a call site not
updated when `macro()` gained its `need` argument. It aborted **before** writing the output CSV. Fixed
(one line) and re-run; every number was reproduced identically and the `zavg` footnote now prints
(PRIMARY ID +0.592 / OOD +0.269; CONTROL ID +0.623 / OOD +0.245), agreeing with `rankavg` in direction and
magnitude. No result changed. Recorded because the outputs were observed before the file was written, and
the chronology should not have to be reconstructed later.

**D4 — outcome recorded (2026-08-18).** Primary `rankavg{CAWSA λ=2, SAPLMA}`: OOD macro **+0.0343**,
**7/8** datasets, CI **[−0.0092, +0.0765]**, exact Wilcoxon **p = 0.1484**; ID macro **+0.0079**, CI
**[−0.0130, +0.0304]**. Under §6 the OOD leg is **not established**: the gain is positive and
consistent in direction, but it does not clear the pre-registered significance bar at n = 8. The
ensemble is **not** promoted to a headline method, and CAWSA remains the principal method contribution.

> **Wording clarified 2026-08-21.** This outcome was first recorded as "the verdict is NULL", using
> §6's verdict category. That label reads as "no effect", which is not what was measured: the OOD gain
> is +0.0343 on 7 of 8 datasets and the combination beats both of its components on all five rungs.
> The numbers, the decision rule and the non-promotion are unchanged. Only the label is stated
> precisely, as "not established under the pre-registered bar".

