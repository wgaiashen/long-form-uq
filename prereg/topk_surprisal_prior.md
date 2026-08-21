# Pre-registration — the top-k surprisal prior

**Written 2026-08-05, before the builder was implemented and before any cell was run.**

---

## What it is, and why it is the tightest analysis-to-method link in the project

PART C established a taxonomy from the `topk_floor_sweep`: how many tokens the error signal occupies
differs by dataset (pubmed k=1 CONCENTRATED, cnn k=`all` SPREAD, med_quad k=10 mixed). That is currently a
*descriptive* finding about the unsupervised floors.

**The top-k surprisal prior turns it into a method.** The prior places mass on the **k highest-NLL
(most surprising) generated tokens** and (near-)zero elsewhere, then the attention pooler either *is* that
prior (arm C) or is tilted by it (arm D). So the pooler is pointed at exactly the tokens the taxonomy says
carry the signal.

It is deliberately a different point in target-space from the existing arms:

| target | sparse? | scale-free? | independent of the probability family? |
|---|---|---|---|
| `content_mass` | no (~half of positions) | n/a | yes |
| `nll` | **no** (dense, every position) | **no** (raw magnitudes) | no |
| **`topk`** | **YES** (k of G) | **YES** (uses RANK of NLL, not magnitude) | no |
| `orgad` | yes | yes | **yes** |

**`topk` fixes BOTH defects of the `nll` prior at once** — density and scale — while holding the
information source constant. So `topk` vs `nll` is a clean two-factor test: if `topk` beats `nll`, it is
the *sparsity/scale* of the target that matters, not the information in it. That contrast is the reason to
run it, and it is why `nll` must be re-run in the same job as the paired reference rather than compared to
the existing table.

## THE k-SELECTION TRAP, AND HOW IT IS HANDLED

`topk_floor_sweep` chose k per dataset **by PRR**, i.e. **using that dataset's labels**. Using those k
values at the eval dataset would leak eval labels into the method and make every number an oracle. This is
the R1 circularity and the D10 selector-provenance concern in a new place.

**Registered:**

- **PRIMARY — `topk_fixed`.** ONE global k for every dataset, selected on a validation slice carved from
  the **TRAINING POOL ONLY**, exactly as the temperature already is. No eval-dataset labels anywhere.
  **This is the number that may be reported as a method result.**
- **SECONDARY — `topk_oracle`.** Per-dataset k taken from the sweep. **Declared an ORACLE ceiling**, must
  always be labelled as such, and may never be quoted as a method result. Its only job is to bound how
  much a perfect per-dataset k could buy — i.e. whether a label-free k-selector would be worth building.

If PRIMARY ≈ SECONDARY, per-dataset k is not worth chasing. If SECONDARY is far above, that is a
motivation for a label-free k-selector and nothing more.

## Design

- **k grid:** `{1, 5, 10, 25, 0.10, 0.25, 0.50}` — absolute counts and fractions of G. Fractions are
  resolved per example as `max(1, round(f*G))`, so a fraction means the same thing on a 32-token pubmed
  answer and a 219-token expertqa one.
- **`floor`:** the prior is `1.0` on the top-k positions and `floor` elsewhere. `floor=0.0` is a hard
  filter. **Registered design note:** with `floor=0` arm D's additive `beta*log(prior)` clamps to
  `log(1e-9) ≈ -20.7`, so **any β > 0 hard-masks the non-top-k tokens and β stops interpolating**. Two
  floors are therefore run — **`floor=0.0` (hard filter, the Orgad-style filter-then-aggregate) and
  `floor=0.05` (soft, so β retains its meaning)** — and the difference between them is itself informative
  about whether hard filtering or soft tilting is the right mechanism.
- **Arms:** the existing `armC` (frozen prior: attention IS the renormalised prior) and `armD`
  (prior-tilted learned query), against the existing `armA`/`armB`/`floor_min` in the same job so every
  comparison is paired and same-population.
- **Population:** the full ProbeDriftLong `cells_long` grid, all 5 rungs, 3 seeds. **Not a subset.**

## Controls that must pass before any number is read

1. **k ≥ G degenerates to uniform.** With k = G and floor irrelevant, the prior is uniform, so `armC` must
   reproduce `armB` (mean-pool) to ~1e-6. If it does not, the builder or the renormalisation is wrong.
2. **k = G reproduces the existing `armD:nll`?** No — deliberately NOT asserted, because `topk` at k=G is
   uniform whereas `nll` is magnitude-weighted. Stating this so the absence of that assertion is not read
   as an oversight.
3. **Sparsity is real.** Report the realised mean fraction of non-zero prior mass per dataset; it must
   match k/G. A builder that silently returns uniform is the recurring failure mode in this project.
4. **`armA` reproduction gate.** The driver already halts if `armA` drifts from the recorded §C.3 value;
   that gate stays on and is the proof the job is on the right population.
5. **No fallback may be silent.** A row that cannot produce a genuine top-k weighting (e.g. G=0) is
   counted and reported, never quietly made uniform.

## Registered predictions

1. **`topk` beats `nll` on the CONCENTRATED datasets (pubmed, med_quad, factscore) and not on the SPREAD
   ones (cnn, asqa).** That is the taxonomy's own prediction applied to the method, and it is the result
   that would make the analysis-to-method link real.
2. **Small k wins on pubmed (its sweep k is 1) and large k on cnn (`all`).** If the validation-selected
   fixed k lands mid-range and the per-dataset pattern does *not* track the sweep, the taxonomy does not
   transfer from floors to probes, which is itself worth reporting.
3. **I do not expect the fixed-k primary to beat SAPLMA's +0.241 OOD mean.** The honest target is beating
   the paired `armD:nll` (+0.222) and `armA` (+0.222) on the concentrated subset. Both `topk` and `nll`
   point a hidden-state probe at where the *probability* family finds signal, so they import a ceiling
   already measured at `msp_min` = +0.186 OOD.

## Decision rule, fixed in advance

- **CARRY FORWARD** if `topk_fixed` beats the paired `nll` prior by **> 0.02** (the aggregation-axis noise
  floor) on the cross-dataset OOD mean, **or** on the concentrated subset with the spread subset not
  worse. Then it enters the routing arm set for A2.
- **DROP** if it is within ±0.02 of `nll` everywhere. Sparsity and scale-freeness then do not matter for
  this target, which narrows the Phase 2 target list to Orgad (the only independent one) and is a useful
  negative.
- **A favourable result is a suspect.** If `topk` wins, the check to run before believing it is the
  **k = G uniform control** (does the win vanish when the prior is degenerate?) and the **shuffled-position
  control** (top-k positions permuted within the example, holding sparsity fixed and destroying only
  *which* tokens are selected). A win that survives a positional shuffle is not a win.
