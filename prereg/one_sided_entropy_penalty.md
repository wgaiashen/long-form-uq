# Pre-registration — the one-sided entropy penalty

**Written 2026-08-01, BEFORE any run.** Committed ahead of the run so the predictions and the stopping
rule are timestamped rather than fitted afterwards.

---

## Why this is a genuinely different test, not a third variant

B.1 and B.2 both supervised the attention toward a **target**, and both found the same thing: the
target carries no information. `real ≈ shuffled` on every cell, across the paper's λ range, the
low-weight regime, the drop schedule, and every J of K.

**B.3 uses no target at all.** It constrains a *property* of the distribution — penalise the attention
only when it becomes too sharp, leave already-broad distributions untouched. So **the "a shuffled
target does just as well" failure mode structurally cannot arise here.** That is what makes it worth
running after two nulls rather than being a third attempt at the same idea, and it should be said that
way in the write-up.

**The form:** `L + λ · mean( relu(τ − H_norm)² )`, with `H_norm` the per-example attention entropy
normalised by `log T` (1.0 = uniform, 0 = all mass on one token). The hinge is the whole point: the
penalty is **exactly zero** for any example already at or above τ.

**A penalty that binds everywhere is just the two-sided entropy control we already ran and found
PRR-neutral.** So `frac_bound` (the fraction of examples where `H_norm < τ`) is reported per arm. If it
is near 1.0 the one-sidedness is nominal, not real, and the result must be read as the old control.

## Where it should bite

**pubmed is the only dataset with genuinely concentrated ID attention**: normalised entropy 0.61,
roughly half its mass on punctuation, peaked on an early comma at a position whose per-token error
signal is 0.41 — below chance. Every other dataset is already near-flat, so a one-sided penalty should
be close to a no-op there.

## Scope — narrower than B.1/B.2 assumed

B.3's *verdict* does not depend on regeneration (penalty-on vs penalty-off is a within-population
comparison), but the stable *footing* is smaller than it looks:

| cell | status |
|---|---|
| **pubmed ID** | **PERMANENT** — pubmed is not being regenerated. **This is the primary screen.** |
| pubmed SameTask / LOO / DiffTask | **not stable** — those pools contain med_quad, regenerating now |
| xsum, any rung | **undecided** — DoC's probe is deciding whether xsum regenerates at all |

**pubmed ID is sufficient.** The entire motivation is that pubmed is the only dataset with concentrated
ID attention. If a one-sided sharpness penalty does anything anywhere, it does it there,
in-distribution. **If it does not help on pubmed ID, it will not help anywhere, and no amount of
regeneration changes that.** xsum ID is included as the no-op control and labelled **provisional**
pending the probe.

## REGISTERED PREDICTIONS

1. **pubmed ID: helps**, if anything does. It is the only over-sharp case.
2. **xsum ID: no-op.** Its attention is already broad, so the hinge should not bind and PRR should not
   move.
3. **A uniform effect across both is SUSPICIOUS** and must be checked for an artefact before being
   believed — most likely a τ set high enough to bind on distributions that were never over-sharp.
   `frac_bound` on xsum is the check: if it is high, the threshold is the problem, not the finding.

## Controls and reporting

- **λ=0 reproduction gate**: at λ=0 every existing ladder result must reproduce **exactly**. Same gate
  B.1 and B.2 passed.
- **Sweep τ as well as λ**, and report the selected τ. τ is selected on a validation slice carved from
  train, never on test.
- **Report realised attention entropy per arm** (`ent_after`), so we can confirm the penalty did what it
  claims *independently of whether PRR moved*. B.1's equivalent check is what made its null
  interpretable rather than merely disappointing.
- **Report `frac_bound` per arm** — the one-sidedness check above.
- Report ID beside OOD for any provisional cells.

## STOPPING RULE, fixed now

**If B.3 nulls, the honest position is that the attention probe cannot be improved by these means.
That is a finding to write up, not a reason to try a fourth variant.**

It also changes what Phase 3's proposed system can claim. The back-off would then be to an
**unimproved** attention probe, which is exactly the weaker story flagged in review at the 31 July
meeting: *"it's not like, oh, we just fall back to some attention probe most of the time because that's
just the best thing out there."* Better to reach that conclusion deliberately than by attrition — and
worth telling him early rather than at the next meeting.
