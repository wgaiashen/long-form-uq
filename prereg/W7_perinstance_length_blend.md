# PRE-REGISTRATION — W7: the per-instance length blend (Joe's §6 ask, run as asked)

**Written 2026-08-09, before the script was implemented.** Joe, 7 Aug §6: apply the HBO pattern
**per test example**, with length in the slot where OOD-ness sat. §8's oracle retractions are at
dataset/cell granularity and do NOT close this; it has never been run in this form on the canonical
population. (The seed-1 `length-blend` in the master is the {floor, armA} version on the router
population with a comparator later shown broken — `STOCKTAKE_post31July.md:1398-1435`.)

## The estimator (ONE parameter)
Per test example i, z-score both uncertainty vectors within the cell (label-free), then
`u_i = w_i·z(msp_min)_i + (1−w_i)·z(SAPLMA)_i` with `w_i = exp(−len_i / L)` — short answers lean on
`msp_min`, long on the probe, exactly Joe's direction. **L is the only parameter**, LODO-selected
over the 8 evals from grid L ∈ {8, 16, 32, 64, 128, 256, 512}. Not a gate, not a selector: one fixed
monotone blend.

## Data and joins
`results/pdl_perex/` (40 canonical cells, 3 seeds, per-example `unc__floor_min`, `unc__saplma`, `y`).
Lengths recomputed from records; **join gate: recomputed `msp_min` must equal `unc__floor_min`
bit-for-bit per cell, else the cell is skipped LOUDLY.**

## Bars and prior
Three-part bar vs **SAPLMA** (the stronger component; beating the weaker one is not a claim), OOD
mean, n = 8: margin > +0.010, signs ≥ 6/8, Wilcoxon p < 0.05. Also reported vs `msp_min` and vs the
best single component per dataset.
**Registered expectation: NULL.** Two adjacent per-instance length results are negative (§3.8
oracle-proof; §4 sign-unstable). This is run because it is the supervisor's direct request and the
exact estimator is untested — a clean pre-registered null on it is the deliverable if it fails.
