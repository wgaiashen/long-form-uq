# Pre-registration — sweep the prior-tilt strength beta

> **Recorded outcome:** run. An interior optimum on every dataset, which overturned an earlier conclusion drawn from two in-distribution cells.

**Written 2026-08-06, BEFORE any cell of this sweep has been run.** Committed with its thresholds.

## Why this exists

Arm D tilts the attention toward a prior:

```
a = softmax( (X·q)/(√d·T)  +  β · log(prior) )
```

**β has never been swept. Every prior-arm number in this project sits at the single hardcoded β = 1.0.**

That matters because β is the dial between two things we have been reporting as *separate methods*:

| β | what the arm becomes |
|---|---|
| **0** | the prior contributes nothing — pure learned attention |
| **1** | current default; `q` starts at zeros so the attention *begins* exactly at the renormalised prior |
| **large** | the tilt dominates `X·q` — the attention is driven toward the prior, i.e. toward arm C |

So arm A and arm C are the two ends of a continuum whose interior has never been looked at. The completed
S4 grid found the prior arms and learned attention tie at **+0.222** OOD. That tie is currently
uninterpretable: it is equally consistent with "the prior carries no information" and with "β = 1.0 happens
to sit at a neutral point". This sweep separates them.

The precedent is direct: the auxiliary-loss screen manufactured a null by omitting λ = 0 from its grid, and
DoC's re-run added it deliberately. **β = 0 is the same safeguard and is mandatory here** — the driver
raises if `--betas` omits it.

## Design

- **Grid:** β ∈ {0, 0.25, 0.5, 1, 2, 4}. Log-spaced above 0, spanning "no prior" to "prior dominates".
- **Prior:** `nll` (every-token surprisal) only. It was the strongest smooth recipe in S4 and is the
  cleanest test of the dial; adding recipes multiplies cost without testing β.
- **Population:** the FULL ProbeDriftLong grid, 8 long evals × 5 rungs, 3 seeds. Not a subset.
- **Reported beside PRR:** mean normalised **attention entropy** per β. A null is only interpretable once
  we know whether the knob actually bound.

## Built-in controls

1. **β = 0 must be the no-prior member of the family.** It is *not* expected to equal `armA` exactly:
   `armA` is trained at a **selected** temperature (`select_temperature`), while arm D trains at T = 1.0.
   So β = 0 is "learned attention at T = 1", and both it and `armA` appear in the table. A β = 0 that
   differs *wildly* from `armA` means something other than the prior is driving arm D.
2. **Monotone endpoints.** As β grows the attention entropy must move steadily toward the prior's own
   entropy. If entropy does not move with β, the tilt is not binding and no PRR difference can be
   attributed to it.
3. **`armC` is the β → ∞ limit** and is computed in the same cells, so the sweep should approach it.

## Decision rule, fixed in advance

- **The prior carries information** only if some β > 0 beats β = 0 by **> +0.02** (the aggregation-axis
  noise floor) on the pooled OOD rungs.
- **If β = 0 is best or within 0.02 of the best**, the prior adds nothing *at any strength*, and the S4
  tie is explained: it was never about the value of β. **That closes the prior-tilt line**, and it closes
  it far more strongly than the S4 tie could, because it rules out the "wrong β" escape.
- **If an interior β wins**, report it with the attention entropy that goes with it, and treat the winning
  β as **selected on this grid** — a value chosen post-hoc from 6 options on the test cells is an oracle,
  and a deployable number needs β chosen on training-pool validation, exactly as the temperature already is.

## Predictions, so a miss is visible

1. **I expect β = 0 to be within noise of β = 1**, i.e. the prior-tilt line to close. The S4 result is that
   selective and smooth priors alike land on the learned-attention number, and the simplest explanation is
   that the tilt is not doing much at any strength.
2. **I expect attention entropy to fall monotonically with β**, because a larger tilt concentrates mass on
   the surprising tokens. If it does not, control 2 has failed and the PRR column is not interpretable.
3. **A large β should approach `armC`, not beat it.** If β = 4 clearly beats the frozen-prior arm, the two
   are not the same limit and the arm-C implementation needs re-checking.
4. **Uniform improvement across all 8 datasets would be suspicious**, not a success — S4 showed the prior's
   value tracks the strength of the unsupervised floor, so any real effect should be dataset-dependent in
   the same way.

## What this does NOT test

β conflates two things: **how much to trust the prior** and **how peaked the prior should be** (since
`β·log p = log p^β`, and `p^β` renormalised is a temperature-scaled prior). Separating trust from
sharpness needs a second parameter and is **not** attempted here. Stated so the result is not
over-claimed.
