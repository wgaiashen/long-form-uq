# Pre-registration — is the selection law really about response length?

**Written 2026-08-07, BEFORE the driver was run.** Committed with its thresholds and its direction.

## The claim under test

We report a **selection law**: across datasets, how much a trained probe beats the untrained `msp_min`
baseline is strongly *negatively* correlated with how strong `msp_min` already is.

- PDL: r = −0.846, p = 0.008, n = 8
- XL: r = −0.895, p = 0.0005, n = 10

Short-form QA (`sciq`, `trivia_qa`) sits at the extreme high-baseline end **and** has by far the shortest
answers. Those two facts are confounded across datasets.

## The mechanical worry, stated as a mechanism and not as a hunch

`msp_min` is a **minimum over tokens**. More tokens means more draws, and the minimum of more draws is
lower in expectation *regardless of whether the model is any less certain*. So `msp_min` may fall with
length for a purely mechanical reason, which would make the law partly a length artefact.

**`perplexity` is length-normalised and should NOT share that dependence.** That contrast is the control:
if length drags `msp_min` but leaves `perplexity` alone, the mechanical effect is real.

## Predictions, fixed in advance

1. **Direction: MONOTONE, not the entropy-style `|x − median|` U-shape.** The mechanism is order
   statistics — every extra token is another chance at a low minimum — and that is monotone by
   construction. The U-shape in the entropy work came from a different argument (both over-sharp and
   near-uniform poolers being fragile) and there is no reason to expect it here. **If the relationship
   turns out U-shaped rather than monotone, my stated mechanism is wrong even if something correlates.**
2. **`length` vs `msp_min_OOD`: negative.** Longer answers → weaker `msp_min`.
3. **`length` vs `perplexity_OOD`: markedly weaker than for `msp_min`** — this is the control, and it is
   the comparison that carries the argument, not either correlation alone.
4. **`length` vs `probe_advantage`: positive**, since advantage is (SAPLMA − `msp_min`).

## Decision rule, fixed in advance

Length **"explains" the law** only if BOTH:

- **(a)** |ρ(length, probe_advantage)| ≥ **0.70**, AND
- **(b)** the partial correlation of baseline-strength with probe_advantage, **controlling for length**,
  drops below **0.40**.

If (a) holds but (b) does not, length and baseline strength are both live and neither is dismissed.
If neither holds, the law survives the length challenge.

**Report whatever happens, including disconfirmation.**

## The within-dataset check is the decisive one

Between-dataset correlations on n = 8 or 10 cannot separate two things that co-vary across datasets. The
**within-dataset** tercile split can: inside a single dataset, the model and the task are held fixed and
only length varies.

**If `msp_min` falls across length terciles while `perplexity` does not, the mechanical effect is real and
goes in the limitations regardless of what the between-dataset correlations say.**

## Length definition — one definition for all 10, and it is NOT the one already on file

`mean_len` in `results/id_entropy_vs_drop__*.csv` is **T = G + 1**: the generated tokens **plus one
prompt token** (the pooling window is `[last_prompt_token] + gen_tokens`, `entropy_delta_vs_drop.py:89-97`).
It therefore **includes a prompt token** and is not "generation only".

**Resolution:** compute `G = len(record["gen_token_ids"])` directly from the Tier-1 records for **all 10**
datasets. Verified present for all of them. This is generation-only, identical across every dataset, and
avoids mixing two definitions. The reused column is reported alongside as a **cross-check**: mean(T) − 1
must equal mean(G) on the shared rows, since T = G+1 for every kept row.

## Statistical discipline for n = 8 / n = 10

- Report Pearson **and** Spearman, with n and p, for every pair.
- Report bootstrap CIs.
- **A partial correlation on n = 8 is NOT reported as significant.** It is reported as a point estimate
  with its CI and read as a direction only. With n = 8 and one control variable there are 5 residual
  degrees of freedom, and nothing at that size settles a confound.
- Flag any dataset with **n_examples < 200**.

## Population caveat that must be carried

The 8 long-form datasets take `msp_min_OOD` as the mean over their **4 long OOD rungs**. `sciq` and
`trivia_qa` have no such rungs — their value is the single **`Long->Short`** cell. These are different
rung constructions, so the 10-dataset correlations mix two definitions of "OOD". **The 8-dataset
correlations are the clean ones; the 10-dataset versions are reported alongside and flagged.**

---

# AMENDMENT — 2026-08-14

**Nothing above this line has been edited.** This section is appended, dated, and records two things:
that the *premise* of the original pre-registration is a statistic I have since retracted, and what
happened when the driver was finally run.

## A1. The premise stated above is mathematically coupled, and is retracted as evidence

The section "The claim under test" asserts a selection law at **PDL r = −0.846, p = 0.008, n = 8** and
**XL r = −0.895, p = 0.0005, n = 10**. Both numbers correlate `msp_min` against
`probe_advantage = SAPLMA − msp_min`. **That puts `msp_min` on both axes**, so the correlation is
driven negative by measurement noise alone. It is arithmetic, not a finding.

I established this on 2026-08-07, the same day this file was written, by simulating the null: if SAPLMA
were **completely independent** of `msp_min` with the same spread, the correlation would average about
**−0.89**. My reported −0.846 was therefore slightly *weaker* than pure noise.

The properly posed replacement is to regress SAPLMA on `msp_min` and ask whether the slope is below 1:
**b = +0.242, one-sided p = 0.004**. Even that has to be stated carefully, because the slope is also not
distinguishable from zero at n = 8 (two-sided p = 0.262, r² = 0.20). **The honest claim is "the probe
does not track the baseline", not "advantage compresses at rate 0.24", and certainly not a law.**

Full working, both models, with the mandated controls:
`results/analysis/REPORT_VERIFICATION_PACK_2026-08-13.md` §2.

## A2. What that does and does not do to this pre-registration

**It does not invalidate the experiment.** The length mechanism set out under "The mechanical worry" is
independent of the coupling. `msp_min` is a minimum over tokens whether or not anyone regresses it
against a quantity containing itself, so "does answer length drag `msp_min`?" remains a real and
separate question. **This is an amendment, not a retraction.**

What changes is only the framing: the decision rule below can no longer be read as deciding whether *the
law* survives, because there is no law to survive. It decides the narrower question it actually tests,
which is **whether answer length is a confound in the relationship between baseline strength and probe
advantage**. The rule, the thresholds and the four predictions all stand exactly as written above. None
has been altered.

## A3. The driver was run for the first time on 2026-08-14

The driver (`scripts/checks/length_vs_selection_law.py`, committed `2984a8a`) was written on 2026-08-08
and **never run**. No result and no job log existed until 2026-08-14, when it was run read-only over the
cached records and tables at zero cost. It prints to stdout and writes no CSV, which is why nothing was
on disk. Output preserved at
`results/analysis/length_vs_selection_law_stdout__meta-llama_Meta-Llama-3.1-8B.txt`.

**Result on the clean 8 long-form sets (n = 8), Llama-3.1-8B:**

| pair | Pearson | Spearman | perm p |
|---|---|---|---|
| length vs `msp_min` OOD | +0.2104 | +0.2381 | 0.582 |
| length vs `perplexity` OOD (control) | +0.1686 | +0.3333 | 0.428 |
| length vs probe advantage | −0.2912 | −0.1429 | 0.752 |
| `msp_min` OOD vs probe advantage | −0.8455 | −0.8095 | 0.022 |

Partial correlations: `msp_min` vs advantage controlling for length **−0.8386**; length vs advantage
controlling for `msp_min` **−0.2171**.

**Decision rule (a) |ρ(length, advantage)| ≥ 0.70 → 0.1429 FAIL. (b) partial < 0.40 → 0.8386 FAIL.
VERDICT: length does NOT explain the relationship.** The 10-dataset variant agrees (0.3818 FAIL,
0.8689 FAIL).

**Note what the last row of that table is.** `msp_min` vs probe advantage at −0.8455 **is the coupled
statistic from A1** (it is the same quantity as the coupled r² of 0.7150 reported in the verification
pack, and √0.7150 = 0.8456). The partial correlation at −0.8386 is coupled for the same reason:
controlling for *length* does nothing about `msp_min` appearing on both axes. **So this experiment rules
out one confound while leaving the fatal one untouched, and it must never be cited as evidence that the
law holds.**

## A4. All three directional predictions were DISCONFIRMED

This is the part that matters most, and it is not visible from the verdict line.

| # | predicted | observed | outcome |
|---|---|---|---|
| 2 | `length` vs `msp_min_OOD` **negative** | **+0.2104** | wrong sign |
| 3 | `length` vs `perplexity_OOD` **markedly weaker** than for `msp_min` | **+0.1686** vs +0.2104, essentially the same | control does not separate |
| 4 | `length` vs `probe_advantage` **positive** | **−0.2912** | wrong sign |

Prediction 3 is the one I said "carries the argument, not either correlation alone". It fails. The
length-normalised control tracks `msp_min` almost exactly, which is the opposite of what the mechanism
requires.

**So the decision rule returns "length does not explain it" for the least interesting possible reason:
the mechanical effect is not detectable between datasets at all.** I registered above that "if the
relationship turns out U-shaped rather than monotone, my stated mechanism is wrong even if something
correlates". The outcome is weaker still. Nothing correlates. **My stated mechanism is not supported.**

## A5. The within-dataset check does not establish the mechanical effect either

I registered the tercile split as "the decisive one", with the reading: *if `msp_min` falls across
length terciles while `perplexity` does not, the mechanical effect is real.* The antecedent does not
hold.

| dataset | tercile mean length | `msp_min` | `perplexity` |
|---|---|---|---|
| expertqa | 96 / 200 / 346 | −2.726 / −2.864 / −3.089 | +1.035 / +0.963 / +0.959 |
| cnn_dailymail | 25 / 40 / 121 | −1.574 / −1.471 / −1.984 | +0.412 / +0.258 / +0.225 |
| pubmed_qa | 9 / 23 / 64 | −1.599 / −1.490 / −1.716 | +0.734 / +0.270 / +0.219 |

`msp_min` does fall from the shortest to the longest tercile on all three, but **non-monotonically** (the
middle tercile sits *above* the shortest on cnn_dailymail and pubmed_qa), and **`perplexity` falls too**,
proportionally more on cnn_dailymail and pubmed_qa than `msp_min` does. Both statistics move with length,
so the contrast that was supposed to isolate the order-statistics mechanism does not isolate anything.
The monotone direction registered in prediction 1 is also not met.

## A6. What may and may not be claimed from S9

| supported | not supported |
|---|---|
| **Answer length is not the confound** in the baseline-strength / probe-advantage relationship, on both the clean 8-set and the mixed 10-set populations, by the rule fixed in advance. | That the selection law survives. It does not, for the unrelated reason in A1. |
| The length definition question is settled: `G = len(gen_token_ids)` computed identically for all 10 datasets, with the `T = G + 1` column reproduced as a cross-check. | That `msp_min` drifts with length for the registered order-statistics reason. Predictions 2, 3 and 4 all failed and the tercile control does not separate. |
| The driver reproduces the master tables to 1e-6 on the four spot-checked cells, so it is reading the intended population. | Any partial correlation here as a significance claim. Registered above as point estimates only, 5 residual degrees of freedom, and that still holds. |

## A7. Deviations from the registered analysis plan

- **Bootstrap CIs were registered; the driver reports jackknife CIs** (exact leave-one-out at n = 8,
  stride-97 at n = 10). Recorded as a deviation. It does not change any verdict, since no verdict here
  rests on an interval, but the plan said bootstrap and the code does not do that.
- **The "flag any dataset with n_examples < 200" rule was applied and did not fire.** The smallest
  population is factscore at 500 records; every other dataset is 948 or more.
- **The run happened six days after the driver was committed and seven after this file was written.**
  Recorded because "registered, then run" and "registered, then forgotten, then run" are different
  claims, and this was the second one.
