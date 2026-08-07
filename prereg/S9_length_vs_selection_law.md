# Pre-registration — S9: is the "selection law" really about ANSWER LENGTH?

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

⚠️ `mean_len` in `results/id_entropy_vs_drop__*.csv` is **T = G + 1**: the generated tokens **plus one
prompt token** (the pooling window is `[last_prompt_token] + gen_tokens`, `entropy_delta_vs_drop.py:89-97`).
It therefore **includes a prompt token** and is not "generation only".

**Resolution:** compute `G = len(record["gen_token_ids"])` directly from the Tier-1 records for **all 10**
datasets. Verified present for all of them. This is generation-only, identical across every dataset, and
avoids mixing two definitions. The reused column is reported alongside as a **cross-check**: mean(T) − 1
must equal mean(G) on the shared rows, since T = G+1 for every kept row.

## Statistical discipline for n = 8 / n = 10

- Report Pearson **and** Spearman, with n and p, for every pair.
- Report bootstrap CIs.
- ⚠️ **A partial correlation on n = 8 is NOT reported as significant.** It is reported as a point estimate
  with its CI and read as a direction only. With n = 8 and one control variable there are 5 residual
  degrees of freedom, and nothing at that size settles a confound.
- Flag any dataset with **n_examples < 200**.

## Population caveat that must be carried

The 8 long-form datasets take `msp_min_OOD` as the mean over their **4 long OOD rungs**. `sciq` and
`trivia_qa` have no such rungs — their value is the single **`Long->Short`** cell. These are different
rung constructions, so the 10-dataset correlations mix two definitions of "OOD". **The 8-dataset
correlations are the clean ones; the 10-dataset versions are reported alongside and flagged.**
