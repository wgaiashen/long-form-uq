# Pre-registration — the missing 2x2 cell: learned attention with an MLP head

Written **before** any run of `scripts/checks/head_aggregation_2x2.py` (date: 2026-07-30, DoC).
Mirrors the driver docstring. This file is a `results/` artifact, not a living doc.

## The design

|                    | linear head              | MLP head (256/128/64)   |
|--------------------|--------------------------|-------------------------|
| **mean-pool**      | `meanpool_linear` (=armB)| `meanpool_mlp`          |
| **learned attn**   | `attention_linear`(=armA)| `attention_mlp` (NEW)   |

All four cells trained in ONE paired loop, ONE recipe (60 epochs, Adam lr=1e-3, wd=1e-2 on head params,
wd_query=0, bs=32, BCEWithLogitsLoss; temperature re-selected with the head in place). Reference SAPLMA
(`saplma_ref`, 5ep, wd=0, `conf_meanpool`) computed alongside as a bridge.

## Registered predictions and contrasts (fixed before seeing OOD numbers)

- **PRIMARY contrast: `attention_mlp(60ep)` vs `meanpool_mlp(60ep)`** -- the aggregation effect *at the
  MLP head*, fully recipe-controlled. This is the headline. Direction not asserted as a target.
- **SECONDARY contrast: `meanpool_mlp(60ep)` vs `armB(60ep)`** -- the CLEAN, recipe-controlled head
  effect. The motivating "+0.038 head effect" compared armB (linear, 60ep, wd=1e-2) against SAPLMA (MLP,
  **5ep, wd=0**) -- head architecture AND training recipe both differ, so it is **not established**. If
  this clean delta is well below +0.038, part of the original margin was epochs/weight-decay and the
  headline claim must be rewritten.
- **Rough expectation, NOT a target:** the ~+0.260 figure for `attention_mlp` was derived from the
  confounded margin; treat it as a rough expectation only.
- **Informative failure case (reported as a finding, not a null):** if `attention_mlp ≤ meanpool_mlp`,
  the two axes interact -- a stronger head cannot rescue an attention distribution that dissolves under
  shift.
- **Mechanism prediction:** learned attention is claimed to dissolve toward uniform under shift (armA's
  ID→OOD attention-entropy gap ≈ +0.102). The MLP head changes the gradient path to the query, so it may
  change how much the attention dissolves -- reported via the `attn_entropy` column for both attention
  cells. Less dissolution, or equal dissolution but still a win, are both results.

## Correctness gates (must clear before any `attention_mlp` PRR is trusted)

1. `meanpool_linear` reproduces **armB** and `attention_linear` reproduces **armA**, per-cell, within
   seed noise -- gated against the raw per-cell values in `results/fixed_prior_ladder__*.csv` (NOT the
   assembler's re-derived COMMON-set averages). If these fail, STOP -- the recipe diverged.
2. `meanpool_mlp_5ep` (5ep, wd=0) reproduces `saplma_ref` -- the clean gate that the MLP-head path IS the
   SAPLMA head (proven on synthetic data by `tests/test_head_hidden.py::test_meanpool_mlp_5ep_equals_saplma`).
3. **q-moved assertion:** after training, `attention_mlp`'s ‖q‖ must be well off its zero init (S6 saw
   0→10.70). If ‖q‖ ≈ 0 (< 0.1 absolute, or < 5% of `attention_linear`'s ‖q‖), the query never trained →
   the cell is **RETRACTED** (blank PRR, `retracted=True`), never reported as a number.
4. `attention_mlp_5ep` is expected to collapse to `meanpool_mlp_5ep` (query can't train in 5 epochs) -- a
   positive diagnostic confirming why 60ep (or `--early-stop`) is required.

## Recipe-overfitting risk and the decision gate

The MLP head is ~1M params against ~1800 rows (p/n ≈ 580); SAPLMA's 5 epochs *is* the regularisation.
60 epochs at wd=1e-2 may overfit and collapse both MLP cells (which would falsely read as "the MLP head
does not help"). Therefore **pubmed_qa is run ALONE first as a decision gate**: report the armB/armA
gates, the `meanpool_mlp_5ep` vs `saplma_ref` gate, the bridge delta `meanpool_mlp(60ep) − saplma_ref(5ep)`,
and `q_final_norm`; then WAIT for go-ahead. If `meanpool_mlp(60ep)` is well below `saplma_ref(5ep)`, switch
to `--early-stop` (validation-selected epochs for the MLP-head cells; linear cells stay fixed-60ep armA/armB
so their gates still pass) and re-clear the pubmed gate before fanning out to the other seven evals.

## Update -- two implementations, and the stopping rule (after the earlier med_quad run landed, before this driver's pubmed result)

An earlier run (commit `506b470`, driver `scripts/checks/head_agg_2x2.py`) already implemented
and ran the same 2x2, med_quad complete, held fan-out for the rest (now cancelled). Its reference cells
(`meanpool_linear`/`meanpool_mlp`/`attention_linear`) reproduce the base run to ~1e-4, so it is a valid
cross-check on those. **But its `attention_mlp` is a DIFFERENT method from this driver's** -- the two do
not test the same thing, so its med_quad null does not transfer directly:

| cell | this driver (`head_aggregation_2x2.py`, Option 1) | parallel (`head_agg_2x2.py`, 506b470) |
|---|---|---|
| meanpool_linear | linear head, 60ep, wd=1e-2 (=armB) | same (=armB) |
| meanpool_mlp | SAPLMA MLP, **60ep, wd=1e-2**, joint | SAPLMA MLP, **5ep, wd=0** (== `saplma`) |
| attention_linear | linear head, 60ep, wd=1e-2 (=armA) | same (=armA) |
| attention_mlp | query + MLP head **JOINTLY** trained (60ep, wd=1e-2 head, T re-selected WITH the MLP head) | **armA's** query (60ep, linear-head-trained) FROZEN, pooled, then SAPLMA MLP **5ep/wd=0** fit post-hoc on the frozen pooled vectors -- a **decoupled head swap**, query never sees the MLP gradient |

Consequence: the parallel `attnmlp_vs_armA` null (ns on all 5 med_quad rungs) is specifically about the
**decoupled** head swap -- reading armA's fixed pooled vector with a 5ep MLP instead of the jointly-trained
linear head. This driver's `attention_mlp` lets the MLP's gradient reshape the query (the mechanism the
`attn_entropy` column probes), so it *could* diverge from armA where the decoupled swap does not. The two
are complementary, not redundant.

**Stopping rule (updated): the gate is pubmed (this driver) AND the existing med_quad (parallel).** After
pubmed lands, report (a) armB/armA reproduction; (b) the CLEAN recipe-controlled head effect
`meanpool_mlp@60ep vs armB@60ep` -- near +0.038 or near zero? (the parallel run never computes this -- its
meanpool_mlp is 5ep=saplma); (c) `attention_mlp vs armA` -- does it agree with med_quad's ns *given* it is
the joint variant?; (d) `q_final_norm` for attention_mlp; (e) the bridge delta. **IF (b) ≈ 0 AND (c) ns,
STOP** -- do not fan out; the negative on two evals is the result ("the apparent head advantage was a
training-recipe artefact, not an architectural one"). IF (b) is materially positive OR (c) shows a real
gain, fan out as planned.
