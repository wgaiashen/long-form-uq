# Pre-registration -- the missing 2x2 cell (learned attention + MLP head)

Written 2026-07-30, BEFORE the experiment ran. Committed to git so the SHA/timestamp prove it
predates any result. Do not edit after the first result lands.

## The design

Two separable axes, four cells, ALL recomputed together in one run (paired: same seeds, splits,
optimiser, pool). No comparison against numbers from other runs.

|                    | linear head            | MLP head (SAPLMA's)      |
|--------------------|------------------------|--------------------------|
| mean-pool          | meanpool_linear (armB) | meanpool_mlp (SAPLMA)    |
| learned attention  | attention_linear (armA)| attention_mlp (NEW)      |

- meanpool_linear = `attn_unc(train_attn(freeze_query=True))`  (= the existing `uniform`/armB)
- meanpool_mlp    = `1 - conf_meanpool(Xmean)`                 (= the existing SAPLMA)
- attention_linear= `attn_unc(train_attn(learned query))`      (= the existing `attention`/armA)
- attention_mlp   = `1 - conf_meanpool(Xattn)` where Xattn = armA's learned-attention pooled
  vectors (`sum_t a_t x_t`, the same attention as attention_linear, re-headed with the IDENTICAL
  SAPLMA MLP). This is a head swap of armA, not a jointly retrained module, chosen precisely so
  the MLP stays byte-identical to SAPLMA (`probe.train_probe_mlp`: 256/128/64 ReLU -> 1, Adam
  lr=1e-3, 5 epochs, batch 32, no weight decay/dropout, raw features, BCEWithLogits, seeded).

## Predictions (recorded before running)

1. PRIMARY: attention_mlp about +0.260 OOD (the two axes are at least partly additive: better head
   +0.038, better aggregation +0.019, from the completed base table). ID above armA's +0.623.
   If it lands there it is the best single method on the board, above SAPLMA (+0.241 OOD), with no
   ensemble and no router.
2. THE INFORMATIVE FAILURE CASE: if attention_mlp <= SAPLMA (OOD), the two axes INTERACT: a
   stronger head does not rescue an attention distribution that dissolves under shift. That is a
   real finding and will be reported as such, NOT as a null.
3. Report ID BESIDE OOD for all four cells: the ID/OOD reversal (attention wins ID, SAPLMA wins
   OOD) is the phenomenon; the fourth cell's effect on it is the question.

## Correctness gates (must pass before any number is trusted)

The three reference cells must reproduce the existing widened-pool base values within seed noise:
meanpool_linear vs armB (`uniform`), meanpool_mlp vs SAPLMA (`saplma`), attention_linear vs armA
(`attention`). Because these three ARE the identical functions used in the base run, reproduction
should be near-exact. If any fails, STOP: the recipe differs and attention_mlp is comparable to
nothing.

## Scope

Full cells_long grid: 8 long evals x 5 rungs (ID + 4 OOD), seeds 1,2,3, widened pool, per-row
git_sha/cluster/env_hash, distinct --out per eval with overwrite guards, per-eval jobs.
