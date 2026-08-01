# PRE-REGISTRATION — C2: token-probabilities as a pooler INPUT (Lihu idea) — written 2026-07-30 BEFORE running

## What C2 is (and is NOT)
C2 feeds the model's own per-token NLL (= −logprob) to the attention pooler as an EXTRA INPUT FEATURE
(concatenated onto the 4096-dim hidden state → a 4097-dim input). The learned query can then weight tokens
partly by the model's confidence, and the linear head can read the pooled confidence directly. This FUSES the
supervised pooler with the unsupervised MSP signal.
- **C2 ≠ arm-D-nll.** Arm D uses the NLL to *tilt the attention scores* (score = X·q + β·log(nll_prior)); the
  NLL never enters the head. C2 puts the NLL *inside the representation* the query AND head read. Different
  mechanism, so a null on arm-D does not pre-determine C2.
- **C2 ≠ score-side weighted-MSP.** wMSP weights the NLL and the sum IS the score (no probe). C2 keeps the
  supervised probe and gives it the NLL as one more feature.

## Population + isolation
FULL canonical grid: 8 long evals × 5 `cells_long` rungs × 3 seeds, widened pool. Paired against **armA** (the
incumbent learned pooler, SAME seeds / splits / val-temperature — armA's selected T is reused for C2, so the ONLY
difference is the NLL channel) and against the free **msp_min floor**. armA reproduction gate (vs §C.3) must pass.

## Grounding gates (must pass before any C2 number is trusted)
1. **Alignment:** the NLL window length == the state window length (G+1) per example, else ABORT (no silent pad).
2. **Label-free:** the channel is built from `record["token_logprobs"]` only — asserted (no y in its signature).
3. **Constant-channel wiring gate (in-job, on pubmed_qa ID seed 1):** a CONSTANT (information-free) NLL channel
   must reproduce armA's PRR within 0.05. A constant channel is a uniform score-shift (softmax shift-invariant;
   PRR rank-invariant), so if it moves the result the channel is miswired/mis-scaled → HALT.

## PREDICTION (registered)
- **Expected: NULL vs armA OOD** — the whole aggregation/weighting/gating landscape has come back null OOD, and
  the NLL is already available to the free floor (which C2 is paired against). If C2 helps anywhere it should be
  on the **summarisation family** (where the probe already beats the floor and the NLL is a genuine second view),
  NOT on QA/factuality (where the floor already dominates via msp_min — adding NLL to the probe is redundant with
  what the floor already extracts). A uniform gain everywhere would be SUSPICIOUS (extra-capacity, not the signal).
- **Decisive quantity:** paired (C2 − armA) per cell, bootstrapped over cells, reported per family. Report the
  constant-channel gate margin FIRST. If C2 − armA is null and the constant gate passes, C2 is a clean negative.
