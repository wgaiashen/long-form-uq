# PRE-REGISTRATION — W7b: per-instance length blend of msp_min ↔ wMSP-shrink@1.5

**Written 2026-08-09, before implementation.** Corrects W7's pairing: W7 registered and ran
msp_min↔SAPLMA (a cross-family blend; registered null, kept as the secondary arm). The
supervisor-requested pair (§2b/§6) is msp_min ↔ **weighted MSP** — the same functional family
`q = Σ w_t·nll_t` with different weights, which is the entire reason a continuous axis between them
was hypothesised. The wMSP arm is **shrink@1.5**, the honest-procedure incumbent (§18); its
per-example scores are cached nowhere, so they are recomputed (1 training per cell-seed — the
"no retraining" premise was false).

**Estimator:** `u_i = w_i·z(msp_min)_i + (1−w_i)·z(wMSP@1.5)_i`, `w_i = exp(−len_i/L)`, ONE
parameter, L ∈ {8,16,32,64,128,256,512}, LODO-selected over the 8 evals.

**Controls, registered HERE (not in W7 — claiming they were would invent history):**
(a) SHUFFLED-LENGTH: permute `len_i` within each cell — if LODO does not beat this, length carries
nothing. (b) CONSTANT BLEND w = 0.5 — if this matches or beats the length rule, any gain is
two-score decorrelation, not a length effect; reported as such and never promoted (a constant
z-average is the closed ensemble line).

**Bar:** three-part vs the stronger endpoint on the OOD mean (n = 8). **Registered expectation:
NULL** (three per-instance-length negatives stand: §3.8, §4, W7). Join gate: the wMSP@1.5 per-cell
PRRs must match §15.2b to seed noise.
