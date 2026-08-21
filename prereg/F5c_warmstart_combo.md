# PRE-REGISTRATION — F5c: make the anchor testable everywhere, and try shrink + anchor together

> **Status (2026-08-10):** run complete 8/8. The testability bar failed (same 3/8 evals bite as F5b) and the combo is a null against the incumbent — F5 closed as mechanism-only. *(Wording edited 2026-08-10, cosmetic only — registered claims, thresholds and bars are unchanged; git history is the proof.)*

**Written 2026-08-09, after F5b's results were seen and BEFORE F5c was implemented or run.** This is
therefore a *sequel* registration: it inherits F5b's findings as priors and registers only what is
new. Population and controls unchanged from `F5_anchor_at_msp_min.md`.

## 1. What F5b established, which F5c takes as given

- The anchor mechanism is real and directional (anchor − random = +0.129 on pubmed, −0.139 on cnn).
- The log penalty only **bit on 3 of 8 evals** (`p[k]` ≥ 0.20); the other five are UNTESTED.
- Anchoring **repairs wMSP's catastrophic cells** (pubmed/SameTask −0.05 → +0.32) but the
  anchored-only arm trails `shrink@2` on the mean.

## 2. The two changes

**(a) WARM-START, so every eval is actually tested.** `--pretrain-epochs N` (registered N = 2): before
the normal 5 epochs, train the weighter on the **penalty alone**, pushing `p[k]` up before the rank
loss enters. Deterministic, same seeds. Control A (the copied loop ≡ library) is computed with
pretrain **off**, so its exactness is untouched.

**(b) THE COMBO ARM — the answer to "can this be a mean-PRR method".** `shrink@2` wins the mean
because its uniform anchor (= `perplexity`) is good on 5–6 of the 8 datasets; `msp_min` anchoring
wins where the max is good and repairs collapse. The combo keeps both pressures at once:

```
loss = rank_loss + 2.0 · shrink_to_uniform(w) + λ_a · (−log p[k])
```

with the uniform coefficient **fixed at 2.0** (the incumbent's value, not tuned) and only λ_a swept.
This is one trained model with two fixed regularisers — **not** a gate, not a per-dataset selection,
not an ensemble.

## 3. Registered bars

- **PRIMARY (combo):** LODO-selected λ_a, vs `shrink@2` (+0.2287) on the OOD mean, n = 8, the
  standard three-part bar (margin > +0.010, signs ≥ 6/8, Wilcoxon p < 0.05).
- **SECONDARY (robustness, the F5 story):** does the combo retain the catastrophic-cell repair —
  pubmed/SameTask must stay above +0.15 (vs shrink@2's −0.05) — while conceding < 0.05 on
  cnn's OOD mean vs shrink@2?
- **TESTABILITY:** all 8 evals must reach `p[k]` ≥ 0.20 somewhere on the λ_a grid under warm-start;
  any that do not are again named UNTESTED.
- Controls A–D and the random-anchor arm unchanged.

## 4. Registered expectations (so the outcome is a result either way)

Expected: warm-start makes all 8 testable; the combo **retains most of shrink@2's mean** and
**repairs the collapse cells**, at some cost on cnn. Failure reading: if the combo's mean drops
materially below `shrink@2`, the two pressures interfere and "robust wMSP" survives only as the
anchored-only robustness claim on the concentrated datasets. If λ_a → 0 is what LODO picks, the
anchor adds nothing once shrink is present, and F5 closes as mechanism-only.

Driver: `scripts/checks/anchor_msp_min.py` (extended, still standalone). Outputs `__logws` files.

## 5. AMENDMENT (2026-08-09, on review; the first F5c launch was KILLED before any
## output existed, so this precedes all F5c results)

**(a) PRIMARY READING REORDERED.** The primary question is **not** the combo's mean-PRR bar (that is
now secondary). PRIMARY: **does the rescue property replicate on the five newly-testable datasets?**
— i.e. repair proportional to brokenness (`msp_min − wMSP-norm`), anchor-specific vs the random
control, **at the LODO-selected λ_a, never at the grid top**. The existing rescue correlation
(+0.671) drops to +0.262 without pubmed; replication on new datasets is what turns it from an
anecdote into a claim.

**(b) THE HEADLINE CELL RESTATED AT THE HONEST λ.** pubmed/SameTask under the LODO-selected λ = 3 is
**+0.239** (anchor − random +0.166 there), against shrink@2's −0.052. The +0.32 figure is the λ=10
grid top and is only ever quoted as an oracle. The repair survives honest selection: **−0.05 → +0.24.**

**(c) ATTRIBUTION CONTROL ADDED (`wsonly`).** The λ_a=0 baseline carries no warm-start (the join
check to `pdl_master` is intact), which means λ_a>0 arms differ from it by warm-start AND sustained
penalty jointly. New arm at λ ∈ {1, 10}: warm-start toward the anchor, then train with NO sustained
penalty. wsonly ≈ anchor arm ⇒ the effect is initialisation; wsonly ≈ λ=0 ⇒ the sustained pressure
is what matters. Reported per cell.
