# Pre-registration — per-response adaptive Lehmer aggregation

**Date:** 2026-08-10. **Population:** `meta-llama/Meta-Llama-3.1-8B`, the canonical ProbeDriftLong
grid (8 long evals × 5 rungs, seeds 1/2/3, layer 15, carve `legacy`, 1800-row supervised pools).
**Committed BEFORE any adaptive-Lehmer test PRR existed.** The driver's smoke mode prints no PRR
by design and none had been produced at commit time.

## 0. Honesty preamble — what this is and is not

- Every Llama baseline, the complete Llama Lehmer curve, the W1/W4/W5 sharpening results, the
  complete Qwen master, the Qwen Lehmer curve, and the Qwen replication verdict were **all
  inspected before this method was designed**. The method is motivated by those results.
- This is therefore a **prospective test of a new method on a reused development population**,
  not independent confirmation. A later Qwen run is a **cross-model stress test of a frozen
  method** (Qwen is not untouched: W6 and the 2026-08-10 verdict already exposed its floors,
  Lehmer behaviour, and the ExpertQA length anomaly, which informed the motivation).
- A genuinely confirmatory result would need a third model and/or new held-out datasets.
- The aggregation-regime audit (Decision Gate A, `results/analysis/AGGREGATION_REGIME_AUDIT.md`)
  was completed before this file was committed. Its inputs are exploratory mechanism evidence on
  already-read labels; nothing in it selected any parameter of the method below — the
  architecture, features, bound, and training recipe were fixed in the plan of 2026-08-10
  (the project's planning notes) before the audit ran.

## 1. Method (implementation: `src/luq/adaptive_lehmer.py`, driver
`scripts/checks/adaptive_lehmer.py`, unit tests `tests/test_adaptive_lehmer.py` — all committed)

Score: the Lehmer mean of the response's raw token NLLs (all generated tokens, no mask — the
canonical floor token policy), `U_i(β_i) = Σ l^(β_i+1) / Σ l^β_i`, with `β_i = 16·sigmoid(g_i)`.
β_max = 16 is the top of the canonical finite Lehmer grid; it is fixed and will not be tuned.
Hidden states and NLL-shape features may change **only β**; the score is always a deterministic
function of the token NLLs. Four gates share the identical scorer:

| method name | g_i | role |
|---|---|---|
| `lehmer_nllshape_beta` | linear(10 predeclared NLL-shape features, train-standardised) | **PRIMARY** |
| `lehmer_hybrid_beta` | g_h(mean-pooled L15 state) + g_n(shape) + b, single linear each | secondary: do hidden states add anything |
| `lehmer_hs_beta` | linear(mean-pooled L15 state) | ablation |
| `lehmer_global_beta` | one learned scalar per training cell | control: honest pool-level β selection |

**Why NLL-SHAPE is primary (decided 2026-08-10, before any PRR, per the author's review):** the
hypothesis is that the response's own probability geometry says whether its uncertainty evidence
is diffuse or concentrated. A hidden-state gate risks smuggling correctness-probe signal into
what is claimed as a regime-selection method; the A6 audit shows hidden-state token weights carry
information different from NLL ranking, which makes that contamination risk concrete rather than
hypothetical. HYBRID therefore tests "do hidden states add anything" as a predeclared secondary,
and cannot replace NLL-SHAPE as the headline if it happens to score better.

NLL-shape features (order fixed): log_T, mean_nll, std_nll, max_nll, max_z, top1/top5/top10pct
mass shares, nll_entropy_norm, max_minus_second. Standardised on training-pool stats only.

Training: the canonical weighted-MSP recipe verbatim (AdamW lr 1e-3, 5 epochs, batch 32, the
pairwise sigmoid soft-rank MSE against 1−y, batches <2 skipped, torch.manual_seed(seed)). Only
the gate trains. No validation sweep; no hyperparameter is new.

## 2. Primary estimand and evidence package

For each dataset e: `OOD_mean(m, e)` = mean PRR over the 4 OOD rungs (3-seed means per cell).
`delta_e = OOD_mean(lehmer_nllshape_beta, e) − OOD_mean(msp_min, e)`, n = 8 datasets.
Primary estimand: `macro_delta = mean_e(delta_e)`.

**There is no binary success gate.** The complete evidence package, always reported together:
macro mean delta; median dataset delta; all 8 dataset deltas; positive signs /8; two-sided exact
paired Wilcoxon p; dataset-level bootstrap CI for the macro mean; all 8 leave-one-dataset-out
macro deltas. `msp_min` is the reference because it is the pre-registered floor that motivated
adaptive concentration; every table ALSO reports the same package against `perplexity` (Qwen
showed msp_min is not universally the stronger endpoint), and an adaptive method that beats
msp_min while trailing perplexity on the same population is not presented as a success.

**Report-promotion rule (fixed now):** the method becomes a *headline* method only if the effect
is (a) meaningfully positive across datasets — not a rounding-level macro mean; (b) not driven by
a single dataset (LODO stays positive under every omission); and (c) at least directionally
supported by the predeclared secondary axes. Otherwise it is reported as mechanism evidence or an
exploratory negative. No post-hoc favourable story.

## 3. Predeclared secondary axes (fixed before any PRR; not rescue subsets)

1. Hard OOD: `DiffTask-long` and `1ds-Diff-long`, separately.
2. Summarisation subgroup: xsum, cnn_dailymail, samsum (the regime the audit shows is
   cross-model stable).
3. ID→OOD retention per rung.
4. Endpoint robustness: vs `msp_min` AND vs `perplexity`.
5. Learned-method comparison (descriptive): SAPLMA, attention pooler, wMSP-norm, the regularised
   wMSP incumbent, fixed Lehmer β=1.
6. Adaptation itself: NLL-SHAPE (and HYBRID) vs GLOBAL-BETA — per-response adaptation vs honest
   pool-level selection. Mean delta, signs/8, Wilcoxon p.
7. Signal ablation: HYBRID vs HS vs NLL-SHAPE vs GLOBAL. No ablation is promoted to headline.

## 4. Mandatory controls (B6; all already implemented and unit-tested)

- Fixed-β identity vs `sharpening_family.score_lehmer` at β ∈ {0, 0.5, 1, 2, 4, 8, 16} — PASSING.
- Endpoint sanity: β=0 ≡ perplexity ranking; high β converges toward msp_min — PASSING.
- Constant-gate control: zero weights + β=1 bias reproduces fixed Lehmer β=1 — PASSING.
- Shuffled-hidden control (`*_shuffled_h`): training hidden states permuted within source
  dataset, one fixed permutation per run, seed = 100000·train_seed + crc32(dataset) mod 9973.
  If HS/HYBRID do not beat their shuffled controls, hidden states are not credited.
- NLL-shape permutation control: recorded as OPTIONAL before launch (compute priority goes to
  the hidden shuffle); if run, same within-source design.

## 5. Interpretation discipline

Read the outcome against the eight patterns of Decision Gate B in the plan (broad win / hard-OOD
win / summarisation win / GLOBAL explains it / signal ablations / shuffled-control parity /
null / negative), choosing the **narrowest supported interpretation**. Pattern 8 (null/negative)
closes the line: no sweeps of bounds, layers, widths, features, or losses on this population.
Hard freeze: this experiment gets ONE run; if it has not launched by 2026-08-14 it is dropped.

## 6. Outputs

`results/adaptive_lehmer__meta-llama_Meta-Llama-3.1-8B.csv` (+ `__diag.csv` with the B7 beta
diagnostics). Nothing overwrites `pdl_master`, the sharpening CSVs, or any Qwen file. Provenance
per row: commit, carve, layer, beta_max, train_sources, n_train/n_test, seed.
