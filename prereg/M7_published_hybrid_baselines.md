# M7 — Published probability, distance and hybrid baselines on the corrected-span populations

> **Recorded outcome:** run. The comparator grid completed on the corrected-span populations, and every published distance and hybrid method evaluated here lands below the simplest probability aggregate.

Written before any hybrid prediction-rejection ratio has been computed, inspected or interpreted.

## 0. What this document does and does not register

This registration is **not** an original pre-registration of the learned-weighting hypothesis. That
work is finished and its results are known: the corrected-span eight-dataset masters for all three
model populations already exist, and the direction of the comparison against simple probability
aggregates has already been read off them. Registering a hypothesis whose answer is already visible
would be worthless, and claiming otherwise would be worse than not registering at all.

What is registered here, prospectively, is everything that is genuinely not yet known:

1. the exact implementation of each published baseline, fixed against the reference source before any
   of them has produced a number on this project's data;
2. which comparisons will be reported, and the unit of analysis and test for each;
3. the criteria by which a reproduction is judged faithful, adapted, or failed;
4. the deviations from the published methods that this project's caches force, and their consequences.

The purpose is that no implementation choice can later be made, or revised, in the direction that
improves the comparison.

## 1. Sources

All baselines are ported from the released implementation accompanying the Hidden Failures work, not
from the paper text:

- `satmd_baseline/token_mahalanobis_distance.py`
- `satmd_baseline/relative_token_mahalanobis_distance.py`
- `satmd_baseline/average_token_mahalanobis_distance.py`
- `satmd_baseline/average_token_mahalanobis_distance_hybrid.py`
- `satmd_baseline/huq_msp_lrtmd.py`
- `lm_polygraph_lite/estimators/max_probability.py`
- `lm_polygraph_lite/estimators/mahalanobis_distance.py`
- `run_hbo.py` and `scripts/table6_hbo.sh`

## 2. The sequence-probability baseline is already measured

The reference defines its sequence-probability score as the negative sum of token log-probabilities.
This project's `floor_sum` is `-sum(token_logprobs)`, computed by `luq.msp.msp_uncertainty(..., "sum")`.
These are the same formula, so the baseline is present at every cell of all three corrected-span
masters and requires no new computation.

Registered consequence, fixed now: because that score is not length-normalised and is the weakest of
the three probability aggregates this project measures, it is **never reported alone**. Every table
containing it also contains mean token negative log-likelihood and minimum token probability. A
comparison against the sequence-probability score by itself would be a comparison against the most
convenient baseline, which this project does not do.

## 3. The layer deviation, and what it costs

The reference fits a Mahalanobis distance at every hidden layer and learns a ridge regression over the
resulting per-layer distances; no launch script overrides that default. This project caches per-token
hidden states at one layer per model, and storing every layer for eight datasets would require roughly
500 GB per model, which is not available.

Registered decision: the distance family runs at the middle layer only. Its report-facing names carry
a middle-layer qualifier, and it is described as a middle-layer adaptation, never as a reproduction.

Two consequences, recorded now so that neither is presented later as a finding:

- With one layer the ridge regression has a single input feature under a positivity constraint, so its
  output is a monotone increasing function of the mean distance and its ranking is identical to that
  distance, unless the coefficient is clipped to zero, in which case the score is constant. The fitted
  coefficient is recorded per cell so this is visible.
- The hybrid back-off is **not** affected. Its own launch recipe selects a single layer index, the
  middle one, out of the saved stack. At the middle layer it is exact, and it is reported as a
  reproduction.

## 4. Fixed implementation constants

Taken from the reference and not tuned:

| constant | value | source |
|---|---|---|
| token filter threshold | 0.3 | `scripts/table1_satmd.sh` |
| filter applies only when surviving tokens exceed | 10 tokens | `token_mahalanobis_distance.py` |
| covariance | unbiased, over the token axis | `compute_inv_covariance` |
| jitter ladder | 1e-15 to 1e-1, first value giving non-negative eigenvalues | same |
| inverse | taken in float64, cast to float32 | same |
| distance | square root of the quadratic form | `mahalanobis_distance_with_known_centroids_sigma_inv` |
| aggregation | mean over the response's tokens | `aggregation="mean"` |
| meta-regressor | `Ridge(positive=True)`, target `1 - correctness` | `average_token_mahalanobis_distance.py` |
| dev split | 50 percent, `random_state=42` | same |
| hybrid grid | t_min 0.0-0.3, t_max 0.8-1.0, alpha 0.0-1.0, steps 0.05/0.05/0.1 | `huq_msp_lrtmd.py` |
| background corpus | C4, `en/c4-train.00000-of-01024.json.gz`, first 100,000 rows, 2,000 sampled at seed 1 | `run_polygraph.py` |

The port is checked numerically against the reference implementation by
`scripts/checks/md_port_equivalence.py`, which must pass before any result is read.

## 5. Two further deviations, registered rather than discovered later

1. **Token window.** The reference measures the generated tokens. Every per-token cache in this
   project stores the last prompt position plus the generated tokens, and the background must be
   measured on the same window as the data it is compared against, so the project window is used
   throughout.
2. **Background budget.** The reference regenerates the background at each dataset's token budget.
   Greedy decoding makes a shorter generation an exact prefix of a longer one, so the background is
   generated once at the largest budget in the grid and sliced. This is an identity, not an
   approximation, and the generating script measures it rather than assuming it.

## 6. Populations

The three corrected-span eight-dataset masters, each with its own frozen namespace resolution. No
comparison pools across model populations; a panel carries one explicitly labelled row per population.
Cells, seeds, sampled training rows and evaluation splits come from the same functions the masters were
built with, and the recomputed sequence-probability score and refitted probe are gated per cell against
the master's own columns before any new method is read.

## 7. What will be reported, and how

Unit of analysis is the **dataset**, n = 8. The four out-of-distribution rungs of a dataset are not
four independent observations. Each comparison reports the per-dataset difference, the sign count, an
exact Wilcoxon signed-rank test, a bootstrap interval and a leave-one-dataset-out range. No effect-size
threshold is invented; any margin quoted is derived from measured variability.

Registered comparisons:

- **C1** learned weighting against each of the three probability aggregates, on the two cross-task rungs.
- **C2** learned weighting against the distance family.
- **C3** the probe against the distance family and against the hybrids.
- **C4** each hybrid against its own components, which is the check that the hybrid does anything.
- **C5** the hybrid back-off against the sequence-probability score on the far rungs, where the formula
  forces them to coincide.

## 8. What counts as a successful reproduction

The prediction under test is the published finding that these methods do not transfer to long-form
generation. A weak result is therefore a successful reproduction and is reported as one. A hybrid is
declared **not reproduced** only if it fails its own internal consistency checks:

- the back-off must equal the sequence-probability score exactly on every example whose percentile
  exceeds 0.5, and its supervised weight must never exceed 0.5;
- the hybrid combination's hyperparameters must have been selected without any evaluation label;
- the distance statistics must be fitted without any evaluation row.

A failure of any of these means the implementation is wrong and is fixed. A merely disappointing
prediction-rejection ratio is a result and is **not** grounds for re-tuning anything.

## 9. The vacuity diagnostic, reported either way

For the back-off, the fraction of evaluation examples whose percentile exceeds 0.5 is reported per
rung. If that fraction is close to 1 even in the matched-distribution setting, the method is the
sequence-probability score everywhere, every comparison involving it is trivially satisfied, and it
must be described that way rather than as agreement.

## 10. Excluded in advance

No hyperparameter of the learned-weighting method is re-tuned. No ensemble weight is tuned. No hybrid
threshold is selected on evaluation labels. No corrected-span protocol is revisited. Substituting the
learned score for the sequence-probability branch of a hybrid is a separate question and may not be
attempted before the original hybrid has been reproduced and reported.
