# Pre-registrations

Each file here states a hypothesis, the population it will be tested on, and the numeric bar it has
to clear, and each was committed **before** the run it describes. The commit timestamp is the point:
it shows that a prediction pre-dates its result. Where a registration was later amended, the
amendment is recorded in the file rather than replacing what was originally written.

Several of these are recorded negatives, and they are kept deliberately. A pre-registration that is
removed once it fails is worth nothing, and the analyses in the write-up that ask which nearby
designs fail rest on exactly these files.

Two files here are not pre-registrations. `CANONICAL_v1_MANIFEST.md` and its `.sha256` are a
checksum manifest for the frozen v1 caches, and `xl_contribution_provenance.md` reconstructs
provenance for a set of result rows that were written before provenance stamping existed.

The `Recorded outcome` column quotes the file's own status line where it has one. A dash means the
file does not carry a one-line status, not that the experiment was not run: read the file.

| Registration | What it registered | Recorded outcome |
|---|---|---|
| [`adaptive_lehmer_aggregation.md`](adaptive_lehmer_aggregation.md) | per-response adaptive Lehmer aggregation | — |
| [`anchor_at_min_token_probability.md`](anchor_at_min_token_probability.md) | regularise activation-weighted surprisal toward the minimum token probability, not toward mean token NLL | the linear penalty stalled grid-wide (an optimisation failure, §16.1); the registered −log p[k] fallback ran 8/8 (F5b): mechanism confirmed both directions with the random-anchor control,… |
| [`anchor_warm_start_combination.md`](anchor_warm_start_combination.md) | make the anchor testable everywhere, and try shrinkage and anchoring together | run complete 8/8. The testability bar failed (same 3/8 evals bite as F5b) and the combo is a null against the incumbent — F5 closed as mechanism-only. |
| [`attention_entropy_shift_as_drop_predictor.md`](attention_entropy_shift_as_drop_predictor.md) | does attention flattening predict performance drop? | — |
| [`cawsa_saplma_ensemble.md`](cawsa_saplma_ensemble.md) | does CAWSA supply a complementary signal to SAPLMA? | — |
| [`cawsa_saplma_ensemble_qwen_replication.md`](cawsa_saplma_ensemble_qwen_replication.md) | does the CAWSA + SAPLMA complementarity result replicate on Qwen2.5-14B? | — |
| [`entropy_penalty_on_spread_datasets.md`](entropy_penalty_on_spread_datasets.md) | the one-sided entropy penalty on the untested spread regime | — |
| [`head_and_aggregation_2x2.md`](head_and_aggregation_2x2.md) | the missing 2x2 cell: learned attention with an MLP head | — |
| [`head_and_aggregation_2x2_first_version.md`](head_and_aggregation_2x2_first_version.md) | the missing 2x2 cell, first version | — |
| [`hybrid_backoff_reproduction.md`](hybrid_backoff_reproduction.md) | implement hybrid back-off and validate it before using it as a baseline | — |
| [`id_attention_entropy_as_drop_predictor.md`](id_attention_entropy_as_drop_predictor.md) | does in-distribution attention entropy alone predict the OOD drop? | — |
| [`label_free_gate_experiments.md`](label_free_gate_experiments.md) | two label-free gate attempts | — |
| [`label_free_regime_taxonomy.md`](label_free_regime_taxonomy.md) | is the regime taxonomy decidable from unlabelled data? | — |
| [`lehmer_aggregation_qwen.md`](lehmer_aggregation_qwen.md) | Lehmer beta = 1 on the Qwen2.5-14B grid, out of sample | run. Q1 fails (margin +0.0137 and 7/8 pass, Wilcoxon p = 0.195, dragged by expertqa, which carries a severe length confound on this population); Q2 replicates on 1 of 3 (cnn_dailymail, boot… |
| [`med_quad_degeneracy_gate.md`](med_quad_degeneracy_gate.md) | MedQuAD's degeneracy gate is different from the other three | — |
| [`multi_recipe_pooling_heads.md`](multi_recipe_pooling_heads.md) | pooling heads with different fixed recipes | — |
| [`multimodel_far_ood_panel.md`](multimodel_far_ood_panel.md) | multi-model replication of the shrinkage effect under cross-task shift | — |
| [`nll_as_pooler_feature.md`](nll_as_pooler_feature.md) | token probabilities as a pooler input | — |
| [`one_sided_entropy_penalty.md`](one_sided_entropy_penalty.md) | the one-sided entropy penalty | — |
| [`per_instance_blend_of_min_and_weighted.md`](per_instance_blend_of_min_and_weighted.md) | per-instance length blend of minimum token probability and activation-weighted surprisal | run. Null on the registered same-family pair, with both controls decisive (shuffled-length matches; constant-blend shows the residue is two-score decorrelation). |
| [`per_instance_length_blend.md`](per_instance_length_blend.md) | the per-instance length blend | run. The registered null: the per-instance length blend does not beat the better endpoint. |
| [`prior_tilt_beta_sweep.md`](prior_tilt_beta_sweep.md) | sweep the prior-tilt strength beta | — |
| [`prompt_residual_pooling.md`](prompt_residual_pooling.md) | prompt-residual hidden-state probing | — |
| [`punctuation_ablation.md`](punctuation_ablation.md) | are PubMedQA's punctuation tokens carrying the signal? | — |
| [`qwen14b_replication.md`](qwen14b_replication.md) | the Qwen2.5-14B replication of the ProbeDriftLong findings | — |
| [`qwen_coverage_divergence_prediction.md`](qwen_coverage_divergence_prediction.md) | predicted label-coverage divergence for Qwen2.5-14B | — |
| [`response_length_vs_selection_law.md`](response_length_vs_selection_law.md) | is the selection law really about response length? | — |
| [`samsum_budget_regeneration_pilot.md`](samsum_budget_regeneration_pilot.md) | the SAMSum budget regeneration pilot | — |
| [`sharpening_family_lodo_selection.md`](sharpening_family_lodo_selection.md) | honest selection across the sharpening families, and a rank-weighted arm | run. Lehmer under raw-argmax LODO is a consistent small positive (+0.027, 6/8) that misses its registered bar (p = 0.250); the rank arm removes the length confound and still fails on PRR.… |
| [`shrinkage_lambda_and_nll_prior.md`](shrinkage_lambda_and_nll_prior.md) | lambda = 1.5 from a July prediction, and token NLL inside the weight logits | run 8/8. The registered λ = 1.5 claim FAILS (tie with shrink@2); the anchor-quality mechanism holds (Spearman +0.835, LOO-stable). |
| [`shrinkage_mechanism.md`](shrinkage_mechanism.md) | shrinkage mechanism diagnostic and configuration-selection audit | — |
| [`shrinkage_strength_selection_protocol.md`](shrinkage_strength_selection_protocol.md) | retrospective development-set selection of the shrinkage level | — |
| [`softmax_sharpening_axis.md`](softmax_sharpening_axis.md) | the training-free sharpening family | run. The primary claim FAILED its registered bar (margin +0.0209 PASS, signs 6/8 PASS, Wilcoxon p = 0.148 FAIL — NOT ESTABLISHED); both honest selection arms collapsed to msp_min on every… |
| [`source_relative_weighting.md`](source_relative_weighting.md) | source-relative rank supervision for activation-weighted surprisal | — |
| [`split_rule_and_label_coverage.md`](split_rule_and_label_coverage.md) | the split rule and label coverage | — |
| [`topk_surprisal_prior.md`](topk_surprisal_prior.md) | the top-k surprisal prior | — |
| [`xl_contribution_provenance.md`](xl_contribution_provenance.md) | ProbeDrift-XL contribution ladder — reconstructed provenance | — |
