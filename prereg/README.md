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

Read those outcomes precisely, because "failed its bar" covers three different things here and they
are not interchangeable. One registration fails because the setting it pre-committed to **tied** with
the incumbent rather than beating it. Several fail because the effect is positive and consistent in
direction but **short of significance** at n = 8 datasets. Others record a genuine **absence of
effect**, and those say so in those words. Only the third kind is a null in the ordinary sense.

| Registration | What it registered | Recorded outcome |
|---|---|---|
| [`adaptive_lehmer_aggregation.md`](adaptive_lehmer_aggregation.md) | per-response adaptive Lehmer aggregation | run. A registered negative: the primary claim fails its bar and the method loses to its own fixed-coefficient control. |
| [`anchor_at_min_token_probability.md`](anchor_at_min_token_probability.md) | regularise activation-weighted surprisal toward the minimum token probability, not toward mean token NLL | the linear penalty stalled grid-wide (an optimisation failure, §16.1); the registered −log p[k] fallback ran 8/8 (F5b): mechanism confirmed both di… |
| [`anchor_warm_start_combination.md`](anchor_warm_start_combination.md) | make the anchor testable everywhere, and try shrinkage and anchoring together | run complete 8/8. The testability bar failed (same 3/8 evals bite as F5b) and the combo is a null against the incumbent — F5 closed as mechanism-only. |
| [`attention_entropy_shift_as_drop_predictor.md`](attention_entropy_shift_as_drop_predictor.md) | does attention flattening predict performance drop? | run. A clean negative under the registered decision rule: the two correlation statistics disagree in sign and the rank statistic is wrong-signed, s… |
| [`cawsa_saplma_ensemble.md`](cawsa_saplma_ensemble.md) | does CAWSA supply a complementary signal to SAPLMA? | run. The combination improves on the probe alone in mean out-of-distribution PRR on every population tested, and beats the matched control. |
| [`cawsa_saplma_ensemble_llama_instruct_replication.md`](cawsa_saplma_ensemble_llama_instruct_replication.md) | does the complementarity result replicate on an instruction-tuned population? | run and resolved. The result was not carried into the report, which does not evaluate this population. |
| [`cawsa_saplma_ensemble_qwen_replication.md`](cawsa_saplma_ensemble_qwen_replication.md) | does the CAWSA + SAPLMA complementarity result replicate on Qwen2.5-14B? | run. Replicates, and is the clearest dataset-level effect of the three populations. |
| [`entropy_penalty_on_spread_datasets.md`](entropy_penalty_on_spread_datasets.md) | the one-sided entropy penalty on the untested spread regime | run. The predicted null on the untested regime, with the mechanism confirming it. |
| [`head_and_aggregation_2x2.md`](head_and_aggregation_2x2.md) | the missing 2x2 cell: learned attention with an MLP head | run. A clean negative: the apparent head effect that motivated it was a training-recipe artefact, and all four correctness gates passed. |
| [`head_and_aggregation_2x2_first_version.md`](head_and_aggregation_2x2_first_version.md) | the missing 2x2 cell, first version | — |
| [`hybrid_backoff_reproduction.md`](hybrid_backoff_reproduction.md) | implement hybrid back-off and validate it before using it as a baseline | run. The reproduction was validated on its own terms before the method was used as a baseline anywhere. |
| [`id_attention_entropy_as_drop_predictor.md`](id_attention_entropy_as_drop_predictor.md) | does in-distribution attention entropy alone predict the OOD drop? | run. Fails all three registered bars at the honest unit of analysis, and the registered stopping rule closed this line. |
| [`label_free_gate_experiments.md`](label_free_gate_experiments.md) | two label-free gate attempts | run. Three gate families, three shuffled controls, three clean nulls, with the per-dataset pattern the opposite of the registered prediction. |
| [`label_free_regime_taxonomy.md`](label_free_regime_taxonomy.md) | is the regime taxonomy decidable from unlabelled data? | run. Falsified: the taxonomy is not decidable from unlabelled data by the registered statistic, and plain response length separated the groups better. |
| [`lehmer_aggregation_qwen.md`](lehmer_aggregation_qwen.md) | Lehmer beta = 1 on the Qwen2.5-14B grid, out of sample | run. Q1 fails (margin +0.0137 and 7/8 pass, Wilcoxon p = 0.195, dragged by expertqa, which carries a severe length confound on this population); Q2… |
| [`M10_source_calibrated_cawsa_saplma_fusion.md`](M10_source_calibrated_cawsa_saplma_fusion.md) | a source-calibrated fusion that can score one response on its own | run. Complete on two populations at 40 of 40 cells, with all four registered gates passing exactly. The third population was excluded on a pre-spec… |
| [`M11_alllayer_published_distance_baselines.md`](M11_alllayer_published_distance_baselines.md) | the published distance family at its full layer set | run. The reproduction completed at the full published layer set, 32 of 32 layers at 40 of 40 cells, with every gate passed. The registered predicti… |
| [`M7_published_hybrid_baselines.md`](M7_published_hybrid_baselines.md) | the published probability, distance and hybrid comparators | run. The comparator grid completed on the corrected-span populations, and every published distance and hybrid method evaluated here lands below the… |
| [`M8_cawsa_hbo_substitution.md`](M8_cawsa_hbo_substitution.md) | substituting a learned token weighting for the hybrid back-off's probability branch | run. Substituting the probability branch raises in-distribution PRR substantially and leaves the shifted settings close to the substituted score, b… |
| [`M9_layer_distance_sensitivity.md`](M9_layer_distance_sensitivity.md) | layer sensitivity for the supervised distance family | run and stopped. Extraction completed at 88 of 88 layer files and the first acceptance gate failed, so under this registration no result was comput… |
| [`med_quad_degeneracy_gate.md`](med_quad_degeneracy_gate.md) | MedQuAD's degeneracy gate is different from the other three | run. The predicted difference held, and the gate failure it anticipated was read as predicted rather than as a defect. |
| [`multi_recipe_pooling_heads.md`](multi_recipe_pooling_heads.md) | pooling heads with different fixed recipes | run. The fixed-recipe heads do not beat the single-head pooler. |
| [`multimodel_far_ood_panel.md`](multimodel_far_ood_panel.md) | multi-model replication of the shrinkage effect under cross-task shift | run. Two replication populations completed and were carried into the report; a third was withdrawn on a failed generation-validity gate before any… |
| [`nll_as_pooler_feature.md`](nll_as_pooler_feature.md) | token probabilities as a pooler input | run to a complete 40-cell grid, with the margins against the learned-attention control recorded in the result files. No written verdict was produce… |
| [`one_sided_entropy_penalty.md`](one_sided_entropy_penalty.md) | the one-sided entropy penalty | run. Negative, and the diagnosis that motivated it was itself shown to be wrong. |
| [`per_instance_blend_of_min_and_weighted.md`](per_instance_blend_of_min_and_weighted.md) | per-instance length blend of minimum token probability and activation-weighted surprisal | run. Null on the registered same-family pair, with both controls decisive (shuffled-length matches; constant-blend shows the residue is two-score d… |
| [`per_instance_length_blend.md`](per_instance_length_blend.md) | the per-instance length blend | run. The registered null: the per-instance length blend does not beat the better endpoint. |
| [`prior_tilt_beta_sweep.md`](prior_tilt_beta_sweep.md) | sweep the prior-tilt strength beta | run. An interior optimum on every dataset, which overturned an earlier conclusion drawn from two in-distribution cells. |
| [`prompt_residual_pooling.md`](prompt_residual_pooling.md) | prompt-residual hidden-state probing | run. Complete at 160 of 160 cells with both reproduction gates passing; the method did not clear its gate. |
| [`punctuation_ablation.md`](punctuation_ablation.md) | are PubMedQA's punctuation tokens carrying the signal? | run. The premise it was written to test was only weakly supported, so the causal reading that depended on it is stated as a hypothesis rather than… |
| [`qwen14b_replication.md`](qwen14b_replication.md) | the Qwen2.5-14B replication of the ProbeDriftLong findings | run. Two of the four registered claims replicate, one fails and one reverses. Scored independently on two clusters with the same committed scorer,… |
| [`qwen_coverage_divergence_prediction.md`](qwen_coverage_divergence_prediction.md) | predicted label-coverage divergence for Qwen2.5-14B | run. The predicted coverage divergence matched the observed counts to the row on both affected datasets. |
| [`response_length_vs_selection_law.md`](response_length_vs_selection_law.md) | is the selection law really about response length? | run. The registered direction and thresholds were applied; the per-dataset record is in the result files. |
| [`samsum_budget_regeneration_pilot.md`](samsum_budget_regeneration_pilot.md) | the SAMSum budget regeneration pilot | run. The pilot informed the token budget and the truncation convention. |
| [`sharpening_family_lodo_selection.md`](sharpening_family_lodo_selection.md) | honest selection across the sharpening families, and a rank-weighted arm | run. Lehmer under raw-argmax LODO is a consistent small positive (+0.027, 6/8) that misses its registered bar (p = 0.250); the rank arm removes the… |
| [`shrinkage_lambda_and_nll_prior.md`](shrinkage_lambda_and_nll_prior.md) | lambda = 1.5 from a July prediction, and token NLL inside the weight logits | run 8/8. The registered λ = 1.5 claim FAILS (tie with shrink@2); the anchor-quality mechanism holds (Spearman +0.835, LOO-stable). |
| [`shrinkage_mechanism.md`](shrinkage_mechanism.md) | shrinkage mechanism diagnostic and configuration-selection audit | run. The mechanism diagnostic holds, and the configuration-selection audit is recorded as retrospective rather than prospective. |
| [`shrinkage_strength_selection_protocol.md`](shrinkage_strength_selection_protocol.md) | retrospective development-set selection of the shrinkage level | run. The two missing coefficients were computed and the selection is recorded as retrospective; no clean prospective development-set choice of the… |
| [`softmax_sharpening_axis.md`](softmax_sharpening_axis.md) | the training-free sharpening family | run. The primary claim FAILED its registered bar (margin +0.0209 PASS, signs 6/8 PASS, Wilcoxon p = 0.148 FAIL — NOT ESTABLISHED); both honest sele… |
| [`source_relative_weighting.md`](source_relative_weighting.md) | source-relative rank supervision for activation-weighted surprisal | run, under the condition its registration set. |
| [`split_rule_and_label_coverage.md`](split_rule_and_label_coverage.md) | the split rule and label coverage | run. The rule fired mechanically on the second model, as intended. |
| [`topk_surprisal_prior.md`](topk_surprisal_prior.md) | the top-k surprisal prior | run. Complete on the long grid; the prior did not carry. |
| [`xl_contribution_provenance.md`](xl_contribution_provenance.md) | ProbeDrift-XL contribution ladder — reconstructed provenance | — |
