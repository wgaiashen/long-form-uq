"""Canonical method names, applied when results are read rather than when they are written.

Results accumulated over many runs and the same method acquired different labels in different
drivers: `saplma`, `mean-pool+MLP` and `SAPLMA mean-pool` are one method, and so are `msp_sum`,
`MSP floor`, `msp_sum (floor)`, `floor_sum` and `msp_floor`. Roughly 66 raw strings across the
result CSVs correspond to about 25 actual methods, which made the coverage grid look far sparser
than it is.

The display names here are the ones used in the write-up, so a table built from these matches the
report's terminology: `wmsp_shrink2` is CAWSA at lambda = 2, `wmsp_norm` is the unconstrained
activation-weighted precursor, and the three training-free aggregates are Sum NLL, Mean token NLL
and Minimum token probability. The implementation keys are deliberately left alone, in the code and
in the committed CSVs, so nothing has to be rewritten in place.

`FAMILY` groups methods for presentation. `canonical()` is safe on unknown strings and returns them
unchanged, so a new method appears as itself rather than being silently dropped.
"""

# raw name -> canonical name
CANONICAL = {
    # ---- unsupervised floors ----
    "msp_sum": "Sum NLL", "MSP floor": "Sum NLL", "msp_sum (floor)": "Sum NLL",
    "floor_sum": "Sum NLL", "msp_floor": "Sum NLL", "weighted_msp_msp_sum": "Sum NLL",
    "perplexity": "Mean token NLL", "perplexity (floor)": "Mean token NLL", "floor_ppl": "Mean token NLL",
    "msp_min": "Minimum token probability", "floor_min": "Minimum token probability",
    "fair_floor": "Fair floor", "weighted_msp_fair_floor": "Fair floor",
    "fair_floor:msp_sum": "Fair floor", "fair_floor:msp_min": "Fair floor",
    "fair_floor:perplexity": "Fair floor", "msp_family": "Best probability aggregate",

    # ---- supervised probes on pooled features ----
    "saplma": "SAPLMA", "mean-pool+MLP": "SAPLMA",
    "SAPLMA mean-pool": "SAPLMA",
    "last-token": "SAPLMA (last-token)", "SAPLMA last-token": "SAPLMA (last-token)",
    "per-sentence": "SAPLMA (per-sentence)", "per-sentence(mean)": "SAPLMA (per-sentence)",
    "SAPLMA per-sentence": "SAPLMA (per-sentence)",
    "per-token": "SAPLMA (per-token)", "per-token(mean)": "SAPLMA (per-token)",
    "SAPLMA per-token": "SAPLMA (per-token)",
    "linear": "Mean-pool probe",
    # Table 4.1 lists two distinct P(True) methods and they must not collapse into one: the
    # verdict-state probe reads the hidden state at the verification position, while the
    # training-free variant reads the emitted yes/no probability.
    "ptrue": "P(True) verdict-state probe", "ptrue_accurate": "P(True) verdict-state probe",
    "ptrue_unsup": "P(True), training-free",
    "lookback": "Lookback Lens",

    # ---- learned aggregation over token states ----
    "uniform": "Mean-pool control", "uniform(frozen-q)": "Mean-pool control",
    "attention": "Learned attention pooling",
    "attn_shrink2": "Learned attention pooling +shrink@2", "attn_shrink10": "Learned attention pooling +shrink@10",
    "hier": "Hierarchical pooler (2-level)",
    "hier_seg": "Hierarchical: sentence-choice only", "hier_tok": "Hierarchical: token-choice only",

    # ---- weighted-MSP (score side) ----
    "weighted_msp_norm": "Unconstrained activation weighting", "wmsp_norm": "Unconstrained activation weighting",
    "Unconstrained activation weighting": "Unconstrained activation weighting", "weighted-MSP norm": "Unconstrained activation weighting",
    "weighted_msp": "Unconstrained activation weighting", "weighted_msp_pairwise": "Unconstrained activation weighting",
    "weighted_msp_norm_pairwise": "Unconstrained activation weighting", "weighted_msp_unmasked": "Unconstrained activation weighting",
    "weighted_msp_unc": "Unconstrained activation weighting (unnormalised)", "Unconstrained activation weighting (unnormalised)": "Unconstrained activation weighting (unnormalised)",
    "weighted_msp_blondel": "Activation-weighted +Blondel loss", "wmsp_blondel": "Activation-weighted +Blondel loss",
    "weighted-MSP Blondel": "Activation-weighted +Blondel loss",
    "wmsp_shrink2": "CAWSA (lambda=2)", "wmsp_shrink10": "CAWSA (lambda=10)",
    # STANDARD-ladder names (weighted_msp_all_variants uses these bare forms, no `wmsp_` prefix).
    # These were UNMAPPED, so our primary method was invisible on our primary ladder (2026-07-23).
    "shrink@2": "CAWSA (lambda=2)", "shrink@10": "CAWSA (lambda=10)",
    "shrink@2-blondel": "CAWSA (lambda=2) +Blondel loss", "shrink@10-blondel": "CAWSA (lambda=10) +Blondel loss",
    "kl@2": "Activation-weighted +KL penalty", "entropy_hinge@2": "Activation-weighted +entropy hinge",
    "smooth_n3": "Activation-weighted, smoothed (n=3)", "smooth_n5": "Activation-weighted, smoothed (n=5)",
    "wmsp_shrink2_blondel": "CAWSA (lambda=2) +Blondel loss",
    "wmsp_shrink10_blondel": "CAWSA (lambda=10) +Blondel loss",
    "weighted_msp_orgad": "Activation-weighted, answer-span masked",

    # ---- weighted-MSP token-subset variants (keep-masks) ----
    "special": "Activation-weighted keep=no-EOS (default)", "special_punct": "Activation-weighted keep=no-EOS+punct",
    "content": "Activation-weighted keep=content (no stop-words)", "segment": "Activation-weighted, per-segment",

    # ---- published probability, distance and hybrid comparators ----
    # Two naming conventions coexist in the result files. The ladder writes short implementation
    # keys; the published-baseline and combination drivers write report-style strings directly.
    # Both are mapped so a table built from either reads the same.
    "TokenSAR": "TokenSAR",
    "answer-span mean NLL": "Answer-span mean NLL",
    "answer-span sequence NLL": "Answer-span sequence NLL",
    "mean token NLL": "Mean token NLL",
    "minimum token probability": "Minimum token probability",
    # the published Maximum Sequence Probability, which is the sum aggregate under another name
    "sequence NLL (published MSP)": "Sum NLL", "msp": "Sum NLL",
    "md_mean_mid": "Raw MD, middle layer", "rmd_mean_mid": "Raw RMD, middle layer",
    "satmd_mid": "SATMD-mid", "satrmd_mid": "SATRMD-mid",
    "huq_satmd_mid": "HUQ-SATMD-mid", "huq_satrmd_mid": "HUQ-SATRMD-mid",
    "msp_satmd_mid": "MSP-SATMD-mid", "msp_satrmd_mid": "MSP-SATRMD-mid",
    "hbo": "HBO",

    # ---- score combinations, and the earlier display name for the constrained weighting ----
    "HAPES \u03bb=2": "CAWSA (lambda=2)", "cawsa": "CAWSA (lambda=2)",
    "cawsa_hbo": "HBO with the CAWSA probability branch",
    "cawsa_saplma_cohort50": "CAWSA + SAPLMA, equal-rank average",
    "cawsa_saplma_src50": "CAWSA + SAPLMA, source-calibrated",

    # ---- decompose-and-aggregate (per-sentence probe + aggregator) ----
    "seg_mean": "Seg-probe + mean", "seg_min": "Seg-probe + min",
    "seg_geomean": "Seg-probe + geomean", "seg_learned": "Seg-probe + learned alpha",
}

FAMILY = {
    "Sum NLL": "1. Training-free probability", "Mean token NLL": "1. Training-free probability",
    "Minimum token probability": "1. Training-free probability", "Fair floor": "1. Training-free probability",
    "Best probability aggregate": "1. Training-free probability",
    "SAPLMA": "2. Hidden-state probe", "SAPLMA (last-token)": "2. Hidden-state probe",
    "SAPLMA (per-sentence)": "2. Hidden-state probe", "SAPLMA (per-token)": "2. Hidden-state probe",
    "Mean-pool probe": "2. Hidden-state probe", "P(True)": "2. Hidden-state probe",
    "Lookback Lens": "2. Hidden-state probe",
    "Mean-pool control": "3. Learned pooling", "Learned attention pooling": "3. Learned pooling",
    "Learned attention pooling +shrink@2": "3. Learned pooling", "Learned attention pooling +shrink@10": "3. Learned pooling",
    "Hierarchical pooler (2-level)": "3. Learned pooling",
    "Hierarchical: sentence-choice only": "3. Learned pooling",
    "Hierarchical: token-choice only": "3. Learned pooling",
    "Unconstrained activation weighting": "4. Activation-weighted surprisal", "Unconstrained activation weighting (unnormalised)": "4. Activation-weighted surprisal",
    "Activation-weighted +Blondel loss": "4. Activation-weighted surprisal", "CAWSA (lambda=2)": "4. Activation-weighted surprisal",
    "CAWSA (lambda=10)": "4. Activation-weighted surprisal", "CAWSA (lambda=2) +Blondel loss": "4. Activation-weighted surprisal",
    "CAWSA (lambda=10) +Blondel loss": "4. Activation-weighted surprisal", "Activation-weighted, answer-span masked": "4. Activation-weighted surprisal",
    "Activation-weighted +KL penalty": "4. Activation-weighted surprisal", "Activation-weighted +entropy hinge": "4. Activation-weighted surprisal",
    "Activation-weighted, smoothed (n=3)": "4. Activation-weighted surprisal", "Activation-weighted, smoothed (n=5)": "4. Activation-weighted surprisal",
    "Activation-weighted keep=no-EOS (default)": "5. Activation-weighted token subsets",
    "Activation-weighted keep=no-EOS+punct": "5. Activation-weighted token subsets",
    "Activation-weighted keep=content (no stop-words)": "5. Activation-weighted token subsets",
    "Activation-weighted, per-segment": "5. Activation-weighted token subsets",
    "TokenSAR": "1. Training-free probability",
    "Answer-span mean NLL": "1. Training-free probability",
    "Answer-span sequence NLL": "1. Training-free probability",
    "P(True), training-free": "1. Training-free probability",
    "P(True) verdict-state probe": "2. Hidden-state probe",
    "Raw MD, middle layer": "7. Density", "Raw RMD, middle layer": "7. Density",
    "SATMD-mid": "7. Density", "SATRMD-mid": "7. Density",
    "MSP-SATMD-mid": "7. Density", "MSP-SATRMD-mid": "7. Density",
    "HUQ-SATMD-mid": "8. Hybrid", "HUQ-SATRMD-mid": "8. Hybrid", "HBO": "8. Hybrid",
    "HBO with the CAWSA probability branch": "8. Hybrid",
    "CAWSA + SAPLMA, equal-rank average": "8. Hybrid",
    "CAWSA + SAPLMA, source-calibrated": "8. Hybrid",
    "Seg-probe + mean": "6. Decompose & aggregate", "Seg-probe + min": "6. Decompose & aggregate",
    "Seg-probe + geomean": "6. Decompose & aggregate",
    "Seg-probe + learned alpha": "6. Decompose & aggregate",
}


def canonical(name):
    """Map a raw CSV method string to its canonical name. Unknown names pass through unchanged, so a new
    method is visible as itself rather than silently dropped."""
    return CANONICAL.get(name, name)


def family(name):
    return FAMILY.get(canonical(name), "9. Other")
