"""Canonical method names — one name per method, applied at ANALYSIS time.

WHY. Results accumulated across many waves and the same method acquired different labels in different
drivers: `saplma` / `mean-pool+MLP` / `SAPLMA mean-pool` are one method; so are `msp_sum` / `MSP floor` /
`msp_sum (floor)` / `floor_sum` / `msp_floor`. 66 raw strings across the CSVs correspond to ~25 actual
methods. That fragmentation is why the coverage grid looked far sparser and messier than it is.

DESIGN DECISION. This maps names when READING results. It deliberately does NOT rewrite the committed CSVs
or the drivers, because (a) jobs were running when this was written and (b) the project rule is that results
are additive and never rewritten in place. Drivers can adopt these names later; nothing breaks if they don't.

`FAMILY` groups methods for presentation. `canonical()` is safe on unknown strings (returns them unchanged),
so a new method shows up as itself rather than being silently dropped.
"""

# raw name -> canonical name
CANONICAL = {
    # ---- unsupervised floors ----
    "msp_sum": "MSP-sum", "MSP floor": "MSP-sum", "msp_sum (floor)": "MSP-sum",
    "floor_sum": "MSP-sum", "msp_floor": "MSP-sum", "weighted_msp_msp_sum": "MSP-sum",
    "perplexity": "Perplexity", "perplexity (floor)": "Perplexity", "floor_ppl": "Perplexity",
    "msp_min": "MSP-min", "floor_min": "MSP-min",
    "fair_floor": "Fair floor", "weighted_msp_fair_floor": "Fair floor",
    "fair_floor:msp_sum": "Fair floor", "fair_floor:msp_min": "Fair floor",
    "fair_floor:perplexity": "Fair floor", "msp_family": "MSP-family (best)",

    # ---- supervised probes on pooled features ----
    "saplma": "SAPLMA (mean-pool)", "mean-pool+MLP": "SAPLMA (mean-pool)",
    "SAPLMA mean-pool": "SAPLMA (mean-pool)",
    "last-token": "SAPLMA (last-token)", "SAPLMA last-token": "SAPLMA (last-token)",
    "per-sentence": "SAPLMA (per-sentence)", "per-sentence(mean)": "SAPLMA (per-sentence)",
    "SAPLMA per-sentence": "SAPLMA (per-sentence)",
    "per-token": "SAPLMA (per-token)", "per-token(mean)": "SAPLMA (per-token)",
    "SAPLMA per-token": "SAPLMA (per-token)",
    "linear": "Linear probe", "ptrue": "P(True)", "ptrue_accurate": "P(True)",
    "lookback": "Lookback Lens",

    # ---- learned aggregation over token states ----
    "uniform": "Uniform pooler (mean-pool)", "uniform(frozen-q)": "Uniform pooler (mean-pool)",
    "attention": "Attention pooler",
    "attn_shrink2": "Attention pooler +shrink@2", "attn_shrink10": "Attention pooler +shrink@10",
    "hier": "Hierarchical pooler (2-level)",
    "hier_seg": "Hierarchical: sentence-choice only", "hier_tok": "Hierarchical: token-choice only",

    # ---- weighted-MSP (score side) ----
    "weighted_msp_norm": "wMSP-normalised", "wmsp_norm": "wMSP-normalised",
    "wMSP-normalised": "wMSP-normalised", "weighted-MSP norm": "wMSP-normalised",
    "weighted_msp": "wMSP-normalised", "weighted_msp_pairwise": "wMSP-normalised",
    "weighted_msp_norm_pairwise": "wMSP-normalised", "weighted_msp_unmasked": "wMSP-normalised",
    "weighted_msp_unc": "wMSP-unconstrained", "wMSP-unconstrained": "wMSP-unconstrained",
    "weighted_msp_blondel": "wMSP-Blondel", "wmsp_blondel": "wMSP-Blondel",
    "weighted-MSP Blondel": "wMSP-Blondel",
    "wmsp_shrink2": "wMSP-shrink@2", "wmsp_shrink10": "wMSP-shrink@10",
    # STANDARD-ladder names (weighted_msp_all_variants uses these bare forms, no `wmsp_` prefix).
    # These were UNMAPPED, so our primary method was invisible on our primary ladder (2026-07-23).
    "shrink@2": "wMSP-shrink@2", "shrink@10": "wMSP-shrink@10",
    "shrink@2-blondel": "wMSP-shrink@2 +Blondel", "shrink@10-blondel": "wMSP-shrink@10 +Blondel",
    "kl@2": "wMSP-KL@2", "entropy_hinge@2": "wMSP-entropy-hinge@2",
    "smooth_n3": "wMSP-smooth(n=3)", "smooth_n5": "wMSP-smooth(n=5)",
    "wmsp_shrink2_blondel": "wMSP-shrink@2 +Blondel",
    "wmsp_shrink10_blondel": "wMSP-shrink@10 +Blondel",
    "weighted_msp_orgad": "wMSP-Orgad-masked",

    # ---- weighted-MSP token-subset variants (keep-masks) ----
    "special": "wMSP keep=no-EOS (default)", "special_punct": "wMSP keep=no-EOS+punct",
    "content": "wMSP keep=content (no stop-words)", "segment": "wMSP per-segment weight",

    # ---- decompose-and-aggregate (per-sentence probe + aggregator) ----
    "seg_mean": "Seg-probe + mean", "seg_min": "Seg-probe + min",
    "seg_geomean": "Seg-probe + geomean", "seg_learned": "Seg-probe + learned alpha",
}

FAMILY = {
    "MSP-sum": "1. Unsupervised floor", "Perplexity": "1. Unsupervised floor",
    "MSP-min": "1. Unsupervised floor", "Fair floor": "1. Unsupervised floor",
    "MSP-family (best)": "1. Unsupervised floor",
    "SAPLMA (mean-pool)": "2. Supervised probe", "SAPLMA (last-token)": "2. Supervised probe",
    "SAPLMA (per-sentence)": "2. Supervised probe", "SAPLMA (per-token)": "2. Supervised probe",
    "Linear probe": "2. Supervised probe", "P(True)": "2. Supervised probe",
    "Lookback Lens": "2. Supervised probe",
    "Uniform pooler (mean-pool)": "3. Learned pooling", "Attention pooler": "3. Learned pooling",
    "Attention pooler +shrink@2": "3. Learned pooling", "Attention pooler +shrink@10": "3. Learned pooling",
    "Hierarchical pooler (2-level)": "3. Learned pooling",
    "Hierarchical: sentence-choice only": "3. Learned pooling",
    "Hierarchical: token-choice only": "3. Learned pooling",
    "wMSP-normalised": "4. Weighted-MSP", "wMSP-unconstrained": "4. Weighted-MSP",
    "wMSP-Blondel": "4. Weighted-MSP", "wMSP-shrink@2": "4. Weighted-MSP",
    "wMSP-shrink@10": "4. Weighted-MSP", "wMSP-shrink@2 +Blondel": "4. Weighted-MSP",
    "wMSP-shrink@10 +Blondel": "4. Weighted-MSP", "wMSP-Orgad-masked": "4. Weighted-MSP",
    "wMSP-KL@2": "4. Weighted-MSP", "wMSP-entropy-hinge@2": "4. Weighted-MSP",
    "wMSP-smooth(n=3)": "4. Weighted-MSP", "wMSP-smooth(n=5)": "4. Weighted-MSP",
    "wMSP keep=no-EOS (default)": "5. wMSP token subsets",
    "wMSP keep=no-EOS+punct": "5. wMSP token subsets",
    "wMSP keep=content (no stop-words)": "5. wMSP token subsets",
    "wMSP per-segment weight": "5. wMSP token subsets",
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
