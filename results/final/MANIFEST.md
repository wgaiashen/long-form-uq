# CANONICAL-NUMBERS INDEX for the report — a POINTER LIST, not a copy
(Re-designed 2026-08-09: while work is live, files stay in `results/` ONLY — a second copy here
would need double maintenance and would drift. At SUBMISSION this manifest gains sha256 checksums
and the listed files are frozen in place. Until then: this file is the index of what the report may
quote; everything lives at `results/<name>`.)

⭐ **READ THESE TWO FIRST — they are the report's data spine:**
1. `pdl_master__*.csv` — the base ladder (all baselines, every cell).
2. `sharpening_master__*.csv` — EVERY post-7-Aug sharpening result in one long-format table
   (4479 rows; workstream/method/param/rung/eval/prr/mode/source). Regenerate with
   `python scripts/checks/build_sharpening_master.py` after any new run — NEVER hand-edit; the
   per-dataset CSVs below are its raw provenance and stay for audit only.

# Detail (frozen 2026-08-09, drivers at git 4410f59)
Nothing enters the report that is not in this directory. EXCLUDED BY DECISION: any shrink@2-as-
incumbent framing (superseded, stocktake §18), the λ=10 oracle repair figure (§18.2: quote the
LODO-λ +0.042 over shrink@1.5), the withdrawn +0.2472 oracle (§8).

| file(s) | produced by | PBS jobs | report section |
|---|---|---|---|
| pdl_master__*.csv | assemble_pdl_table.py (pre-existing) | July/Aug ladder jobs | base ladder, all baselines |
| sharpening_family__*__round2.csv | sharpening_family.py --round2 | 3624367 | free families, Lehmer, rank arm |
| sharpening_family__*__lengthtau.csv | --length-tau | 3623070 | length-conditioned τ negative |
| sharpening_lambda_<eval>__*.csv (8) + verdict | sharpening_lambda.py | 3628840-47 | λ ladder incl. shrink@1.5 (THE INCUMBENT, §18) |
| mask_ablation_<eval>__*.csv (8) + verdict | mask_ablation.py | 3628915-22 | EOS mask null |
| anchor_msp_min_<eval>__logpen__*.csv (8) + anchor_verdict | anchor_msp_min.py --penalty log | 3630658-65 | F5b mechanism + repair at LODO-λ |
| orthogonality_map__*.csv | orthogonality_map.py | login (post-hoc) | seed-reliability vs cross-method |
| regime_map__*.csv | regime_map.py | login (post-hoc) | win map, degradation, F3 unidentifiability |
| oracle_shuffle_audit__*.csv | oracle_shuffle_audit.py | login (post-hoc) | §8 retractions |
| special_token_audit / eos_provenance | respective scripts | login (post-hoc) | §12 / §13 |
| sharpening_selection_audit__*.csv | selection_audit.py | login (post-hoc) | §5 (3-arm grid; see §5.1 amendment) |
| blend_msp_wmsp_<eval>__*.csv (8) | blend_msp_wmsp.py | 3632998-3633005 | §19 per-instance null |
PENDING ENTRY when it lands: F5c __logws grid (jobs 3632079-86).
