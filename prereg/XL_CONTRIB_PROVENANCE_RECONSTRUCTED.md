# XL contribution ladder — RECONSTRUCTED provenance (not stamped)

**Written 2026-08-05.** The 600 rows in `results/xlcontrib_fam_*__meta-llama_Meta-Llama-3.1-8B.csv` carry
**no `git_sha`**, because `contribution_ladder.py` did not stamp provenance until `fe55cf1` (2026-08-05
02:49), which is after those runs completed. Every future run stamps; these ten cannot be fixed except by
re-running ~45 GPU-hours purely for a label.

⚠️ **This file is RECONSTRUCTED EVIDENCE, not a stamp.** It records what the artifacts, job logs and git
history jointly imply. It is weaker than a stamp and is labelled as such wherever it is relied on. It
exists so the gap is documented rather than silently absent.

## The claim

All ten runs executed the code at **`7f79e19`** ("Both XL drivers use the shared rung builder and all ten
sources", 2026-08-04 10:42:26 +0100).

## The evidence

Every job started within 3 minutes of that commit, and the next commit touching the driver or any of its
numerically-relevant imports came 6.5 hours later:

| eval | job | started | log |
|---|---|---|---|
| sciq | 3553961 | 2026-08-04 10:44:12 | `luq_xl2.o3553961` |
| trivia_qa | 3553962 | 10:44:17 | `luq_xl2.o3553962` |
| pubmed_qa | 3553963 | 10:44:24 | `luq_xl2.o3553963` |
| med_quad | 3553964 | 10:44:31 | `luq_xl2.o3553964` |
| asqa | 3553965 | 10:44:34 | `luq_xl2.o3553965` |
| xsum | 3553966 | 10:44:41 | `luq_xl2.o3553966` |
| cnn_dailymail | 3553967 | 10:44:51 | `luq_xl2.o3553967` |
| samsum | 3553968 | 10:44:56 | `luq_xl2.o3553968` |
| expertqa | 3553969 | 10:45:03 | `luq_xl2.o3553969` |
| factscore | 3553970 | 10:44:11 | `luq_xl2.o3553970` |

Commits touching `contribution_ladder.py`, `xl_rungs.py`, `weighted_msp.py`, `attn_pool.py`,
`aggregation_table.py` or `results.py` after `7f79e19`:

- **`40da460`** (2026-08-04 17:22:41) — some jobs were still running at this point, but a running Python
  process holds the code it imported at start, so they were unaffected. Its one change to this driver's
  OUTPUT was the per-rung `different_label_projection` flag, which is why
  `scripts/checks/fix_crosslabel_flag.py` was applied to the finished CSVs afterwards (56 rows re-flagged,
  verified metadata-only against the job logs: 350 PRR values cross-checked, 0 differences above 0.0005).
- **`184686f`** (2026-08-05 01:19:20) — the segment-softmax NaN fix. **Cannot affect these rows:**
  `contribution_ladder` never passes `segment_ids` and never uses `segment_mode="softmax"`, so
  `_segment_softmax_weights` is not on its call path. Its wMSP methods are `weighted_msp_norm` and
  `weighted_msp_unc` only.
- **`fe55cf1`** (2026-08-05 02:49:08) — adds the stamping itself; no numerical change.

## Independent corroboration

- **0 NaN and 0 blanks** across all 600 rows.
- The unlabelled-row filter that contaminated three `ood_onegrid` jobs has been present in
  `contribution_ladder` since **`e1b5b65` (2026-07-20)**, so this driver was never exposed to that defect.
- All ten evals reached the full 5 rungs × 11 methods, including expertqa's SameTask rung, which only
  exists when factscore is in the source pool — direct evidence the runs used the post-`7f79e19` cohort.

## What this does NOT establish

The environment. `env_hash` was not recorded either, so the installed package set at run time is not
pinned for these rows. Future runs capture it.
