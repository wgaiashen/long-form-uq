# What label is each results file scored against? (READ THIS FIRST)

> The judge-vs-AlignScore mix-up has bitten this project more than once. This file is the
> authoritative record of **which correctness label the `correctness` column of every results
> file actually holds.** PRR is only comparable across files that share a label. When you add or
> regenerate a results file, update this note.

## The two labels (never mix them in one comparison)

- **Judge** — Joe's LLM-as-a-judge (`gpt-5-2025-08-07` on sciq/trivia; `gpt-5-mini` on the
  long-form sets pubmed/xsum/med_quad/expertqa). Scores the output against the GOLD reference.
  **This is the primary label** for every headline table. It lives in the cached **records**
  (`cache/records/*.jsonl`) as the bare `correctness` field.
- **AlignScore** (`yzha/AlignScore-large`) — a free secondary NLI-style signal, **near-dead on
  long-form** (its per-example mean on xsum/pubmed is ~0.05). Lives in the records as
  `correctness_alignscore`, and — the gotcha below — in the `correctness` column of the
  per-example CSVs.
- (`correctness_strmatch` — the short-form string-match label, kept but not primary.)

## The gotcha, in one line

`scripts/04_eval.py --label-field X` writes whatever label `X` names into a column **always called
`correctness`**. The per-example CSVs on disk were written with `--label-field
correctness_alignscore`, so **their `correctness` column is AlignScore, NOT the judge.** To get
judge-labelled PRR from a per-example CSV you must re-join to the records' `correctness` field
(align by `msp_sum`, which both carry, then read the judge label off the record).

Since 2026-07-06, `04_eval.py` **stamps `label_field` + `label_model` columns** into every CSV it
writes, and the existing per-example CSVs have been retro-stamped, so the label is now visible in
the file itself — check those two columns before computing PRR.

## Per-file label map (meta-llama_Meta-Llama-3.1-8B, the one live model)

| File | `correctness` column = | Notes |
|---|---|---|
| `meta-llama_..._{sciq,trivia_qa,pubmed_qa,xsum}__ID.csv` (per-example) | **AlignScore** | now stamped `label_field=correctness_alignscore`. The method score columns (saplma/linear/ptrue_accurate/lookback/uhead) are label-agnostic — re-join to records for judge PRR. |
| `aggregation_table__*.csv` | **judge** (+ a second block of AlignScore rows) | already self-stamps `label_field`/`label_model` per row. |
| `ood_onegrid__*.csv` | **judge** | reads the bare `correctness` field from records; no per-row stamp (documented here). |
| `contribution_ladder__*.csv` | **judge** | ditto. weighted-MSP + MSP floor + poolers. |
| `ood_floor__*.csv` | **judge** | MSP-family floor, recomputed from record logprobs vs the judge label. |
| `transfer_matrix__*__ID.csv`, `pooled_loo__*__ID.csv` | **judge** | baseline transfer/LOO grids. |
| `Qwen_*`, `google_gemma-2-9b-it_*` per-example CSVs | AlignScore (dev/scaling era) | **dropped models — ignore.** Kept only as history. |

## Rule going forward

1. Headline numbers = **judge**. If a file's `correctness` is AlignScore, either re-join to
   records or regenerate with `--label-field correctness` before quoting PRR.
2. Every new results file either self-stamps `label_field`/`label_model` (preferred) or gets a row
   in the table above.
3. Never compute a verdict (method A vs method B) across files with different labels.
