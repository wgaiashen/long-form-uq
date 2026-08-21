# `scripts/checks/` — the analysis and verification drivers

254 scripts. This is the largest directory in the repository and it is a **research record**, not a
library: one driver per experiment, kept as it was run. It is not a curated API and it is not meant
to be read end to end. This file explains the conventions so you can find what you need.

## The naming conventions

**`topic.py` / `topic_verdict.py`.** Most experiments are a pair. The first runs the experiment and
writes a CSV; the second reads that CSV and applies the pre-registered decision rule to it. The split
is deliberate: it keeps "produce the numbers" separate from "apply the rule to the numbers", so a
verdict cannot quietly change when a run is repeated. Examples:

| runs it | applies the rule |
|---|---|
| `sharpening_lambda.py` | `sharpening_lambda_verdict.py` |
| `mask_ablation.py` | `mask_ablation_verdict.py` |
| `anchor_msp_min.py` | `anchor_verdict.py` |
| `adaptive_lehmer.py` | `adaptive_lehmer_verdict.py` |
| `prompt_residual.py` | `prompt_residual_verdict.py` |
| `source_relative_wmsp.py` | `source_relative_verdict.py` |

**`check_*.py`** verify a ported method against the **original authors' released code**, not against
a paraphrase (`check_lookback_vs_authors.py`, `check_ptrue_verdict.py`, `check_msp_sign.py`,
`check_determinism.py`). These load a model and are run by hand on a GPU node.

**`*_audit.py`, `*_gate.py`, `*_equivalence.py`** are guards rather than experiments. They assert an
invariant and exit non-zero when it fails: that a cache is teacher-forced, that a refactor changed no
per-example vector, that two shards agree, that no method was scored on a different token set.

## The central drivers

If you only read a few, read these:

- **`probedriftlong.py`** — the long-form ladder. Trains and scores every method on one evaluation
  set across all in-distribution and out-of-distribution rungs, for 3 seeds. Nearly every long-form
  number in the project comes from here.
- **`assemble_pdl_table.py`** — rolls the per-evaluation outputs into the master table.
- **`attn_pool.py`** — the learned attention pooler and its variants (multi-head, prior-initialised,
  fixed-recipe), shared by several experiments.
- **`xl_rungs.py`** — defines the train/eval populations for each rung. Highest blast radius in the
  directory: changing it changes what every ladder means.

## Pre-registrations

A verdict script applies a rule that was fixed in advance. Those rules live in `../../prereg/`, one
file per experiment, each committed **before** the run it describes so that the git timestamp shows
the prediction pre-dated the result. `../../prereg/README.md` indexes them with what each registered
and the outcome the file records.

Some comments here still cite a short code such as `W8`, `S4`, `R0`, `PR4` or `M2`. These were the
working labels while the experiments ran, and they survive where a document refers to its own
sections or to an arm by name. The pre-registration filenames themselves are descriptive, so the
index is the way in.

## Two conventions worth knowing before reading any output

**Absence is never a number.** A missing input raises or leaves the field blank. It is never
silently replaced with a neutral value such as an all-ones mask or a uniform weighting, because that
would turn a method into its own control and manufacture a null. A blank cell means "not measured"; a
zero means "measured and zero".

**The dataset is the unit of analysis.** Aggregate statistics are computed over the 8 evaluation sets
(n = 8), with seeds averaged inside a cell first. The training-free floors do not depend on the
training pool, so their value repeats across the four out-of-distribution rungs. Counting those as 32
independent cells inflates them fourfold, and any test at n = 32 over them is wrong.

## Where the outputs go

All of these write into `results/`, which is **not** part of this repository (see the main README).
Create it with `mkdir -p results` before running anything here.
